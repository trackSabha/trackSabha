#!/usr/bin/env python3
"""
Enhanced Parliamentary Chatbot - MongoDB Session Storage with Graph Visualization
===============================================================================

Updated to use MongoDB for persistent ADK chat history and session management,
now includes interactive knowledge graph visualization.
"""

import os
import json
import logging
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set, AsyncGenerator
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Database and ML imports
from pymongo import MongoClient, ASCENDING

# Google ADK imports
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.adk.planners import BuiltInPlanner
from google.genai.types import Content, Part, GenerateContentConfig
from google.genai import types
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, FOAF, OWL, XSD

# Markdown processing
import markdown
from bs4 import BeautifulSoup
import html

# JSON repair
from json_repair import repair_json

# Mount static files and templates
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pytz

# Import our new session manager
from session_manager import MongoSessionManager, ChatMessage, ChatSession
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve important filesystem paths relative to this file
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
PROMPT_PATH = BASE_DIR / "prompt.md"

# RDF Namespaces
LOK = Namespace("http://example.com/Indian-parliament-ontology#")
SESS = Namespace("http://example.com/Indian-parliament-session/")
SCHEMA = Namespace("http://schema.org/")
ORG = Namespace("http://www.w3.org/ns/org#")
PROV = Namespace("http://www.w3.org/ns/prov#")
BBP = Namespace("http://example.com/TrackSabha.in/")

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question about Parliament")
    user_id: str = Field(..., description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Session ID")

class ResponseCard(BaseModel):
    summary: str = Field(..., description="One-sentence overview of the card's content")
    details: str = Field(..., description="Full, detailed answer with markdown formatting")

class StructuredResponse(BaseModel):
    intro_message: str = Field(..., description="Introductory persona message")
    response_cards: List[ResponseCard] = Field(..., description="Array of expandable cards")
    follow_up_suggestions: List[str] = Field(..., description="Follow-up suggestions")

class QueryResponse(BaseModel):
    session_id: str
    user_id: str
    message_id: str
    status: str
    message: Optional[str] = None
    structured_response: Optional[StructuredResponse] = None

class SessionGraphState:
    """Manages cumulative graph state for a session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.clear_graph("New session started")
        
    def add_turtle_data(self, turtle_str: str) -> bool:
        """Add Turtle data to the cumulative graph."""
        try:
            # Parse Turtle into temporary graph
            temp_graph = Graph()
            temp_graph.parse(data=turtle_str, format='turtle')
            
            # Track what we're adding
            new_triples = len(temp_graph)
            
            # Add to cumulative graph
            for triple in temp_graph:
                self.graph.add(triple)
            
            # Update counts
            self.node_count = len(set(self.graph.subjects()) | set(self.graph.objects()))
            self.edge_count = len(self.graph)
            
            logger.info(f"📈 Session {self.session_id[:8]}: Added {new_triples} triples, total: {self.edge_count}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to parse Turtle: {e}")
            return False
    
    def get_turtle_dump(self) -> str:
        """Get current graph as Turtle format."""
        try:
            header = f"""# Session Graph Dump
# Session: {self.session_id}
# Created: {self.created_at.isoformat()}
# Nodes: {self.node_count}, Edges: {self.edge_count}
"""
            return header + self.graph.serialize(format='turtle')
        except Exception as e:
            logger.error(f"Failed to serialize graph: {e}")
            return f"# Error serializing graph: {e}\n"
    
    def clear_graph(self, reason: str = "Topic change"):
        """Clear the cumulative graph."""
        self.graph = Graph()
        self.created_at = datetime.now(timezone.utc)
        
        # Re-bind namespaces
        self.graph.bind("lok", LOK)
        self.graph.bind("bbp", BBP)
        self.graph.bind("sess", SESS)
        self.graph.bind("schema", SCHEMA)
        self.graph.bind("org", ORG)
        self.graph.bind("prov", PROV)
        self.graph.bind("foaf", FOAF)
        self.graph.bind("owl", OWL)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)
        
        self.node_count = 0
        self.edge_count = 0
        
        logger.info(f"🧹 Session {self.session_id[:8]}: Graph cleared. Reason: {reason}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "session_id": self.session_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "created_at": self.created_at.isoformat(),
            "size_mb": len(self.get_turtle_dump()) / (1024 * 1024)
        }

class ParliamentaryGraphQuerier:
    """Database querier with full search functionality."""

    def __init__(self):
        self.client = None
        self.db = None
        self.nodes = None
        self.edges = None
        self.statements = None
        self.genai_client = None
        self.use_google_embeddings = False
        self.embedding_model_name = "gemini-embedding-001"
        self._initialize_database()
        self._initialize_embeddings()

    def _initialize_database(self):
        """Initialize database connection."""
        connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("MONGODB_CONNECTION_STRING environment variable not set")
        
        try:
            self.client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
                maxPoolSize=3,
                minPoolSize=1,
                retryWrites=True,
                w='majority'
            )
            
            # Test connection
            self.client.admin.command("ping", maxTimeMS=3000)
            
            # Initialize database references
            self.db = self.client["youtube_data"]
            self.nodes = self.db.nodes
            self.edges = self.db.edges
            self.statements = self.db.statements
            
            # Create indexes if they don't exist
            try:
                self.nodes.create_index([("pagerank_score", ASCENDING)])
                self.nodes.create_index([("pagerank_rank", ASCENDING)])
            except:
                pass
            
            logger.info("✅ Connected to MongoDB")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            if self.client:
                self.client.close()
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    def _initialize_embeddings(self):
        """Initialize Gemini embedding client."""
        api_key = (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_GENAI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY, GOOGLE_GENAI_API_KEY, or GEMINI_API_KEY environment variable required for embeddings"
            )

        try:
            from google import genai

            logger.info("🔄 Initializing Gemini embedding client...")
            self.genai_client = genai.Client(api_key=api_key)
            self.use_google_embeddings = True
            logger.info("✅ Vector search enabled via gemini-embedding-001")
        except Exception as e:
            self.genai_client = None
            self.use_google_embeddings = False
            logger.error(f"❌ Failed to initialize Gemini embeddings: {e}")
            raise RuntimeError(f"Vector search failed to initialize: {e}")

    def _generate_query_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the provided text using Gemini."""
        if not self.use_google_embeddings or not self.genai_client:
            raise RuntimeError("Gemini embedding client not initialized")

        clean_text = " ".join(text.split()).strip()
        if not clean_text:
            raise ValueError("Query text required for embedding")

        try:
            response = self.genai_client.models.embed_content(
                model=self.embedding_model_name,
                contents=[clean_text]
            )
        except Exception as e:
            logger.error(f"❌ Gemini embedding request failed: {e}")
            raise RuntimeError(f"Gemini embedding request failed: {e}") from e

        embedding_values = self._extract_embedding_values(response)
        if not embedding_values:
            raise RuntimeError("Gemini embedding response missing values")

        return embedding_values

    @staticmethod
    def _extract_embedding_values(response: Any) -> Optional[List[float]]:
        """Normalize Gemini embedding responses into a flat list of floats."""
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")

        if embeddings is None:
            return None

        candidate = None
        if isinstance(embeddings, (list, tuple)):
            if not embeddings:
                return None
            candidate = embeddings[0]
        else:
            candidate = embeddings

        if candidate is None:
            return None

        values = getattr(candidate, "values", None)
        if isinstance(values, (list, tuple)):
            try:
                return [float(v) for v in values]
            except Exception:
                return None

        if isinstance(candidate, dict):
            for key in ("embedding", "value", "vector", "values"):
                raw = candidate.get(key)
                if isinstance(raw, (list, tuple)):
                    try:
                        return [float(v) for v in raw]
                    except Exception:
                        return None

        attr_embedding = getattr(candidate, "embedding", None)
        if isinstance(attr_embedding, (list, tuple)):
            try:
                return [float(v) for v in attr_embedding]
            except Exception:
                return None

        if isinstance(candidate, (list, tuple)):
            try:
                return [float(v) for v in candidate]
            except Exception:
                return None

        try:
            return [float(v) for v in candidate]  # type: ignore[arg-type]
        except Exception:
            return None

    def unified_hybrid_search(self, query: str, limit: int = 8) -> List[Dict]:
        """
        Performs both node vector search and statement text search,
        then intelligently combines and weights the results.
        """
        try:
            logger.info(f"🔍 Unified search for: '{query}'")
            
            # Run both searches in parallel
            try:
                node_results = self._search_nodes_vector(query, limit * 2)
            except Exception as e:
                logger.error(f"❌ Vector search failed: {e}")
                node_results = []
            try:
                statement_results = self._search_statements_atlas(query, limit * 2)
            except Exception as e:
                logger.error(f"❌ Atlas search failed: {e}")
                statement_results = [] 

            # Convert both result types to unified format
            unified_results = []
            
            # Process node results
            for node in node_results:
                unified_results.append({
                    'uri': node['uri'],
                    'source_type': 'node',
                    'content': node.get('searchable_text', ''),
                    'label': node.get('label') or node.get('name', ''),
                    'node_data': node,
                    'vector_score': node.get('similarity_score', 0),
                    'text_score': 0,
                    'provenance': None
                })
            
            # Process statement results and find their related nodes
            if statement_results:
                # Step 1: Collect all unique URIs from all statements
                all_related_uris = set()
                stmt_to_uris = {}  # Track which URIs belong to which statement
                
                for i, stmt in enumerate(statement_results):
                    related_uris = []
                    if stmt.get('subject'): related_uris.append(stmt['subject'])
                    if stmt.get('object'): related_uris.append(stmt['object'])
                    
                    stmt_to_uris[i] = related_uris
                    all_related_uris.update(related_uris)
                
                # Step 2: Fetch ALL related nodes in ONE database call
                if all_related_uris:
                    nodes_cursor = self.nodes.find(
                        {'uri': {'$in': list(all_related_uris)}},
                        {'uri': 1, 'label': 1, 'name': 1}  # Only fetch needed fields
                    )
                    
                    # Create a lookup dictionary for O(1) access
                    uri_to_node = {node['uri']: node for node in nodes_cursor}
                    
                    # Step 3: Build results using the lookup dictionary
                    for i, stmt in enumerate(statement_results):
                        for uri in stmt_to_uris[i]:
                            node = uri_to_node.get(uri)
                            if node:
                                unified_results.append({
                                    'uri': uri,
                                    'source_type': 'statement',
                                    'content': stmt.get('transcript_text', ''),
                                    'label': node.get('label') or node.get('name', ''),
                                    'node_data': node,
                                    'vector_score': 0,
                                    'text_score': stmt.get('score', 0),
                                    'provenance': {
                                        'video_id': stmt.get('source_video'),
                                        'video_title': stmt.get('video_title'),
                                        'start_time': stmt.get('start_offset'),
                                        'transcript_excerpt': stmt.get('transcript_text', '')[:200] + '...'
                                    }
                                })
            
            # Deduplicate by URI while preserving best scores
            uri_to_result = {}
            for result in unified_results:
                uri = result['uri']
                if uri not in uri_to_result:
                    uri_to_result[uri] = result
                else:
                    # Merge scores - keep highest of each type
                    existing = uri_to_result[uri]
                    existing['vector_score'] = max(existing['vector_score'], result['vector_score'])
                    existing['text_score'] = max(existing['text_score'], result['text_score'])
                    
                    # Prefer statement provenance if available
                    if result['provenance'] and not existing['provenance']:
                        existing['provenance'] = result['provenance']
                        existing['content'] = result['content']  # Use transcript text
            
            # Calculate unified scores and rank
            final_results = list(uri_to_result.values())
            final_results = self._calculate_unified_scores(final_results, query)
            
            # Sort by unified score and return top results
            final_results.sort(key=lambda x: x['unified_score'], reverse=True)
            
            logger.info(f"🎯 Unified search: {len(final_results)} unique results")
            return final_results[:limit]
            
        except Exception as e:
            logger.error(f"❌ Unified search failed: {e}")
            return []

    def _search_nodes_vector(self, query: str, limit: int) -> List[Dict]:
        """Vector search on nodes using Gemini embeddings."""
        query_embedding = self._generate_query_embedding(query)
        
        pipeline = [
            {"$vectorSearch": {
                "index": "vector_index_1",
                "path": "embedding", 
                "queryVector": query_embedding,
                "numCandidates": limit * 4,
                "limit": limit
            }},
            {"$addFields": {
                "similarity_score": {"$meta": "vectorSearchScore"}
            }}
        ]
        
        return list(self.nodes.aggregate(pipeline))

    def _search_statements_atlas(self, query: str, limit: int) -> List[Dict]:
        """Atlas Search on statements - much better than $text"""
        pipeline = [
            {
                "$search": {
                    "index": "default",
                    "compound": {
                        "should": [
                            {
                                "phrase": {
                                    "query": query,
                                    "path": "transcript_text",
                                    "score": {"boost": {"value": 3}}
                                }
                            },
                            {
                                "text": {
                                    "query": query,
                                    "path": ["transcript_text", "video_title"],
                                    "fuzzy": {"maxEdits": 1}
                                }
                            }
                        ]
                    }
                }
            },
            {
                "$addFields": {
                    "search_score": {"$meta": "searchScore"}
                }
            },
            {
                "$project": {
                    "subject": 1, "object": 1, "transcript_text": 1,
                    "source_video": 1, "video_title": 1, "start_offset": 1,
                    "search_score": 1
                }
            },
            {"$sort": {"search_score": -1}},
            {"$limit": limit}
        ]
        
        return list(self.statements.aggregate(pipeline))

    def _calculate_unified_scores(self, results: List[Dict], query: str) -> List[Dict]:
        """Calculate unified scores using multiple factors"""
        
        # Normalize scores to 0-1 range
        max_vector = max((r['vector_score'] for r in results), default=1)
        max_text = max((r['text_score'] for r in results), default=1)
        
        for result in results:
            # Normalize individual scores
            norm_vector = result['vector_score'] / max_vector if max_vector > 0 else 0
            norm_text = result['text_score'] / max_text if max_text > 0 else 0
            
            # Dynamic weighting based on query characteristics
            weights = self._get_dynamic_weights(query, result)
            
            # Base score combination
            base_score = (
                weights['vector_weight'] * norm_vector + 
                weights['text_weight'] * norm_text
            )
            
            # Boost factors
            pagerank_boost = self._get_pagerank_boost(result['node_data'])
            provenance_boost = self._get_provenance_boost(result['provenance'])
            content_quality_boost = self._get_content_quality_boost(result['content'])
            
            # Final unified score
            result['unified_score'] = base_score * (1 + pagerank_boost + provenance_boost + content_quality_boost)
            
            # Store components for debugging
            result['score_components'] = {
                'base_score': base_score,
                'norm_vector': norm_vector,
                'norm_text': norm_text,
                'weights': weights,
                'pagerank_boost': pagerank_boost,
                'provenance_boost': provenance_boost,
                'content_quality_boost': content_quality_boost
            }
        
        return results

    def _get_dynamic_weights(self, query: str, result: Dict) -> Dict:
        """Dynamically adjust vector vs text weights based on query and result characteristics"""
        # Default weights
        vector_weight = 0.6
        text_weight = 0.4
        
        # Adjust based on query characteristics
        query_lower = query.lower()
        
        # Favor text search for:
        if any(indicator in query_lower for indicator in [
            'said', 'stated', 'mentioned', 'quote', 'exactly',
            '$', 'bbd', 'usd', 'payment', 'amount', 'cost',
            'bill', 'section', 'act', 'regulation'
        ]):
            text_weight += 0.3
            vector_weight -= 0.3
        
        # Favor vector search for:
        if any(indicator in query_lower for indicator in [
            'about', 'regarding', 'related to', 'concerning',
            'policy', 'strategy', 'approach', 'similar'
        ]):
            vector_weight += 0.2
            text_weight -= 0.2
        
        # Boost text weight if we have good provenance
        if result.get('provenance'):
            text_weight += 0.1
            vector_weight -= 0.1
        
        # Normalize to ensure they sum to 1
        total = vector_weight + text_weight
        return {
            'vector_weight': vector_weight / total,
            'text_weight': text_weight / total
        }

    def _get_pagerank_boost(self, node_data: Dict) -> float:
        pagerank_rank = node_data.get('pagerank_rank', 1000)
        # Convert rank to boost: top 10 nodes get big boost
        if pagerank_rank <= 10:
            return 0.5  # 50% boost for top 10
        elif pagerank_rank <= 50:
            return 0.3  # 30% boost for top 50
        elif pagerank_rank <= 100:
            return 0.1  # 10% boost for top 100
        else:
            return 0.0

    def _get_provenance_boost(self, provenance: Dict) -> float:
        """Boost if we have good provenance (video links, timestamps)"""
        if not provenance:
            return 0
        
        boost = 0
        if provenance.get('video_id'): boost += 0.1
        if provenance.get('start_time'): boost += 0.1
        if provenance.get('transcript_excerpt'): boost += 0.1
        
        return min(0.3, boost)

    def _get_content_quality_boost(self, content: str) -> float:
        """Boost based on content richness"""
        if not content:
            return 0
        
        # Simple content quality indicators
        word_count = len(content.split())
        boost = min(0.1, word_count / 1000)  # Up to 0.1 boost for rich content
        
        return boost

    def get_connected_nodes(self, uris: Set[str], hops: int = 1) -> Set[str]:
        """Get nodes connected to the given URIs."""
        try:
            current, seen = set(uris), set(uris)
            for hop in range(max(0, hops)):
                if not current or len(seen) > 500:
                    break
                    
                edges = self.edges.find({
                    "$or": [
                        {"subject": {"$in": list(current)}},
                        {"object": {"$in": list(current)}},
                    ]
                })
                
                nxt = set()
                for edge in edges:
                    nxt.add(edge["subject"])
                    nxt.add(edge["object"])
                
                current = nxt - seen
                seen.update(nxt)
                    
            return seen
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return uris

    def get_subgraph(self, uris: Set[str]) -> Dict[str, Any]:
        """Get subgraph for the given URIs."""
        try:
            if len(uris) > 500:
                uris = set(list(uris)[:500])
            
            # Get nodes
            raw_nodes = list(self.nodes.find(
                {"uri": {"$in": list(uris)}}, 
                {
                    "uri": 1,
                    "label": 1,
                    "name": 1,
                    "type": 1,
                    "searchable_text": 1
                }
            ))
            
            # Clean nodes
            cleaned_nodes = []
            for node in raw_nodes:
                cleaned = {
                    "uri": node.get("uri"),
                    "type": node.get("type", [])
                }
                
                # Handle labels - prefer label over name
                label = node.get("label") or node.get("name")
                if label:
                    cleaned["label"] = label
                
                if "searchable_text" in node:
                    cleaned["searchable_text"] = node["searchable_text"]
                
                cleaned_nodes.append(cleaned)
            
            # Get edges
            edges = list(self.edges.find({
                "subject": {"$in": list(uris)}, 
                "object": {"$in": list(uris)}
            }))
            
            # Clean edges
            for edge in edges:
                if "_id" in edge:
                    edge["_id"] = str(edge["_id"])
            
            return {"nodes": cleaned_nodes, "edges": edges}
            
        except Exception as e:
            logger.error(f"Subgraph retrieval failed: {e}")
            return {"nodes": [], "edges": []}

    def to_turtle(self, subgraph: Dict[str, Any]) -> str:
        """Convert subgraph to Turtle format."""
        try:
            g = Graph()
            
            # Add prefixes
            g.bind("lok", LOK)
            g.bind("bbp", BBP)
            g.bind("sess", SESS)
            g.bind("schema", SCHEMA)
            g.bind("org", ORG)
            g.bind("prov", PROV)
            g.bind("foaf", FOAF)
            g.bind("owl", OWL)
            g.bind("rdf", RDF)
            g.bind("rdfs", RDFS)
            g.bind("xsd", XSD)
            
            # Add nodes
            for node in subgraph["nodes"]:
                try:
                    uri = URIRef(node["uri"])
                    
                    if "label" in node and node["label"]:
                        g.add((uri, RDFS.label, Literal(str(node["label"]))))
                    
                    for t in node.get("type", []):
                        g.add((uri, RDF.type, URIRef(t)))
                    
                except Exception as e:
                    logger.warning(f"Skipping node: {e}")
            
            # Add edges
            for edge in subgraph["edges"]:
                try:
                    g.add((
                        URIRef(edge["subject"]),
                        URIRef(edge["predicate"]),
                        URIRef(edge["object"]) if edge["object"].startswith("http") else Literal(edge["object"])
                    ))
                except Exception as e:
                    logger.warning(f"Skipping edge: {e}")
            
            header = f"# Generated {datetime.now(timezone.utc).isoformat()}Z\n"
            header += f"# Nodes: {len(subgraph['nodes'])}, Edges: {len(subgraph['edges'])}\n\n"
            
            return header + g.serialize(format="turtle")
            
        except Exception as e:
            logger.error(f"Turtle serialization failed: {e}")
            return f"# Error: {str(e)}\n"

    def get_provenance_turtle(self, node_uris: List[str], include_transcript: bool = True) -> str:
        """Get provenance information as Turtle format."""
        try:
            logger.info(f"📚 Getting provenance for {len(node_uris)} nodes")
            
            g = Graph()
            g.bind("lok", LOK)
            g.bind("bbp", BBP)
            g.bind("sess", SESS)
            g.bind("schema", SCHEMA)
            g.bind("org", ORG)
            g.bind("prov", PROV)
            g.bind("foaf", FOAF)
            g.bind("owl", OWL)
            g.bind("rdf", RDF)
            g.bind("rdfs", RDFS)
            g.bind("xsd", XSD)
            
            for uri in node_uris[:10]:  # Limit to prevent explosion
                try:
                    node_uri = URIRef(uri)
                    
                    # Get related statements
                    projection = {
                        "subject": 1,
                        "predicate": 1, 
                        "object": 1,
                        "source_video": 1,
                        "video_title": 1,
                        "start_offset": 1,
                        "end_offset": 1
                    }
                    
                    if include_transcript:
                        projection["transcript_text"] = 1
                    
                    statements = list(self.statements.find({
                        "$or": [
                            {"subject": uri},
                            {"predicate": uri}, 
                            {"object": uri}
                        ]
                    }, projection))
                    
                    # Process statements
                    for i, stmt in enumerate(statements[:5]):
                        stmt_uri = URIRef(f"{uri}/statement/{i}")
                        
                        # Basic provenance
                        g.add((stmt_uri, RDF.type, PROV.Entity))
                        g.add((stmt_uri, PROV.wasDerivedFrom, node_uri))
                        g.add((stmt_uri, SCHEMA.about, node_uri))
                        
                        # Video information
                        video_id = stmt.get("source_video")
                        video_title = stmt.get("video_title")
                        start_time = stmt.get("start_offset")
                        
                        if video_id:
                            if start_time is not None:
                                timestamped_url = f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"
                            else:
                                timestamped_url = f"https://www.youtube.com/watch?v={video_id}"
                            
                            g.add((stmt_uri, SCHEMA.url, Literal(timestamped_url)))
                            
                            if video_title:
                                g.add((stmt_uri, SCHEMA.videoTitle, Literal(video_title)))
                        
                        if start_time is not None:
                            g.add((stmt_uri, SCHEMA.startTime, Literal(int(start_time))))
                        
                        # Transcript text
                        if include_transcript and "transcript_text" in stmt:
                            transcript = stmt["transcript_text"]
                            if transcript and len(transcript.strip()) > 0:
                                if len(transcript) > 1000:
                                    transcript = transcript[:1000] + "..."
                                g.add((stmt_uri, SCHEMA.text, Literal(transcript)))
                        
                except Exception as e:
                    logger.warning(f"Skipping provenance for {uri}: {e}")
            
            header = f"# Provenance information generated {datetime.now(timezone.utc).isoformat()}Z\n\n"
            return header + g.serialize(format="turtle")
            
        except Exception as e:
            logger.error(f"❌ Provenance turtle generation failed: {e}")
            return f"# Error: {str(e)}\n"

    def close(self):
        """Close database connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()

def parse_llm_json_response(response_text: str) -> Optional[StructuredResponse]:
    """Parse LLM response as JSON with automatic repair and validation."""
    try:
        logger.info(f"🔧 Parsing LLM response, length: {len(response_text)}")
        
        # Try to find JSON in the response
        response_text = response_text.strip()
        
        # Look for JSON block markers
        json_text = None
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text[start:].strip()
        elif response_text.startswith("{"):
            # Direct JSON response
            json_text = response_text
        else:
            # Try to find JSON object in the text
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_text = response_text[start:end]
            else:
                logger.error("No JSON found in response")
                return None
        
        if not json_text:
            logger.error("Could not extract JSON from response")
            return None
        
        # Use json-repair which handles both valid and malformed JSON automatically
        try:
            parsed_data = repair_json(json_text, return_objects=True)
            logger.info("✅ JSON parsed successfully with json-repair")
        except Exception as e:
            logger.error(f"❌ JSON repair failed: {e}")
            return None
        
        if not parsed_data:
            logger.error("Failed to get valid JSON data")
            return None
        
        # Validate required fields
        required_fields = ["intro_message", "response_cards", "follow_up_suggestions"]
        missing_fields = [field for field in required_fields if field not in parsed_data]
        
        if missing_fields:
            logger.error(f"Missing required fields in JSON response: {missing_fields}")
            return None
        
        # Validate response_cards structure
        if not isinstance(parsed_data["response_cards"], list):
            logger.error("response_cards must be a list")
            return None
        
        if not parsed_data["response_cards"]:
            logger.warning("response_cards is empty, adding default card")
            parsed_data["response_cards"] = [{
                "summary": "Parliamentary information found",
                "details": "No specific details were provided in the response."
            }]
        
        # Validate each card
        for i, card in enumerate(parsed_data["response_cards"]):
            if not isinstance(card, dict):
                logger.error(f"Card {i} is not a dictionary")
                return None
            
            if "summary" not in card:
                logger.warning(f"Card {i} missing summary, adding default")
                card["summary"] = f"Information card {i + 1}"
            
            if "details" not in card:
                logger.warning(f"Card {i} missing details, adding default")
                card["details"] = "No details provided."
        
        # Validate follow_up_suggestions
        if not isinstance(parsed_data["follow_up_suggestions"], list):
            logger.warning("follow_up_suggestions is not a list, creating default")
            parsed_data["follow_up_suggestions"] = [
                "Tell me more about this topic",
                "What are the latest developments?",
                "Who are the key people involved?"
            ]
        
        # Ensure we have at least some follow-up suggestions
        if not parsed_data["follow_up_suggestions"]:
            parsed_data["follow_up_suggestions"] = [
                "What else would you like to know?",
                "Any other parliamentary questions?"
            ]
        
        # Create structured response
        try:
            return StructuredResponse(
                intro_message=parsed_data["intro_message"],
                response_cards=[
                    ResponseCard(summary=card["summary"], details=card["details"])
                    for card in parsed_data["response_cards"]
                ],
                follow_up_suggestions=parsed_data["follow_up_suggestions"]
            )
        except Exception as validation_error:
            logger.error(f"Failed to create StructuredResponse: {validation_error}")
            return None
        
    except Exception as e:
        logger.error(f"❌ Unexpected error parsing LLM response: {e}")
        return None

def convert_structured_response_to_html(structured_response: StructuredResponse, message_id: str = None) -> str:
    """Convert structured response to HTML with expandable cards."""
    try:
        html_parts = []
        
        # Generate unique message ID if not provided
        if not message_id:
            import time
            message_id = f"msg-{int(time.time() * 1000)}"
        
        # Intro message
        intro_html = markdown.markdown(structured_response.intro_message, extensions=['extra', 'codehilite'])
        html_parts.append(f'<div class="intro-message">{intro_html}</div>')
        
        # Response cards
        html_parts.append('<div class="response-cards">')
        
        for i, card in enumerate(structured_response.response_cards):
            # Use message_id to ensure unique card IDs across all messages
            card_id = f"{message_id}-card-{i}"
            
            # Convert details markdown to HTML
            details_html = markdown.markdown(card.details, extensions=['extra', 'codehilite'])
            
            # Filter out non-YouTube links from details
            soup = BeautifulSoup(details_html, 'html.parser')
            links = soup.find_all('a')
            for link in links:
                href = link.get('href', '')
                if 'youtube.com' not in href.lower() and 'youtu.be' not in href.lower():
                    link.replace_with(link.get_text())
            details_html = str(soup)
            
            card_html = f'''
            <div class="response-card" data-card-id="{card_id}">
                <div class="card-header" onclick="toggleCard('{card_id}')">
                    <div class="card-summary">{card.summary}</div>
                    <div class="card-toggle">
                        <span class="toggle-icon">▼</span>
                    </div>
                </div>
                <div class="card-details collapsed" id="{card_id}-details">
                    <div class="card-content">
                        {details_html}
                    </div>
                </div>
            </div>
            '''
            html_parts.append(card_html)
        
        html_parts.append('</div>')
        
        # Follow-up suggestions
        html_parts.append('<div class="follow-up-suggestions">')
        html_parts.append('<h4>Follow-up questions:</h4>')
        html_parts.append('<ul class="suggestions-list">')
        
        for suggestion in structured_response.follow_up_suggestions:
            safe_text = html.escape(str(suggestion))
            safe_attr = html.escape(str(suggestion), quote=True)
            html_parts.append(
                f'<li class="suggestion-item" data-suggestion="{safe_attr}" onclick="sendSuggestion(this.dataset.suggestion)">{safe_text}</li>'
            )
        
        html_parts.append('</ul>')
        html_parts.append('</div>')
        
        return ''.join(html_parts)
        
    except Exception as e:
        logger.error(f"Error converting structured response to HTML: {e}")
        # Fallback to simple display
        return f'<div class="error-fallback">Error displaying response: {str(e)}</div>'

# Updated ParliamentarySystem class with graph visualization
class ParliamentarySystem:
    """Main parliamentary system using MongoDB session management with graph visualization."""
    
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        self.querier = ParliamentaryGraphQuerier()
        
        # Initialize MongoDB session manager
        self.session_manager = MongoSessionManager()
        
        # Keep graph memory separate from chat history (in memory for now)
        self.session_graphs = {}  # Store SessionGraphState by session_id
        
        # Track current session for tool context
        self.current_session_id = None
        
        # Create enhanced search tools with session context
        def search_parliament_hybrid(query: str, hops: int = 2, limit: int = 5) -> str:
            """
            Search parliamentary records using hybrid search with session graph integration.
            
            Args:
                query: Search query for parliamentary information
                hops: Number of relationship hops to explore (1-3)
                limit: Maximum number of results (1-10)
            
            Returns:
                Parliamentary data in Turtle format with facts and relationships
            """
            try:
                logger.info(f"🔍 Searching parliament: {query}")
                
                # Perform hybrid search
                seeds = self.querier.unified_hybrid_search(query, limit)
                if not seeds:
                    return f"# No parliamentary data found for: {query}\n"
                
                # Get connected nodes
                seed_uris = {node["uri"] for node in seeds if "uri" in node}
                all_uris = self.querier.get_connected_nodes(seed_uris, hops)
                
                # Get subgraph and convert to turtle
                subgraph = self.querier.get_subgraph(all_uris)
                turtle_data = self.querier.to_turtle(subgraph)
                
                # Get provenance data
                provenance_data = ""
                if seed_uris:
                    provenance_turtle = self.querier.get_provenance_turtle(list(seed_uris)[:5])
                    provenance_data = f"\n\n# PROVENANCE DATA:\n{provenance_turtle}"
                
                combined_data = turtle_data + provenance_data
                
                # Update session graph if we have session context
                if self.current_session_id:
                    try:
                        session_graph = self.get_or_create_session_graph(self.current_session_id)
                        main_data = turtle_data.split("# PROVENANCE DATA:")[0].strip()
                        
                        if main_data and not main_data.startswith("# Error"):
                            session_graph.add_turtle_data(main_data)
                            logger.info(f"📈 Updated session graph: {session_graph.get_stats()}")
                    except Exception as e:
                        logger.warning(f"Failed to update session graph: {e}")
                
                logger.info(f"🎯 Found {len(subgraph.get('nodes', []))} nodes, {len(subgraph.get('edges', []))} edges")
                
                return combined_data
                
            except Exception as e:
                logger.error(f"Parliament search failed: {e}")
                return f"# Error searching parliament: {str(e)}\n"
        
        def clear_session_graph(reason: str = "Topic change detected") -> str:
            """
            Clear the cumulative session graph when topic changes significantly.
            
            Args:
                reason: Why you're clearing the session graph
            
            Returns:
                Confirmation message with graph statistics
            """
            if self.current_session_id:
                try:
                    session_graph = self.get_or_create_session_graph(self.current_session_id)
                    old_stats = session_graph.get_stats()
                    session_graph.clear_graph(reason)
                    logger.info(f"🧹 Session graph cleared: {reason}")
                    return f"Session graph cleared successfully. Previous state: {old_stats['edge_count']} edges. Reason: {reason}"
                except Exception as e:
                    logger.error(f"Failed to clear session graph: {e}")
                    return f"Error clearing session graph: {e}"
            else:
                return "No active session to clear"
        
        def get_session_graph_stats() -> str:
            """
            Get statistics about the current session's cumulative graph.
            
            Returns:
                JSON string with graph statistics
            """
            if self.current_session_id:
                try:
                    session_graph = self.get_or_create_session_graph(self.current_session_id)
                    stats = session_graph.get_stats()
                    return json.dumps(stats, indent=2)
                except Exception as e:
                    logger.error(f"Failed to get session stats: {e}")
                    return json.dumps({"error": str(e)})
            else:
                return json.dumps({"error": "No active session"})
        
        def visualize_knowledge_graph(reason: str = "User requested visualization") -> str:
            """
            Visualize the current session's knowledge graph as an interactive network.
            
            Args:
                reason: Why you're showing the graph (optional)
            
            Returns:
                Interactive HTML graph visualization
            """
            try:
                if not self.current_session_id:
                    return "No active session to visualize. Start a conversation first!"
                
                logger.info(f"📊 Generating graph visualization: {reason}")
                html_viz = self.visualize_session_graph(self.current_session_id)
                
                return html_viz
                
            except Exception as e:
                logger.error(f"Failed to visualize graph: {e}")
                return f"Error generating graph visualization: {e}"
        
        bb_timezone = pytz.timezone("America/Barbados")
        current_date = datetime.now(bb_timezone).strftime("%Y-%m-%d")

        # Read and format the prompt safely
        try:
            with open(PROMPT_PATH, "r", encoding="utf-8") as f:
                prompt_content = f.read()
            
            # Replace only the specific current_date placeholder, not any other braces
            formatted_prompt = prompt_content.replace("{current_date}", current_date)
            logger.info("✅ Prompt loaded and formatted successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load prompt.md: {e}")
            # Fallback prompt
            formatted_prompt = f"""You are TrackSabha, an AI assistant for Indian Parliament. 
Current Date: {current_date}

You must ALWAYS respond with valid JSON in this exact format:
{{
  "intro_message": "Your introductory message here",
  "response_cards": [
    {{
      "summary": "Brief one-sentence summary",
      "details": "Full detailed response with markdown formatting"
    }}
  ],
  "follow_up_suggestions": [
    "Follow-up question 1",
    "Follow-up question 2", 
    "Follow-up question 3"
  ]
}}

Search for parliamentary information when asked about topics, ministers, debates, or policies."""

        # Create the main agent with enhanced tools including graph visualization
        self.agent = LlmAgent(
            name="TrackSabha",
            model="gemini-flash-latest",
            description="AI assistant for Barbados Parliament with cumulative graph memory and visualization",
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=0)),
            instruction=formatted_prompt,
            tools=[
                FunctionTool(search_parliament_hybrid),
                FunctionTool(clear_session_graph),
                FunctionTool(get_session_graph_stats),
                FunctionTool(visualize_knowledge_graph)
            ],
            generate_content_config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=5000
            )
        )
        
        # Use simpler session approach - just create sessions when we need them
        from google.adk.sessions import InMemorySessionService
        self.adk_session_service = InMemorySessionService()
        
        from google.adk.runners import Runner
        self.runner = Runner(
            agent=self.agent,
            session_service=self.adk_session_service,
            app_name="TrackSabha"
        )
    
    def get_or_create_session_graph(self, session_id: str) -> SessionGraphState:
        """Get or create session graph state."""
        if session_id not in self.session_graphs:
            self.session_graphs[session_id] = SessionGraphState(session_id)
            logger.info(f"📊 Created new session graph for {session_id[:8]}")
        return self.session_graphs[session_id]
    
    def visualize_session_graph(self, session_id: str = None, max_nodes: int = 500) -> str:
        """
        Visualize the current session graph as an interactive network with better connectivity.
        
        Args:
            session_id: Session to visualize (uses current if None)
            max_nodes: Maximum nodes to display for performance
        
        Returns:
            HTML with embedded D3.js visualization
        """
        try:
            target_session = session_id or self.current_session_id
            if not target_session:
                return "# Error: No active session to visualize\n"
            
            if target_session not in self.session_graphs:
                return "# Error: Session graph not found\n"
            
            session_graph = self.session_graphs[target_session]
            
            if session_graph.edge_count == 0:
                return "# Session graph is empty - start a conversation to build the knowledge graph!\n"
            
            # Extract all nodes and their metadata from the RDF graph
            nodes_data = {}
            all_edges = []
            node_connections = {}  # Track connection counts for each node
            
            # First pass: collect all node URIs, properties, and count connections
            for subject, predicate, obj in session_graph.graph:
                subj_uri = str(subject)
                pred_uri = str(predicate)
                obj_uri = str(obj) if str(obj).startswith("http") else None
                
                # Initialize node data if not exists
                if subj_uri not in nodes_data:
                    nodes_data[subj_uri] = {
                        "uri": subj_uri,
                        "id": self._extract_display_name(subj_uri),
                        "label": None,
                        "name": None,
                        "type": None,
                        "properties": {},
                        "connection_count": 0
                    }
                
                if obj_uri and obj_uri not in nodes_data:
                    nodes_data[obj_uri] = {
                        "uri": obj_uri,
                        "id": self._extract_display_name(obj_uri),
                        "label": None,
                        "name": None,
                        "type": None,
                        "properties": {},
                        "connection_count": 0
                    }
                
                # Collect properties for subject node
                if pred_uri.endswith("/label") or pred_uri.endswith("#label") or "label" in pred_uri.lower():
                    nodes_data[subj_uri]["label"] = str(obj)
                elif pred_uri.endswith("/name") or pred_uri.endswith("#name") or "name" in pred_uri.lower():
                    nodes_data[subj_uri]["name"] = str(obj)
                elif pred_uri.endswith("/type") or pred_uri.endswith("#type") or "type" in pred_uri.lower():
                    nodes_data[subj_uri]["type"] = str(obj)
                else:
                    # Store other properties
                    prop_name = self._extract_display_name(pred_uri)
                    nodes_data[subj_uri]["properties"][prop_name] = str(obj)
                
                # Count connections (both incoming and outgoing)
                if obj_uri:
                    nodes_data[subj_uri]["connection_count"] += 1
                    nodes_data[obj_uri]["connection_count"] += 1
                    
                    all_edges.append({
                        "source_uri": subj_uri,
                        "target_uri": obj_uri,
                        "source": self._extract_display_name(subj_uri),
                        "target": self._extract_display_name(obj_uri),
                        "label": self._extract_display_name(pred_uri),
                        "predicate": pred_uri
                    })
            
            # Calculate node importance score combining properties and connections
            for uri, node_data in nodes_data.items():
                property_score = len(node_data["properties"]) * 2  # Properties are valuable
                connection_score = node_data["connection_count"]   # Connections show importance
                node_data["importance_score"] = property_score + connection_score
            
            # Smart node selection: prioritize connected nodes and important hubs
            all_nodes = list(nodes_data.values())
            
            # Sort by importance score (properties + connections)
            all_nodes.sort(key=lambda x: x["importance_score"], reverse=True)
            
            if len(all_nodes) <= max_nodes:
                # If we can show all nodes, do it
                selected_nodes = all_nodes
            else:
                # Smart selection: ensure we keep well-connected nodes
                selected_nodes = []
                selected_uris = set()
                
                # First, take the most important nodes
                for node in all_nodes[:max_nodes // 2]:
                    selected_nodes.append(node)
                    selected_uris.add(node["uri"])
                
                # Then, add nodes that connect to already selected nodes
                remaining_budget = max_nodes - len(selected_nodes)
                connection_candidates = []
                
                for edge in all_edges:
                    # If one end is selected but the other isn't, consider the unselected one
                    if edge["source_uri"] in selected_uris and edge["target_uri"] not in selected_uris:
                        target_node = nodes_data[edge["target_uri"]]
                        if target_node not in connection_candidates:
                            connection_candidates.append(target_node)
                    elif edge["target_uri"] in selected_uris and edge["source_uri"] not in selected_uris:
                        source_node = nodes_data[edge["source_uri"]]
                        if source_node not in connection_candidates:
                            connection_candidates.append(source_node)
                
                # Sort connection candidates by importance and add the best ones
                connection_candidates.sort(key=lambda x: x["importance_score"], reverse=True)
                
                for node in connection_candidates[:remaining_budget]:
                    if node not in selected_nodes:
                        selected_nodes.append(node)
                        selected_uris.add(node["uri"])
            
            # Convert to final node format with rich labels
            final_nodes = []
            for node_data in selected_nodes:
                # Determine the best display label
                display_label = (
                    node_data["label"] or 
                    node_data["name"] or 
                    node_data["id"]
                )
                
                # Clean up the label (remove quotes, limit length)
                if display_label and display_label.startswith('"') and display_label.endswith('"'):
                    display_label = display_label[1:-1]
                
                # Truncate very long labels
                if display_label and len(display_label) > 50:
                    display_label = display_label[:47] + "..."
                
                node = {
                    "id": node_data["id"],
                    "uri": node_data["uri"],
                    "label": display_label or node_data["id"],
                    "original_label": node_data["label"],
                    "name": node_data["name"],
                    "type": node_data["type"],
                    "properties": node_data["properties"],
                    "connection_count": node_data["connection_count"],
                    "importance_score": node_data["importance_score"]
                }
                final_nodes.append(node)
            
            # Filter edges to only include selected nodes
            selected_node_ids = {node["id"] for node in final_nodes}
            final_edges = [
                edge for edge in all_edges 
                if edge["source"] in selected_node_ids and edge["target"] in selected_node_ids
            ]
            
            # Remove duplicate edges (can happen with bidirectional relationships)
            unique_edges = []
            seen_edges = set()
            for edge in final_edges:
                # Create a normalized edge key for deduplication
                edge_key = tuple(sorted([edge["source"], edge["target"]]) + [edge["predicate"]])
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    unique_edges.append(edge)
            
            # Load and render template
            from jinja2 import Environment, FileSystemLoader
            env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
            template = env.get_template('graph_visualization.html')
            
            html_content = template.render(
                nodes=final_nodes,
                edges=unique_edges,
                stats=session_graph.get_stats()
            )
            
            logger.info(f"📊 Generated graph visualization: {len(final_nodes)} nodes, {len(unique_edges)} edges")
            logger.info(f"📊 Node selection: {len(selected_nodes)} selected from {len(all_nodes)} total")
            
            return html_content
            
        except Exception as e:
            logger.error(f"❌ Graph visualization failed: {e}")
            return f"# Error generating graph visualization: {str(e)}\n"

    def _extract_display_name(self, uri: str) -> str:
        """Extract a human-readable name from a URI."""
        if not uri.startswith("http"):
            return uri[:50]  # Literal value, truncate if long
        
        # Extract the fragment or last path component
        if "#" in uri:
            return uri.split("#")[-1]
        elif "/" in uri:
            return uri.split("/")[-1]
        else:
            return uri

    def _get_node_type(self, uri: str) -> str:
        """Determine node type from URI for color coding."""
        uri_lower = uri.lower()
        
        if "person" in uri_lower or "mp" in uri_lower or "minister" in uri_lower:
            return "person"
        elif "bill" in uri_lower or "act" in uri_lower or "legislation" in uri_lower:
            return "legislation"
        elif "committee" in uri_lower or "parliament" in uri_lower:
            return "institution"
        elif "topic" in uri_lower or "policy" in uri_lower:
            return "topic"
        else:
            return "entity"
    
    async def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> Tuple[str, str]:
        """Get existing session or create a new one using MongoDB."""
        try:
            # If session_id provided, try to get existing session
            if session_id:
                existing_session = await self.session_manager.get_session(session_id)
                if existing_session:
                    logger.info(f"✅ Using existing session: {session_id[:8]}...")
                    
                    # Create a fresh ADK session for this interaction
                    adk_session = await self.adk_session_service.create_session(
                        app_name="TrackSabha",
                        user_id=user_id
                    )
                    
                    return session_id, adk_session.id
            
            # Create new session
            new_session = await self.session_manager.create_session(
                user_id=user_id,
                session_id=session_id,
                metadata={"created_via": "parliamentary_system", "version": "3.6.0"}
            )
            
            # Create corresponding ADK session
            adk_session = await self.adk_session_service.create_session(
                app_name="TrackSabha",
                user_id=user_id
            )
            
            logger.info(f"✅ Created new session: {new_session.session_id[:8]}...")
            return new_session.session_id, adk_session.id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            # Fallback to simple UUID
            fallback_id = str(uuid.uuid4())
            logger.warning(f"Using fallback session ID: {fallback_id[:8]}...")
            
            # Still try to create ADK session
            try:
                adk_session = await self.adk_session_service.create_session(
                    app_name="TrackSabha",
                    user_id=user_id
                )
                return fallback_id, adk_session.id
            except:
                return fallback_id, fallback_id
    
    async def build_conversation_context(self, session_id: str, max_messages: int = 6) -> str:
        """Build conversation context from MongoDB stored messages."""
        try:
            # Get recent messages from MongoDB
            messages = await self.session_manager.get_session_messages(session_id, limit=max_messages)
            
            if not messages:
                return ""
            
            context = "\n\nRECENT CONVERSATION:\n"
            for message in messages:
                if message.role == "user":
                    context += f"User: {message.content}\n"
                elif message.role == "assistant":
                    # Use truncated version for context
                    assistant_preview = message.content[:200]
                    context += f"Assistant: {assistant_preview}...\n\n"
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to build conversation context: {e}")
            return ""
    
    async def process_query(self, query: str, user_id: str = "user", session_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Process a query through the enhanced parliamentary agent with MongoDB session storage."""
        try:
            logger.info(f"🚀 Processing query: {query[:50]}...")
            
            # Check if this is a graph visualization request
            if "visualize" in query.lower() and "graph" in query.lower():
                logger.info("📊 Detected graph visualization request")
            
            # Get or create session - returns both tracking session ID and ADK session ID
            tracking_session_id, adk_session_id = await self.get_or_create_session(user_id, session_id)
            
            # Set current session context for tools
            self.current_session_id = tracking_session_id
            
            # Get session graph
            session_graph = self.get_or_create_session_graph(tracking_session_id)

            # Proactively seed/update the session graph so visualization works after first Q
            try:
                seeds = self.querier.unified_hybrid_search(query, limit=6)
                seed_uris = {item["uri"] for item in seeds if "uri" in item}
                if seed_uris:
                    connected = self.querier.get_connected_nodes(seed_uris, hops=1)
                    subgraph = self.querier.get_subgraph(connected)
                    turtle_data = self.querier.to_turtle(subgraph)
                    main_data = turtle_data.split("# PROVENANCE DATA:")[0].strip()
                    if main_data and not main_data.startswith("# Error"):
                        session_graph.add_turtle_data(main_data)
                        logger.info("📈 Seeded session graph for visualization")
            except Exception as e:
                logger.warning(f"Graph seeding skipped: {e}")
            
            # Build context from MongoDB conversation history AND session graph
            context = await self.build_conversation_context(tracking_session_id)
            
            # Add session graph context
            if session_graph.edge_count > 0:
                context += "\n\nCURRENT SESSION GRAPH:\n"
                context += session_graph.get_turtle_dump()
                context += "\n\n"
            
            # Store user message in MongoDB
            await self.session_manager.add_message(
                tracking_session_id,
                "user",
                query,
                metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
            )
            
            # Create message with enhanced context
            full_query = f"{context}CURRENT QUESTION: {query}"
            user_message = Content(role="user", parts=[Part.from_text(text=full_query)])
            
            # Run agent and collect ALL events before responding
            all_events = []
            try:
                events = self.runner.run(
                    user_id=user_id,
                    session_id=adk_session_id,  # Use the ADK session_id for the runner
                    new_message=user_message
                )
                
                # Collect all events first (don't stream intermediate responses)
                for event in events:
                    all_events.append(event)
                
            except Exception as runner_error:
                logger.error(f"ADK Runner failed: {runner_error}")
                # Clear session context on error
                self.current_session_id = None
                return "I encountered a technical issue processing your query. Please try again.", {"success": False}
            
            # Now process events and get the final response only
            response_text = ""
            for event in all_events:
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                # Only use the final text response, not intermediate ones
                                response_text = part.text  # This will be the final response
            
            # Parse JSON response
            structured_response = parse_llm_json_response(response_text)
            
            if structured_response:
                # Convert to HTML with unique message ID
                message_id = f"msg-{int(datetime.utcnow().timestamp() * 1000)}"
                html_response = convert_structured_response_to_html(structured_response, message_id)
                
                # Store assistant response in MongoDB
                await self.session_manager.add_message(
                    tracking_session_id,
                    "assistant",
                    response_text,  # Store original response for context
                    metadata={
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "message_id": message_id,
                        "response_type": "structured",
                        "graph_stats": session_graph.get_stats(),
                        "rendered_html": html_response
                    }
                )
                
                # Clear session context
                self.current_session_id = None
                
                # Return structured response
                return html_response, {
                    "success": True, 
                    "session_id": tracking_session_id,
                    "graph_stats": session_graph.get_stats(),
                    "structured_response": structured_response,
                    "response_type": "structured"
                }
            else:
                # Fallback to plain text if JSON parsing fails
                logger.warning("Failed to parse JSON response, falling back to plain text")
                
                # Create a user-friendly fallback message instead of showing raw JSON
                fallback_message = "I encountered an issue formatting my response properly. Here's what I found:\n\n" + response_text[:500]
                if len(response_text) > 500:
                    fallback_message += "..."
                
                # Convert markdown to HTML as fallback
                html_response = markdown.markdown(fallback_message, extensions=['extra', 'codehilite'])
                
                # Filter out non-YouTube links
                soup = BeautifulSoup(html_response, 'html.parser')
                links = soup.find_all('a')
                for link in links:
                    href = link.get('href', '')
                    if 'youtube.com' not in href.lower():
                        link.replace_with(link.get_text())
                html_response = str(soup)
                
                # Store assistant response in MongoDB
                await self.session_manager.add_message(
                    tracking_session_id,
                    "assistant",
                    fallback_message,
                    metadata={
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "response_type": "fallback_error",
                        "graph_stats": session_graph.get_stats(),
                        "rendered_html": html_response
                    }
                )
                
                # Clear session context
                self.current_session_id = None
                
                return html_response, {
                    "success": True, 
                    "session_id": tracking_session_id,
                    "graph_stats": session_graph.get_stats(),
                    "response_type": "fallback_error"
                }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Clear session context on error
            self.current_session_id = None
            
            return f"❌ Error processing query: {str(e)}", {"success": False}
    
    async def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get session history from MongoDB."""
        try:
            messages = await self.session_manager.get_session_messages(session_id, limit)
            return [
                {
                    "message_id": msg.message_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Failed to get session history: {e}")
            return []
    
    async def get_user_sessions(self, user_id: str, limit: int = 10, include_archived: bool = False) -> List[Dict[str, Any]]:
        """Get user's recent sessions from MongoDB."""
        try:
            sessions = await self.session_manager.get_user_sessions(user_id, limit, include_archived)
            return [
                {
                    "session_id": session.session_id,
                    "created_at": session.created_at.isoformat(),
                    "last_updated": session.last_updated.isoformat(),
                    "message_count": len(session.messages),
                    "metadata": session.metadata,
                    "archived": session.metadata.get("archived", False)
                }
                for session in sessions
            ]
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    async def archive_session(self, session_id: str, reason: str = "User requested") -> bool:
        """Archive a session instead of deleting for audit purposes."""
        try:
            # Archive in MongoDB
            success = await self.session_manager.archive_session(session_id, reason)
            
            # Keep graph data in memory but mark it as archived
            if session_id in self.session_graphs:
                logger.info(f"🗃️ Session graph for {session_id[:8]} kept for audit purposes")
            
            return success
        except Exception as e:
            logger.error(f"Failed to archive session: {e}")
            return False
    
    def close(self):
        """Close system resources."""
        if hasattr(self, 'querier'):
            self.querier.close()
        if hasattr(self, 'session_manager'):
            self.session_manager.close()

    # --- API helpers for session graph management ---
    def api_clear_session_graph(self, session_id: str, reason: str = "API request") -> Dict[str, Any]:
        """Clear a session's in-memory knowledge graph and return stats before/after."""
        if session_id not in self.session_graphs:
            return {"cleared": False, "reason": "Session graph not found"}
        sg = self.session_graphs[session_id]
        before = sg.get_stats()
        sg.clear_graph(reason)
        after = sg.get_stats()
        return {"cleared": True, "before": before, "after": after, "reason": reason}

# Global system instance
parliamentary_system = None

def create_system() -> ParliamentarySystem:
    """Create the parliamentary system."""
    google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_GENAI_API_KEY')
    
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY environment variable required")
    
    return ParliamentarySystem(google_api_key)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting Enhanced Parliamentary Chatbot System with MongoDB Sessions and Graph Visualization...")
    
    global parliamentary_system
    try:
        parliamentary_system = create_system()
        logger.info("✅ Enhanced Parliamentary System with MongoDB sessions and graph visualization created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create Parliamentary system: {e}")
        parliamentary_system = None
    
    logger.info("✅ Enhanced Parliamentary Chatbot System ready!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Enhanced Parliamentary Chatbot System...")
    if parliamentary_system:
        parliamentary_system.close()

# Create FastAPI app
app = FastAPI(
    title="Enhanced Parliamentary Research API",
    description="AI-powered Parliamentary research system with MongoDB session persistence and interactive graph visualization",
    version="4.1.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_sse_event(event_type: str, agent: str, message: str, data: Optional[Dict[str, Any]] = None) -> str:
    """Format Server-Sent Event data."""
    event_data = {
        "type": event_type,
        "agent": agent,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if data:
        event_data["data"] = data
    
    return f"data: {json.dumps(event_data)}\n\n"

async def process_query_with_events(query: str, user_id: str, session_id: str):
    """Process query with real-time events."""
    try:
        # Emit an early session_ready so the client can sync sessionId immediately
        yield format_sse_event("session_ready", "System", "Session initialized", {
            "session_id": session_id
        })

        yield format_sse_event("query_start", "System", f"Processing query: {query[:50]}...")
        
        response_text, status = await parliamentary_system.process_query(query, user_id, session_id)
        
        if status.get("success", False):
            # Use actual session ID from the response
            actual_session_id = status.get("session_id", session_id)
            graph_stats = status.get("graph_stats", {})
            structured_response = status.get("structured_response")
            
            yield format_sse_event("response_ready", "Assistant", "Response completed", {
                "response": response_text,
                "message_id": str(uuid.uuid4()),
                "session_id": actual_session_id,
                "type": "parliamentary",
                "graph_stats": graph_stats,
                "structured_response": structured_response.dict() if structured_response else None,
                "response_type": status.get("response_type", "structured")
            })
        else:
            yield format_sse_event("error", "System", f"Error: {response_text}")
        
        yield format_sse_event("stream_complete", "System", "Query processing complete.")
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        yield format_sse_event("error", "System", f"Error processing query: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "TrackSabha - Enhanced Parliamentary Research API with MongoDB Session Storage and Graph Visualization",
        "status": "running",
        "version": "4.1.0",
        "features": ["mongodb_sessions", "audit_compliant_archiving", "persistent_chat_history", "robust_json_repair", "expandable_cards", "session_graph_persistence", "turtle_processing", "cumulative_memory", "interactive_graph_visualization"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not parliamentary_system:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "System not initialized"}
        )
    
    try:
        # Test graph database connection
        parliamentary_system.querier.client.admin.command("ping", maxTimeMS=3000)
        graph_db_connected = True
    except Exception:
        graph_db_connected = False
    
    try:
        # Test session database connection
        parliamentary_system.session_manager.client.admin.command("ping", maxTimeMS=3000)
        session_db_connected = True
        
        # Get session stats
        session_stats = parliamentary_system.session_manager.get_session_stats()
    except Exception as e:
        session_db_connected = False
        session_stats = {"error": str(e)}
    
    total_graph_edges = sum(graph.edge_count for graph in parliamentary_system.session_graphs.values())
    
    return {
        "status": "healthy" if (graph_db_connected and session_db_connected) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "graph_database_connected": graph_db_connected,
        "session_database_connected": session_db_connected,
        "session_graphs_in_memory": len(parliamentary_system.session_graphs),
        "total_graph_edges": total_graph_edges,
        "session_stats": session_stats,
        "version": "4.1.0"
    }

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information and history."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Get session from MongoDB
        session = await parliamentary_system.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get graph stats if available
        graph_stats = None
        if session_id in parliamentary_system.session_graphs:
            graph_stats = parliamentary_system.session_graphs[session_id].get_stats()
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
            "message_count": len(session.messages),
            "metadata": session.metadata,
            "graph_stats": graph_stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 20):
    """Get messages from a session."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        messages = await parliamentary_system.get_session_history(session_id, limit)
        return {
            "session_id": session_id,
            "messages": messages,
            "count": len(messages)
        }
        
    except Exception as e:
        logger.error(f"Failed to get session messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/graph")
async def get_session_graph(session_id: str):
    """Get the current session graph as Turtle format."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if session_id not in parliamentary_system.session_graphs:
        raise HTTPException(status_code=404, detail="Session graph not found")
    
    session_graph = parliamentary_system.session_graphs[session_id]
    turtle_data = session_graph.get_turtle_dump()
    
    return {
        "session_id": session_id,
        "turtle_data": turtle_data,
        "stats": session_graph.get_stats()
    }

@app.get("/session/{session_id}/graph/visualize")
async def visualize_session_graph(session_id: str):
    """Get the session graph as interactive HTML visualization."""
    def render_info_page(title: str, message: str, tips: list[str] = None) -> str:
        tips = tips or []
        tips_html = "".join([f"<li>{t}</li>" for t in tips])
        list_html = f'<ul style="margin:10px 0 0 18px; color:#334155;">{tips_html}</ul>' if tips_html else ''
        # Return a fragment suitable for embedding inside the panel
        return (
            f'<div class="graph-info-message" style="border:1px solid #e2e8f0; border-radius:12px; padding:16px; background:#ffffff;">'
            f'<div style="font-weight:600; font-size:14px; color:#0f172a;">{title}</div>'
            f'<div style="color:#475569; margin-top:6px;">{message}</div>'
            f'{list_html}'
            f'<div style="margin-top:12px; display:flex; gap:8px;">'
            f'<a href="#" onclick="location.reload(); return false;" style="display:inline-block; background:#0ea5e9; color:#fff; padding:6px 10px; border-radius:8px; text-decoration:none; font-size:12px;">Refresh</a>'
            f'<a href="/" style="display:inline-block; background:#64748b; color:#fff; padding:6px 10px; border-radius:8px; text-decoration:none; font-size:12px; margin-left:8px;">Back to Home</a>'
            f'</div>'
            f'</div>'
        )

    if not parliamentary_system:
        # Return a friendly HTML page and a 503 so client can display a clear error
        return HTMLResponse(content=render_info_page(
            title="System not initialized",
            message="The backend system isn't ready yet. Please wait a moment and try again.",
            tips=[
                "Ensure the server process has started successfully.",
                "Check environment variables for database/API keys.",
            ],
        ), status_code=503)
    
    try:
        # Helpful guards for missing/empty graphs
        sg = parliamentary_system.session_graphs.get(session_id)
        if sg is None:
            return HTMLResponse(content=render_info_page(
                title="No graph found for this session",
                message=f"We couldn't find an in-memory knowledge graph for session <code>{session_id}</code>.",
                tips=[
                    "Start by asking a question to seed the session graph.",
                    "Make sure you passed the correct session_id from the chat response.",
                    "The graph is built incrementally as you search or ask questions.",
                ],
            ), status_code=404)

        if getattr(sg, 'edge_count', 0) == 0:
            return HTMLResponse(content=render_info_page(
                title="Session graph is empty",
                message="This session exists but doesn't have any relationships yet to visualize.",
                tips=[
                    "Ask a question about entities, people, or bills to populate the graph.",
                    "The system attempts to seed graph context on your first query.",
                    "After a query completes, click Refresh to update the view.",
                ],
            ), status_code=404)

        html_content = parliamentary_system.visualize_session_graph(session_id)
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Failed to visualize session graph: {e}")
        return HTMLResponse(content=render_info_page(
            title="Graph visualization error",
            message="We hit an unexpected error while creating the visualization.",
            tips=[
                "Try refreshing this page.",
                "Run one more query, then refresh again.",
                "Check server logs for details if the issue persists.",
            ],
        ), status_code=500)

@app.get("/session/{session_id}/graph/refresh")
async def refresh_session_graph(session_id: str):
    """Alias to visualize the session graph (convenience for clients)."""
    return await visualize_session_graph(session_id)

@app.get("/user/{user_id}/sessions")
async def get_user_sessions(user_id: str, limit: int = 10, include_archived: bool = False):
    """Get user's recent sessions."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        sessions = await parliamentary_system.get_user_sessions(user_id, limit, include_archived)
        return {
            "user_id": user_id,
            "sessions": sessions,
            "count": len(sessions),
            "include_archived": include_archived
        }
        
    except Exception as e:
        logger.error(f"Failed to get user sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/{session_id}/archive")
async def archive_session(session_id: str, reason: str = "User requested"):
    """Archive a session for audit purposes instead of deleting."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        success = await parliamentary_system.archive_session(session_id, reason)
        if success:
            return {
                "message": f"Session {session_id} archived successfully for audit purposes",
                "reason": reason,
                "note": "Data preserved for compliance and audit requirements"
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to archive session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/{session_id}/graph/clear")
async def clear_session_graph_api(session_id: str, reason: str = "User requested"):
    """Clear a session's in-memory knowledge graph and return before/after stats."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        result = parliamentary_system.api_clear_session_graph(session_id, reason)
        if not result.get("cleared"):
            raise HTTPException(status_code=404, detail=result.get("reason", "Session graph not found"))
        return {
            "message": f"Session {session_id} graph cleared",
            "reason": reason,
            **result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear session graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a query."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        response_text, status = await parliamentary_system.process_query(
            request.query, 
            request.user_id, 
            request.session_id
        )
        
        if status.get("success", False):
            actual_session_id = status.get("session_id", "parliamentary_session")
            structured_response = status.get("structured_response")
            
            return QueryResponse(
                session_id=actual_session_id,
                user_id=request.user_id,
                message_id="msg_" + str(datetime.utcnow().timestamp()),
                status="success",
                message=response_text,
                structured_response=structured_response
            )
        else:
            raise HTTPException(status_code=500, detail=response_text)
            
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream query processing."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    async def event_generator():
        async for event_data in process_query_with_events(request.query, request.user_id, session_id):
            yield event_data
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/stats")
async def get_system_stats():
    """Get overall system statistics."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        session_stats = parliamentary_system.session_manager.get_session_stats()
        graph_stats = {
            "active_session_graphs": len(parliamentary_system.session_graphs),
            "total_graph_edges": sum(graph.edge_count for graph in parliamentary_system.session_graphs.values())
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "4.1.0",
            "session_stats": session_stats,
            "graph_stats": graph_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/archive-old-sessions")
async def archive_old_sessions(days_old: int = 365):
    """Admin endpoint to archive old sessions for audit compliance."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        archived_count = await parliamentary_system.session_manager.cleanup_old_sessions(days_old)
        return {
            "message": f"Archived {archived_count} sessions older than {days_old} days",
            "archived_count": archived_count,
            "note": "Sessions archived for audit compliance, not deleted",
            "retention_policy": f"Sessions auto-archived after {days_old} days of inactivity"
        }
        
    except Exception as e:
        logger.error(f"Failed to archive old sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info("🚀 Starting Enhanced Parliamentary Chatbot with MongoDB Session Storage and Graph Visualization")
    logger.info(f"📡 Server will run on 0.0.0.0:{port}")
    logger.info("📋 Required: GOOGLE_API_KEY, MONGODB_CONNECTION_STRING")
    logger.info("🔧 New: Interactive knowledge graph visualization with D3.js")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        #reload=True
    )