#!/usr/bin/env python3
"""
TrackSabha - Indian Parliament Chatbot with MongoDB Session Storage and Graph Visualization
=========================================================================================

Updated to use MongoDB for persistent chat history and session management;
this service (TrackSabha) focuses on Indian Parliament transcripts, quotes, and provenance.
Includes interactive knowledge graph visualization.
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
import re
import time
import random
import numpy as np
import math
from bs4 import BeautifulSoup

# Load environment variables from .env (if present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; if not present the environment may be set externally
    pass

# JSON repair
from json_repair import repair_json

# Mount static files and templates
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pytz

# Import our new session manager
from session_manager import MongoSessionManager, ChatMessage, ChatSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RDF Namespaces (aligned with scripts/06_demo prefixes)
LOK = Namespace("http://example.com/Indian-parliament-ontology#")
SESS = Namespace("http://example.com/Indian-parliament-session/")
SCHEMA = Namespace("http://schema.org/")
ORG = Namespace("http://www.w3.org/ns/org#")
PROV = Namespace("http://www.w3.org/ns/prov#")
BBP = Namespace("http://example.com/JanSetu.in/")

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
    """Manages cumulative graph state for a session using JSON format."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.clear_graph("New session started")
        
    def add_json_data(self, json_data: Dict[str, Any]) -> bool:
        """Add JSON data to the cumulative graph."""
        try:
            # Merge entities
            new_entities = json_data.get('entities', [])
            new_statements = json_data.get('statements', [])
            
            # Track existing entity IDs to avoid duplicates
            existing_entity_ids = {entity['entity_id'] for entity in self.entities}
            
            # Add new entities
            for entity in new_entities:
                if entity.get('entity_id') not in existing_entity_ids:
                    self.entities.append(entity)
                    existing_entity_ids.add(entity['entity_id'])
            
            # Add new statements (allow duplicates as they may have different provenance)
            self.statements.extend(new_statements)
            
            # Update counts
            self.node_count = len(self.entities)
            self.edge_count = len(self.statements)
            
            logger.info(f"📈 Session {self.session_id[:8]}: Added {len(new_entities)} entities, {len(new_statements)} statements. Total: {self.node_count} nodes, {self.edge_count} edges")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process JSON data: {e}")
            return False
    
    def add_turtle_data(self, turtle_str: str) -> bool:
        """Legacy method for backward compatibility - converts Turtle to basic JSON."""
        try:
            # For backward compatibility, convert basic Turtle to JSON structure
            temp_graph = Graph()
            temp_graph.parse(data=turtle_str, format='turtle')
            
            # Convert to basic JSON structure
            entities = []
            statements = []
            
            for subject, predicate, obj in temp_graph:
                # Create basic entities and statements from triples
                subject_id = str(subject).split('#')[-1] if '#' in str(subject) else str(subject)
                object_id = str(obj).split('#')[-1] if '#' in str(obj) else str(obj)
                
                # Add entities if they don't exist
                for entity_id in [subject_id, object_id]:
                    if not any(e['entity_id'] == entity_id for e in entities):
                        entities.append({
                            'entity_id': entity_id,
                            'entity_name': entity_id,
                            'entity_type': 'Unknown',
                            'entity_description': f'Entity from RDF: {entity_id}'
                        })
                
                # Add statement
                statements.append({
                    'source_entity_id': subject_id,
                    'target_entity_id': object_id,
                    'relationship_description': str(predicate).split('#')[-1] if '#' in str(predicate) else str(predicate),
                    'relationship_strength': 5,
                    'provenance_segment_id': 'legacy_rdf'
                })
            
            return self.add_json_data({'entities': entities, 'statements': statements})
            
        except Exception as e:
            logger.error(f"Failed to parse Turtle: {e}")
            return False
    
    def get_json_dump(self) -> Dict[str, Any]:
        """Get current graph as JSON format."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'node_count': self.node_count,
            'edge_count': self.edge_count,
            'entities': self.entities,
            'statements': self.statements
        }
    
    def get_turtle_dump(self) -> str:
        """Legacy method for backward compatibility - returns JSON as string."""
        try:
            header = f"""# Session Graph Dump (JSON Format)
# Session: {self.session_id}
# Created: {self.created_at.isoformat()}
# Nodes: {self.node_count}, Edges: {self.edge_count}
"""
            return header + json.dumps(self.get_json_dump(), indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to serialize graph: {e}")
            return f"# Error serializing graph: {e}\n"
    
    def clear_graph(self, reason: str = "Topic change"):
        """Clear the cumulative graph."""
        self.entities = []
        self.statements = []
        self.created_at = datetime.now(timezone.utc)
        
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
            "size_mb": len(json.dumps(self.get_json_dump(), default=str)) / (1024 * 1024)
        }

class ParliamentaryGraphQuerier:
    """Database querier with full search functionality."""

    def __init__(self):
        self.client = None
        self.db = None
        self.nodes = None
        self.edges = None
        self.statements = None
        self.embedding_model = None
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
            
            # Initialize database references - use project-level `youtube_data` database
            # Use the canonical graph collections (`nodes`, `edges`, `statements`, `videos`) used by the scripts
            self.db = self.client["youtube_data"]
            # Keep `entities` as an alias for the canonical `nodes` collection so the webapp mapping is compatible
            self.entities = self.db.nodes
            self.nodes = self.db.nodes
            self.edges = self.db.edges
            self.statements = self.db.statements
            self.provenance_segments = self.db.get_collection("provenance_segments")  # For video details
            self.videos = self.db.videos  # For video metadata
            
            # Legacy collections for fallback (if needed)
            self.nodes = self.db.nodes if "nodes" in self.db.list_collection_names() else None
            self.edges = self.db.edges if "edges" in self.db.list_collection_names() else None
            
            # Create indexes if they don't exist
            try:
                # Indexes for nodes collection (canonical schema)
                try:
                    self.entities.create_index([("uri", ASCENDING)], unique=True)
                except Exception:
                    pass
                try:
                    self.entities.create_index([("label", ASCENDING)])
                except Exception:
                    pass
                try:
                    self.entities.create_index([("searchable_text", ASCENDING)])
                except Exception:
                    pass
            except:
                pass
            
            logger.info("✅ Connected to MongoDB (new knowledge graph)")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            if self.client:
                self.client.close()
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    def _initialize_embeddings(self):
        """Initialize embedding model."""
        # Prefer Google GenAI embeddings. Initialize client if credentials are present.
        self.google_client = None
        self.use_google_embeddings = False
        try:
            logger.info("🔄 Initializing Google GenAI embeddings client (if available)...")
            # Try multiple GenAI SDK variants. Some environments have `google.genai` with Client,
            # others use the module-level `google.genai.embeddings`, and older code may use
            # `google.generativeai` (module-level). We'll store whichever module we can import
            # and keep the API key for fallback configuration.
            try:
                import google.genai as genai
            except Exception:
                genai = None

            try:
                import google.generativeai as generativeai
            except Exception:
                generativeai = None

            self.genai_module = genai or generativeai
            self.google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

            # Prefer to create a client if the genai package exposes Client
            if genai and self.google_api_key:
                try:
                    self.google_client = genai.Client()
                    self.use_google_embeddings = True
                    logger.info("✅ google.genai.Client() created for embeddings")
                except Exception as e:
                    logger.warning(f"google.genai.Client() not usable: {e}")
                    self.google_client = None
                    self.use_google_embeddings = False

            # If no client, but module-level embeddings API exists, mark embeddings usable
            if not self.google_client and self.genai_module is not None:
                # For google.genai the module may expose `embeddings.create`; for google.generativeai
                # the module exposes embeddings.create after configure()
                self.use_google_embeddings = True
                logger.info("✅ Found GenAI module for embeddings (will attempt module-level calls)")

            if not self.use_google_embeddings:
                logger.warning("Google GenAI client/module not available or credentials missing; vector search disabled")

        except Exception as e:
            logger.warning(f"Embedding initialization error: {e}")

    def _generate_query_embedding(self, text: str) -> Optional[List[float]]:
        """Generate an embedding for the given text using Google GenAI and normalize the response.

        Returns a list of floats or None on failure.
        """
        # We'll attempt multiple GenAI call patterns and normalize responses similarly to the backfill script.
        if not getattr(self, 'genai_client', None) and not getattr(self, 'genai_module', None):
            logger.debug("No GenAI client or module available for embeddings")
            return None

        # Clean text
        clean_text = re.sub(r"\s+", " ", text).strip()
        if len(clean_text) < 1:
            return None

        max_retries = 5
        base_delay = 1.0
        attempt = 0
        resp = None

        while attempt <= max_retries:
            try:
                # 1) Prefer client.models.embed_content() as used in scripts/05_backfill_embeddings.py
                if getattr(self, 'genai_client', None) and getattr(self.genai_client, 'models', None) and getattr(self.genai_client.models, 'embed_content', None):
                    resp = self.genai_client.models.embed_content(model="gemini-embedding-001", contents=[clean_text])
                # 2) Older or module-level APIs: genai_module.embeddings.create(...)
                elif getattr(self, 'genai_module', None) and getattr(self.genai_module, 'embeddings', None):
                    try:
                        # If module has configure and we have an API key, configure it
                        if getattr(self.genai_module, 'configure', None) and self.google_api_key:
                            try:
                                self.genai_module.configure(api_key=self.google_api_key)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    resp = self.genai_module.embeddings.create(model="gemini-embedding-001", input=[clean_text])
                # 3) As a last resort, some SDKs expose client.embeddings.create
                elif getattr(self, 'google_client', None) and hasattr(self.google_client, 'embeddings'):
                    resp = self.google_client.embeddings.create(model="gemini-embedding-001", input=[clean_text])
                else:
                    logger.warning("No supported GenAI embedding method found in environment")
                    return None

                # If we got a response, break the retry loop
                break
            except Exception as e:
                msg = str(e)
                if '429' in msg or 'RESOURCE_EXHAUSTED' in msg or 'quota' in msg.lower():
                    sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(f"⚠️ Rate limit encountered (attempt {attempt+1}/{max_retries}). Sleeping {sleep_time:.1f}s before retry...")
                    time.sleep(sleep_time)
                    attempt += 1
                    continue
                else:
                    logger.error(f"Google embedding generation failed: {e}")
                    return None

        if resp is None:
            logger.error("Embedding generation failed after retries")
            return None

        # Normalize different possible return shapes
        emb_obj = getattr(resp, 'embeddings', None)
        if emb_obj is None and isinstance(resp, dict) and 'embeddings' in resp:
            emb_obj = resp['embeddings']
        if emb_obj is None and isinstance(resp, dict) and 'data' in resp:
            emb_obj = resp['data']

        if emb_obj is None:
            # Some clients return the embedding directly as a list or nested structure
            emb_entry = resp
        else:
            if isinstance(emb_obj, (list, tuple)) and len(emb_obj) > 0:
                emb_entry = emb_obj[0]
            else:
                emb_entry = emb_obj

        emb_list = None

        # Try attribute 'values'
        try:
            values_attr = getattr(emb_entry, 'values', None)
            if values_attr is not None and isinstance(values_attr, (list, tuple)):
                emb_list = list(values_attr)
        except Exception:
            pass

        # If it's a plain list/tuple of numbers
        if emb_list is None and isinstance(emb_entry, (list, tuple)) and all(not isinstance(x, (list, tuple, dict)) for x in emb_entry):
            emb_list = list(emb_entry)
        elif emb_list is None and isinstance(emb_entry, dict):
            for key in ("embedding", "value", "vector", "data"):
                if key in emb_entry and isinstance(emb_entry[key], (list, tuple)):
                    emb_list = list(emb_entry[key])
                    break
            if emb_list is None and 'data' in emb_entry and isinstance(emb_entry['data'], (list, tuple)) and len(emb_entry['data']) > 0:
                inner = emb_entry['data'][0]
                if isinstance(inner, dict):
                    for key in ("embedding", "value", "vector"):
                        if key in inner and isinstance(inner[key], (list, tuple)):
                            emb_list = list(inner[key])
                            break
        else:
            try:
                cand = getattr(emb_entry, 'embedding', None)
                if isinstance(cand, (list, tuple)):
                    emb_list = list(cand)
            except Exception:
                emb_list = None

        if emb_list is None:
            try:
                emb_list = list(emb_entry)
            except Exception:
                logger.warning(f"Could not parse embedding entry type={type(emb_entry)} repr={str(emb_entry)[:200]}")
                return None

        # Normalize numeric values (handle tuples like (index, value))
        normalized = []
        for item in emb_list:
            if isinstance(item, (int, float)):
                normalized.append(float(item))
            elif isinstance(item, (list, tuple)):
                num = None
                for part in item:
                    if isinstance(part, (int, float)):
                        num = float(part)
                        if len(item) == 2:
                            num = float(item[1])
                            break
                if num is not None:
                    normalized.append(num)
            else:
                try:
                    normalized.append(float(item))
                except Exception:
                    continue

        if not normalized:
            logger.warning("Embedding normalization produced empty vector")
            return None

        # Optionally warn if length differs from expectation (embedding_dim may not be set)
        try:
            expected = getattr(self, 'embedding_dim', None)
            if expected and len(normalized) != int(expected):
                logger.warning(f"Warning: received embedding length {len(normalized)} != expected {expected}")
        except Exception:
            pass

        return normalized

    def _search_nodes_vector(self, query: str, limit: int) -> List[Dict]:
        """Vector search on entities using new schema"""
        # Use Google GenAI to generate the query embedding
        query_embedding = self._generate_query_embedding(query)
        if not query_embedding:
            logger.error("No embedding available for query; skipping vector search")
            return []

        pipeline = [
            {"$vectorSearch": {
                "index": "vector_index_1",  # Project standard vector index name
                "path": "embedding",  # Standard embedding field path
                "queryVector": query_embedding,
                "numCandidates": limit * 3,
                "limit": limit
            }},
            {"$addFields": {
                "similarity_score": {"$meta": "vectorSearchScore"}
            }}
        ]
        
        return list(self.entities.aggregate(pipeline))

    def _search_statements_atlas(self, query: str, limit: int) -> List[Dict]:
        """Atlas Search on statements - adapted for new schema"""
        # New statements don't have transcript_text, so we search on relationship_description
        pipeline = [
            {
                "$search": {
                    "index": "default",
                    "compound": {
                        "should": [
                            {
                                "phrase": {
                                    "query": query,
                                    "path": "relationship_description",
                                    "score": {"boost": {"value": 3}}
                                }
                            },
                            {
                                "text": {
                                    "query": query,
                                    "path": "relationship_description",
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
                    "source_entity_id": 1, "target_entity_id": 1, 
                    "relationship_description": 1, "relationship_strength": 1,
                    "provenance_segment_id": 1, "_id": 1,
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
        vector_weight /= total
        text_weight /= total
        
        return {
            'vector_weight': vector_weight,
            'text_weight': text_weight
        }

    def unified_hybrid_search(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Performs both node vector search and statement text search,
        then intelligently combines and weights the results.
        """
        try:
            # Get results from both search methods
            node_results = self._search_nodes_vector(query, limit)
            statement_results = self._search_statements_atlas(query, limit * 2)

            # Convert both result types to unified format
            unified_results = []
            
            # Process entity results (from vector search)
            for entity in node_results:
                uri = entity.get('uri') or entity.get('entity_id') or entity.get('_id')
                label = entity.get('label') or entity.get('entity_name') or entity.get('local_name')
                content = entity.get('searchable_text') or entity.get('entity_description') or entity.get('description') or ''

                unified_results.append({
                    'uri': uri,
                    'source_type': 'entity',
                    'content': content,
                    'label': label,
                    'node_data': entity,
                    'vector_score': entity.get('similarity_score', 0),
                    'text_score': 0,
                    'provenance': None
                })

            # Process statement results (from text search) and find their related entities
            if statement_results:
                all_related_entity_ids = set()
                stmt_to_entity_ids = {}
                
                for i, stmt in enumerate(statement_results):
                    related_ids = []
                    if stmt.get('source_entity_id'):
                        related_ids.append(stmt['source_entity_id'])
                    if stmt.get('target_entity_id'):
                        related_ids.append(stmt['target_entity_id'])
                    
                    stmt_to_entity_ids[i] = related_ids
                    all_related_entity_ids.update(related_ids)
                
                if all_related_entity_ids:
                    entities_cursor = self.entities.find(
                        {'$or': [
                            {'uri': {'$in': list(all_related_entity_ids)}},
                            {'entity_id': {'$in': list(all_related_entity_ids)}}
                        ]},
                        {'uri': 1, 'label': 1, 'entity_id': 1, 'entity_name': 1, 'searchable_text': 1}
                    )

                    entity_id_to_entity = {
                        (entity.get('uri') or entity.get('entity_id')): entity
                        for entity in entities_cursor
                    }

                    for i, stmt in enumerate(statement_results):
                        for entity_id in stmt_to_entity_ids.get(i, []):
                            entity = entity_id_to_entity.get(entity_id)
                            if entity:
                                unified_results.append({
                                    'uri': entity_id,
                                    'source_type': 'statement',
                                    'content': stmt.get('relationship_description', ''),
                                    'label': entity.get('label') or entity.get('entity_name', ''),
                                    'node_data': entity,
                                    'vector_score': 0,
                                    'text_score': stmt.get('search_score', 0),
                                    'provenance': {
                                        'statement_id': stmt.get('_id'),
                                        'relationship': stmt.get('relationship_description'),
                                        'strength': stmt.get('relationship_strength'),
                                        'provenance_segment_id': stmt.get('provenance_segment_id')
                                    }
                                })
            
            # Deduplicate by URI
            uri_to_result = {}
            for result in unified_results:
                uri = result['uri']
                if uri not in uri_to_result:
                    uri_to_result[uri] = result
                else:
                    existing = uri_to_result[uri]
                    existing['vector_score'] = max(existing['vector_score'], result['vector_score'])
                    existing['text_score'] = max(existing['text_score'], result['text_score'])
                    if result['provenance'] and not existing['provenance']:
                        existing['provenance'] = result['provenance']
                        existing['content'] = result['content']
            
            final_results = list(uri_to_result.values())
            final_results = self._calculate_unified_scores(final_results, query)
            
            final_results.sort(key=lambda x: x.get('unified_score', 0), reverse=True)
            
            logger.info(f"🎯 Unified search: {len(final_results)} unique results")
            return final_results[:limit]
            
        except Exception as e:
            logger.error(f"❌ Unified search failed: {e}")
            return []

    def _calculate_pagerank_boost(self, node_data: Dict) -> float:
        """Calculate PageRank boost factor for a node."""
        try:
            if not node_data.get("pagerank"):
                return 0
            
            # Simple exponential decay: higher PageRank gets a higher boost
            return min(1.5, 1.0 + node_data["pagerank"] / 10.0)
        
        except Exception as e:
            logger.warning(f"Error calculating PageRank boost: {e}")
            return 0

    def _calculate_provenance_boost(self, provenance: Any) -> float:
        """Calculate boost factor based on provenance quality."""
        if not provenance:
            return 0
        
        try:
            boost = 0
            if isinstance(provenance, dict):
                if provenance.get('video_id'): boost += 0.1
                if provenance.get('start_time'): boost += 0.1
                if provenance.get('transcript_excerpt'): boost += 0.1
            
            return min(0.3, boost)
        except Exception as e:
            logger.warning(f"Error calculating provenance boost: {e}")
            return 0

    def _get_content_quality_boost(self, content: str) -> float:
        """Boost based on content richness"""
        if not content:
            return 0
        
        # Simple content quality indicators
        word_count = len(content.split())
        boost = min(0.1, word_count / 1000)  # Up to 0.1 boost for rich content
        
        return boost

    def get_connected_nodes(self, entity_ids: Set[str], hops: int = 1) -> Set[str]:
        """Get entities connected to the given entity IDs via statements."""
        try:
            current, seen = set(entity_ids), set(entity_ids)
            for hop in range(max(0, hops)):
                if not current or len(seen) > 500:
                    break
                    
                statements = self.statements.find({
                    "$or": [
                        {"source_entity_id": {"$in": list(current)}},
                        {"target_entity_id": {"$in": list(current)}},
                    ]
                })
                
                nxt = set()
                for stmt in statements:
                    nxt.add(stmt["source_entity_id"])
                    nxt.add(stmt["target_entity_id"])
                
                current = nxt - seen
                seen.update(nxt)
                    
            return seen
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return entity_ids

    def get_subgraph(self, entity_ids: Set[str]) -> Dict[str, Any]:
        """Get subgraph for the given entity IDs."""
        try:
            if len(entity_ids) > 500:
                entity_ids = set(list(entity_ids)[:500])
            
            # Get entities
            raw_entities = list(self.entities.find(
                {"entity_id": {"$in": list(entity_ids)}}, 
                {
                    "entity_id": 1,
                    "entity_name": 1,
                    "entity_type": 1,
                    "entity_description": 1
                }
            ))
            
            # Clean entities (convert to node-like format for compatibility)
            cleaned_nodes = []
            for entity in raw_entities:
                cleaned = {
                    "uri": entity.get("entity_id"),
                    "type": [entity.get("entity_type", "")]
                }
                
                # Handle labels - use entity_name as label
                label = entity.get("entity_name")
                if label:
                    cleaned["label"] = label
                
                if "entity_description" in entity:
                    cleaned["searchable_text"] = entity["entity_description"]
                
                cleaned_nodes.append(cleaned)
            
            # Get statements (convert to edge-like format for compatibility)
            statements = list(self.statements.find({
                "source_entity_id": {"$in": list(entity_ids)}, 
                "target_entity_id": {"$in": list(entity_ids)}
            }))
            
            # Convert statements to edge format
            edges = []
            for stmt in statements:
                edge = {
                    "_id": str(stmt.get("_id", "")),
                    "subject": stmt.get("source_entity_id"),
                    "predicate": "relationship",  # Generic predicate
                    "object": stmt.get("target_entity_id"),
                    "relationship_description": stmt.get("relationship_description"),
                    "relationship_strength": stmt.get("relationship_strength")
                }
                edges.append(edge)
            
            return {"nodes": cleaned_nodes, "edges": edges}
            
        except Exception as e:
            logger.error(f"Subgraph retrieval failed: {e}")
            return {"nodes": [], "edges": []}

    def to_turtle(self, subgraph: Dict[str, Any]) -> str:
        """Convert subgraph to Turtle format."""
        try:
            g = Graph()
            
            # Add prefixes
            # Bind prefixes consistent with scripts/06_demo
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

    def get_provenance_turtle(self, entity_ids: List[str], include_transcript: bool = True) -> str:
        """Get provenance information as Turtle format."""
        try:
            logger.info(f"📚 Getting provenance for {len(entity_ids)} entities")
            
            g = Graph()
            # Bind prefixes consistent with scripts/06_demo
            g.bind("lok", LOK)
            g.bind("bbp", BBP)
            g.bind("sess", SESS)
            g.bind("schema", SCHEMA)
            g.bind("org", ORG)
            g.bind("prov", PROV)
            g.bind("rdfs", RDFS)
            
            for entity_id in entity_ids[:10]:  # Limit to prevent explosion
                try:
                    entity_uri = URIRef(entity_id)
                    
                    # Get related statements using new schema
                    projection = {
                        "source_entity_id": 1,
                        "target_entity_id": 1,
                        "relationship_description": 1,
                        "relationship_strength": 1,
                        "provenance_segment_id": 1,
                        "_id": 1
                    }
                    
                    statements = list(self.statements.find({
                        "$or": [
                            {"source_entity_id": entity_id},
                            {"target_entity_id": entity_id}
                        ]
                    }, projection))
                    
                    # Process statements
                    for i, stmt in enumerate(statements[:5]):
                        stmt_uri = URIRef(f"{entity_id}/statement/{i}")
                        
                        # Basic provenance
                        g.add((stmt_uri, RDF.type, PROV.Entity))
                        g.add((stmt_uri, PROV.wasDerivedFrom, entity_uri))
                        g.add((stmt_uri, SCHEMA.about, entity_uri))
                        
                        # Relationship information (new schema)
                        relationship_desc = stmt.get("relationship_description")
                        if relationship_desc:
                            g.add((stmt_uri, SCHEMA.description, Literal(relationship_desc)))
                        
                        strength = stmt.get("relationship_strength")
                        if strength:
                            g.add((stmt_uri, SCHEMA.ratingValue, Literal(strength)))
                        
                        # Provenance segment information
                        segment_id = stmt.get("provenance_segment_id")
                        if segment_id:
                            g.add((stmt_uri, PROV.hadPrimarySource, Literal(segment_id)))
                            
                            # Try to get video information from segment ID
                            # Segment IDs typically contain video ID
                            if "_" in segment_id:
                                video_id = segment_id.split("_")[0]
                                video_url = f"https://www.youtube.com/watch?v={video_id}"
                                g.add((stmt_uri, SCHEMA.url, Literal(video_url)))
                        
                except Exception as e:
                    logger.warning(f"Skipping provenance for {entity_id}: {e}")
            
            header = f"# Provenance information generated {datetime.now(timezone.utc).isoformat()}Z\n\n"
            return header + g.serialize(format="turtle")
            
        except Exception as e:
            logger.error(f"❌ Provenance turtle generation failed: {e}")
            return f"# Error: {str(e)}\n"

    def get_provenance_details(self, segment_ids: List[str]) -> Dict[str, Dict]:
        """Look up full provenance details for segment IDs."""
        try:
            if not segment_ids:
                return {}
            
            # Get provenance segments
            segments = list(self.provenance_segments.find(
                {"_id": {"$in": segment_ids}},
                {
                    "_id": 1,
                    "video_id": 1, 
                    "time_seconds": 1,
                    "end_time_seconds": 1,
                    "transcript_segment": 1
                }
            ))
            
            # Get unique video IDs to fetch video metadata
            video_ids = list(set(seg.get("video_id") for seg in segments if seg.get("video_id")))
            
            # Get video metadata
            videos = {}
            if video_ids:
                video_docs = list(self.videos.find(
                    {"video_id": {"$in": video_ids}},
                    {
                        "video_id": 1,
                        "title": 1,
                        "video_url": 1,
                        "upload_date": 1
                    }
                ))
                videos = {v["video_id"]: v for v in video_docs}
            
            # Build detailed provenance info
            provenance_details = {}
            for segment in segments:
                segment_id = segment["_id"]
                video_id = segment.get("video_id")
                video_info = videos.get(video_id, {})
                
                start_time = segment.get("time_seconds", 0)
                end_time = segment.get("end_time_seconds", start_time + 30)  # Default 30 sec if no end time
                
                # Construct YouTube URL with timestamp
                base_url = video_info.get("video_url", f"https://www.youtube.com/watch?v={video_id}")
                if not base_url.startswith("http"):
                    base_url = f"https://www.youtube.com/watch?v={video_id}"
                
                # Add timestamp parameter
                timestamped_url = f"{base_url}&t={int(start_time)}s" if start_time else base_url
                logger.debug(f"Constructed URL for segment {segment_id}: {timestamped_url} (start_time: {start_time})")
                
                provenance_details[segment_id] = {
                    "segment_id": segment_id,
                    "video_id": video_id,
                    "video_title": video_info.get("title", "Parliamentary Session"),
                    "video_url": base_url,
                    "timestamped_url": timestamped_url,
                    "start_time": int(start_time) if start_time else None,
                    "end_time": int(end_time) if end_time else None,
                    "duration": int(end_time - start_time) if (start_time and end_time) else None,
                    "transcript_text": segment.get("transcript_segment", ""),
                    "upload_date": video_info.get("upload_date"),
                    "formatted_timestamp": self._format_timestamp(start_time) if start_time else None
                }
            
            return provenance_details
            
        except Exception as e:
            logger.error(f"Error getting provenance details: {e}")
            return {}
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS or MM:SS format."""
        try:
            total_seconds = int(seconds)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            
            if hours > 0:
                return f"{hours}:{minutes:02d}:{secs:02d}"
            else:
                return f"{minutes}:{secs:02d}"
        except:
            return "0:00"
    
    def _find_bridge_connections(self, nodes_data: Dict, existing_edges: List[Dict]) -> List[Dict]:
        """Find bridge connections to link disconnected clusters."""
        try:
            # Build connectivity graph
            connected_components = self._find_connected_components(nodes_data, existing_edges)
            
            if len(connected_components) <= 1:
                return []  # Already connected
            
            # Find potential bridge entities by searching for shared contexts
            bridge_edges = []
            entity_ids = list(nodes_data.keys())
            
            # Look for entities that could bridge clusters based on name similarity
            bridge_candidates = []
            for comp1_idx, comp1 in enumerate(connected_components):
                for comp2_idx, comp2 in enumerate(connected_components):
                    if comp1_idx >= comp2_idx:
                        continue
                    
                    # Look for semantic bridges between clusters
                    for entity1 in comp1:
                        for entity2 in comp2:
                            # Check if they're related by name/type
                            node1 = nodes_data[entity1]
                            node2 = nodes_data[entity2]
                            
                            # Government/Ministry connections
                            if self._are_government_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "government connection",
                                    "predicate": "government_related",
                                    "strength": 7,
                                    "bridge": True
                                })
                            
                            # Policy domain connections
                            elif self._are_policy_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "policy connection",
                                    "predicate": "policy_related",
                                    "strength": 6,
                                    "bridge": True
                                })
                            
                            # Speaker/Person bridges
                            elif self._are_speaker_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "speaker connection",
                                    "predicate": "speaker_related",
                                    "strength": 7,
                                    "bridge": True
                                })
                            
                            # Institutional bridges
                            elif self._are_institutional_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "institutional connection",
                                    "predicate": "institutional_related",
                                    "strength": 6,
                                    "bridge": True
                                })
                            
                            # Geographic/Constituency bridges
                            elif self._are_geographic_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "geographic connection",
                                    "predicate": "geographic_related",
                                    "strength": 5,
                                    "bridge": True
                                })
                            
                            # Broader topic domain bridges
                            elif self._are_topic_domain_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "topic domain connection",
                                    "predicate": "topic_domain_related",
                                    "strength": 5,
                                    "bridge": True
                                })
                            
                            # Parliamentary procedure bridges
                            elif self._are_procedure_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "procedural connection",
                                    "predicate": "procedure_related",
                                    "strength": 5,
                                    "bridge": True
                                })
            
            return bridge_edges[:50]  # Allow many more bridge connections
            
        except Exception as e:
            logger.warning(f"Error finding bridge connections: {e}")
            return []
    
    def _find_connected_components(self, nodes_data: Dict, edges: List[Dict]) -> List[List[str]]:
        """Find connected components in the graph."""
        visited = set()
        components = []
        
        # Build adjacency list
        adj_list = {node_id: [] for node_id in nodes_data.keys()}
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source in adj_list and target in adj_list:
                adj_list[source].append(target)
                adj_list[target].append(source)
        
        # DFS to find components
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in adj_list.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node_id in nodes_data.keys():
            if node_id not in visited:
                component = []
                dfs(node_id, component)
                if component:
                    components.append(component)
        
        return components
    
    def _are_government_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are government-related."""
        gov_terms = ["ministry", "minister", "government", "parliament", "mp", "senator"]
        
        name1 = (node1.get("name", "") + " " + node1.get("type", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("type", "")).lower()
        
        gov1 = any(term in name1 for term in gov_terms)
        gov2 = any(term in name2 for term in gov_terms)
        
        return gov1 and gov2
    
    def _are_policy_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are policy-related."""
        policy_terms = ["policy", "health", "education", "housing", "welfare", "economic"]
        
        name1 = (node1.get("name", "") + " " + node1.get("description", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("description", "")).lower()
        
        for term in policy_terms:
            if term in name1 and term in name2:
                return True
        
        return False
    
    def _are_speaker_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are speaker/person related."""
        speaker_terms = ["honourable", "minister", "mp", "senator", "prime", "leader", "chairman", "speaker"]
        person_indicators = ["mr", "mrs", "ms", "dr", "hon", "rt", "right"]
        
        name1 = (node1.get("name", "") + " " + node1.get("type", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("type", "")).lower()
        
        # Check for person-type entities
        person1 = any(term in name1 for term in speaker_terms + person_indicators)
        person2 = any(term in name2 for term in speaker_terms + person_indicators)
        
        # Also check for shared name words (same person in different contexts)
        words1 = set(name1.split())
        words2 = set(name2.split())
        shared_words = words1.intersection(words2)
        
        return (person1 and person2) or (len(shared_words) >= 2 and any(word in speaker_terms for word in shared_words))
    
    def _are_institutional_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are institutional related."""
        institution_terms = ["ministry", "department", "commission", "board", "authority", "corporation", 
                           "agency", "committee", "parliament", "house", "senate", "cabinet", "office"]
        
        name1 = (node1.get("name", "") + " " + node1.get("type", "") + " " + node1.get("description", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("type", "") + " " + node2.get("description", "")).lower()
        
        inst1 = any(term in name1 for term in institution_terms)
        inst2 = any(term in name2 for term in institution_terms)
        
        return inst1 and inst2
    
    def _are_geographic_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are geographic/constituency related."""
        geo_terms = ["parish", "constituency", "christ church", "st michael", "st james", "st peter", 
                    "st andrew", "st joseph", "st john", "st philip", "st lucy", "st thomas", "st george",
                    "bridgetown", "barbados", "caribbean", "region", "area", "district", "community"]
        
        name1 = (node1.get("name", "") + " " + node1.get("description", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("description", "")).lower()
        
        # Check for shared geographic terms
        for term in geo_terms:
            if term in name1 and term in name2:
                return True
                
        return False
    
    def _are_topic_domain_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are in related topic domains."""
        topic_domains = {
            "infrastructure": ["water", "roads", "transport", "utilities", "electricity", "sewage", "drainage"],
            "social": ["health", "education", "welfare", "social", "housing", "family", "children"],
            "economic": ["economy", "finance", "budget", "tax", "business", "trade", "investment", "jobs"],
            "governance": ["parliament", "government", "law", "legal", "court", "justice", "administration"],
            "environment": ["environment", "climate", "waste", "conservation", "energy", "sustainability"],
            "culture": ["culture", "arts", "music", "heritage", "festival", "tourism", "sport"]
        }
        
        name1 = (node1.get("name", "") + " " + node1.get("description", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("description", "")).lower()
        
        # Check if both nodes belong to the same topic domain
        for domain, terms in topic_domains.items():
            domain1 = any(term in name1 for term in terms)
            domain2 = any(term in name2 for term in terms)
            if domain1 and domain2:
                return True
                
        return False
    
    def _are_procedure_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are parliamentary procedure related."""
        procedure_terms = ["motion", "bill", "act", "resolution", "amendment", "debate", "question", 
                          "committee", "session", "sitting", "reading", "vote", "division", "order"]
        
        name1 = (node1.get("name", "") + " " + node1.get("type", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("type", "")).lower()
        
        proc1 = any(term in name1 for term in procedure_terms)
        proc2 = any(term in name2 for term in procedure_terms)
        
        return proc1 and proc2
    
    def _detect_temporal_intent(self, query: str) -> Dict[str, Any]:
        """Detect if user is asking for recent/temporal information."""
        temporal_keywords = [
            "recent", "recently", "latest", "current", "now", "today", 
            "this year", "2024", "2025", "new", "modern", "contemporary"
        ]
        
        query_lower = query.lower()
        temporal_detected = any(keyword in query_lower for keyword in temporal_keywords)
        
        # Define "recent" as 1 year
        from datetime import datetime, timedelta
        recent_threshold = datetime.now() - timedelta(days=365)
        
        return {
            "is_temporal_query": temporal_detected,
            "recent_threshold": recent_threshold,
            "detected_keywords": [kw for kw in temporal_keywords if kw in query_lower]
        }
    
    def _calculate_recency_boost(self, video_published_date: str, recent_threshold: datetime) -> float:
        """Calculate boost factor based on video published date."""
        if not video_published_date:
            return 1.0
        
        try:
            from datetime import datetime
            # Parse the published date
            if isinstance(video_published_date, str):
                # Handle different date formats
                pub_date = datetime.fromisoformat(video_published_date.replace('Z', '+00:00'))
            else:
                pub_date = video_published_date
            
            # Calculate days since publication
            days_old = (datetime.now() - pub_date.replace(tzinfo=None)).days
            
            if days_old < 0:  # Future date, shouldn't happen but handle gracefully
                return 1.0
            
            # Apply exponential decay: 3x boost for very recent, decaying over 1 year
            import math
            boost = max(1.0, 3.0 * math.exp(-days_old / 365))
            return boost
            
        except Exception as e:
            logger.warning(f"Error calculating recency boost: {e}")
            return 1.0

    def _filter_hop_candidates(self, connected_entity_ids: set, query: str) -> List[str]:
        """Filter hop candidates using relative degree-aware strategy to avoid hub nodes."""
        if not connected_entity_ids:
            return []
            
        # Get entity details and calculate degrees for candidates
        entity_details = list(self.entities.find(
            {"entity_id": {"$in": list(connected_entity_ids)}},
            {"entity_id": 1, "entity_name": 1, "entity_description": 1}
        ))
        
        # Calculate degrees for all candidates
        candidate_degrees = {}
        for entity_id in connected_entity_ids:
            degree = self.statements.count_documents({
                "$or": [
                    {"source_entity_id": entity_id},
                    {"target_entity_id": entity_id}
                ]
            })
            candidate_degrees[entity_id] = degree
        
        # Get degree distribution statistics from the entire graph
        degree_stats = self._get_degree_distribution_stats()
        
        # Filter and score entities using relative metrics
        scored_entities = []
        query_lower = query.lower()
        
        for entity in entity_details:
            entity_id = entity["entity_id"]
            degree = candidate_degrees.get(entity_id, 0)
            name = entity.get("entity_name", "").lower()
            desc = entity.get("entity_description", "").lower()
            
            # Skip ultra-hubs (>99th percentile)
            if degree > degree_stats["p99"]:
                continue
                
            # Calculate base relevance score from query matching
            relevance_score = 0
            query_terms = query_lower.split()
            for term in query_terms:
                if term in name or term in desc:
                    relevance_score += 10
            
            # Apply degree-based penalty/boost using relative position
            hub_penalty = self._calculate_hub_penalty(degree, degree_stats)
            relevance_score *= hub_penalty
            
            # Boost for moderate connectivity (25th-95th percentile range)
            if degree_stats["p25"] <= degree <= degree_stats["p95"]:
                relevance_score *= 1.2
                
            # Penalize isolated nodes (<5th percentile)
            if degree < degree_stats["p5"]:
                relevance_score *= 0.5
                
            scored_entities.append((entity_id, relevance_score, degree))
        
        # Sort by relevance score (descending)
        scored_entities.sort(key=lambda x: -x[1])
        
        # Return top 8 candidates
        selected_ids = [entity_id for entity_id, score, degree in scored_entities[:8]]
        
        ultra_hubs_skipped = len([d for d in candidate_degrees.values() if d > degree_stats["p99"]])
        logger.info(f"🎯 Hop filtering: {len(connected_entity_ids)} candidates → {len(selected_ids)} selected (skipped {ultra_hubs_skipped} ultra-hubs >p99={degree_stats['p99']})")
        
        return selected_ids
    
    def _get_degree_distribution_stats(self) -> Dict[str, float]:
        """Calculate degree distribution statistics for the entire graph."""
        # Use aggregation pipeline to calculate degrees efficiently
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "source_entities": {"$addToSet": "$source_entity_id"},
                    "target_entities": {"$addToSet": "$target_entity_id"}
                }
            }
        ]
        
        result = list(self.statements.aggregate(pipeline))
        if not result:
            # Fallback if no statements
            return {"mean": 0, "std": 1, "p5": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0, "p99": 0}
        
        # Get all unique entity IDs
        all_entities = set(result[0].get("source_entities", [])) | set(result[0].get("target_entities", []))
        
        # Calculate degree for each entity (sample for performance if graph is very large)
        if len(all_entities) > 10000:
            # Sample for very large graphs
            import random
            sample_entities = random.sample(list(all_entities), 5000)
        else:
            sample_entities = list(all_entities)
        
        degrees = []
        for entity_id in sample_entities:
            degree = self.statements.count_documents({
                "$or": [
                    {"source_entity_id": entity_id},
                    {"target_entity_id": entity_id}
                ]
            })
            degrees.append(degree)
        
        if not degrees:
            return {"mean": 0, "std": 1, "p5": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0, "p99": 0}
        
        # Calculate statistics
        import numpy as np
        degrees_array = np.array(degrees)
        
        return {
            "mean": float(np.mean(degrees_array)),
            "std": float(np.std(degrees_array)),
            "p5": float(np.percentile(degrees_array, 5)),
            "p25": float(np.percentile(degrees_array, 25)),
            "p50": float(np.percentile(degrees_array, 50)),
            "p75": float(np.percentile(degrees_array, 75)),
            "p95": float(np.percentile(degrees_array, 95)),
            "p99": float(np.percentile(degrees_array, 99))
        }
    
    def _calculate_hub_penalty(self, degree: int, degree_stats: Dict[str, float]) -> float:
        """Calculate hub penalty based on relative position in degree distribution."""
        mean_degree = degree_stats["mean"]
        std_degree = max(degree_stats["std"], 1)  # Avoid division by zero
        
        # Calculate z-score (how many standard deviations from mean)
        z_score = (degree - mean_degree) / std_degree
        
        # Apply smooth penalty: penalty increases exponentially with z-score
        import math
        hub_penalty = 1.0 / (1.0 + math.exp(z_score - 1))  # Sigmoid function
        
        # Ensure penalty is between 0.1 and 1.0
        return max(0.1, min(1.0, hub_penalty))

    def get_structured_search_results(self, query: str, limit: int = 20, hops: int = 1) -> Dict[str, Any]:
        """Get search results as structured JSON with focused, relevant data and temporal awareness."""
        try:
            logger.info(f"🔍 Structured search for: '{query}'")
            
            # Detect temporal intent
            temporal_analysis = self._detect_temporal_intent(query)
            logger.info(f"🕒 Temporal intent: {temporal_analysis}")
            
            # Perform hybrid search to get the most relevant initial results
            search_results = self.unified_hybrid_search(query, limit)
            if not search_results:
                return {
                    "query": query,
                    "entities": [],
                    "statements": [],
                    "provenance": {},
                    "summary": f"No results found for: {query}"
                }
            
            # Get only the top search result entity IDs (much more selective)
            seed_entity_ids = {result["uri"] for result in search_results[:min(limit*2, 40)] if "uri" in result}  # Use more seeds
            logger.info(f"Starting with {len(seed_entity_ids)} seed entities: {list(seed_entity_ids)[:10]}")
            
            # Get entities data for seeds only
            entities_data = list(self.entities.find(
                {"entity_id": {"$in": list(seed_entity_ids)}},
                {
                    "entity_id": 1,
                    "entity_name": 1, 
                    "entity_type": 1,
                    "entity_description": 1,
                    "extracted_at": 1,
                    "video_id": 1
                }
            ))
            
            # Get only high-strength statements directly involving our seed entities
            statements_pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"source_entity_id": {"$in": list(seed_entity_ids)}},
                            {"target_entity_id": {"$in": list(seed_entity_ids)}}
                        ],
                        "relationship_strength": {"$gte": 5}  # Medium to high-confidence relationships
                    }
                },
                {"$sort": {"relationship_strength": -1, "extracted_at": -1}},
                {"$limit": 150}  # Allow many more statements for richer context
            ]
            
            statements_data = list(self.statements.aggregate(statements_pipeline))
            logger.info(f"Found {len(statements_data)} statements for seed entities (includes all 1-hop connections, even to ultra-hubs)")
            
            # If we need more context and hops > 1, get 1-hop neighbors selectively
            if hops > 1 and len(statements_data) < 80:
                # Get entities connected to our seeds through high-strength relationships
                connected_entity_ids = set()
                for stmt in statements_data:
                    if stmt.get("source_entity_id") not in seed_entity_ids:
                        connected_entity_ids.add(stmt.get("source_entity_id"))
                    if stmt.get("target_entity_id") not in seed_entity_ids:
                        connected_entity_ids.add(stmt.get("target_entity_id"))
                
                # Smart filtering: avoid expanding FROM high-degree hub nodes (but keep 1-hop connections TO them)
                connected_entity_ids = self._filter_hop_candidates(connected_entity_ids, query)
                
                if connected_entity_ids:
                    # Get additional entity data
                    additional_entities = list(self.entities.find(
                        {"entity_id": {"$in": connected_entity_ids}},
                        {
                            "entity_id": 1,
                            "entity_name": 1, 
                            "entity_type": 1,
                            "entity_description": 1,
                            "extracted_at": 1,
                            "video_id": 1
                        }
                    ))
                    entities_data.extend(additional_entities)
                    
                    # Get additional high-strength statements involving these entities
                    additional_statements = list(self.statements.find(
                        {
                            "$or": [
                                {"source_entity_id": {"$in": connected_entity_ids}},
                                {"target_entity_id": {"$in": connected_entity_ids}}
                            ],
                            "relationship_strength": {"$gte": 6}  # Reasonable threshold for 2nd hop
                        },
                        {
                            "_id": 1,
                            "source_entity_id": 1,
                            "target_entity_id": 1,
                            "relationship_description": 1,
                            "relationship_strength": 1,
                            "provenance_segment_id": 1,
                            "extracted_at": 1
                        }
                    ).sort("relationship_strength", -1).limit(80))
                    
                    statements_data.extend(additional_statements)
                    logger.info(f"Added {len(additional_statements)} additional statements from {len(connected_entity_ids)} filtered hop candidates")
            
            # Collect unique segment IDs for provenance lookup (limit to top statements)
            top_statements = sorted(statements_data, 
                                  key=lambda x: x.get("relationship_strength", 0), 
                                  reverse=True)[:120]
            
            segment_ids = list(set(
                stmt.get("provenance_segment_id") 
                for stmt in top_statements 
                if stmt.get("provenance_segment_id")
            ))
            
            # Get provenance details only for the most relevant segments
            provenance_details = self.get_provenance_details(segment_ids[:15])
            logger.debug(f"Retrieved provenance details for {len(provenance_details)} segments out of {len(segment_ids[:15])} requested")
            
            # Enhance statements with provenance info and temporal scoring
            enhanced_statements = []
            recent_count = 0
            total_count = 0
            date_range = {"newest": None, "oldest": None}
            
            for stmt in statements_data:
                segment_id = stmt.get("provenance_segment_id")
                enhanced_stmt = {
                    "statement_id": str(stmt.get("_id")),
                    "source_entity_id": stmt.get("source_entity_id"),
                    "target_entity_id": stmt.get("target_entity_id"),
                    "relationship_description": stmt.get("relationship_description"),
                    "relationship_strength": stmt.get("relationship_strength", 5),
                    "provenance_segment_id": segment_id,
                    "extracted_at": stmt.get("extracted_at")
                }
                
                # Add provenance details and calculate temporal scores
                if segment_id and segment_id in provenance_details:
                    provenance = provenance_details[segment_id]
                    enhanced_stmt["provenance"] = provenance
                    logger.debug(f"Added provenance to statement: {stmt.get('relationship_description', '')[:50]}... with URL: {provenance.get('timestamped_url', 'NO_URL')}")
                    
                    # Get video published date for temporal scoring
                    video_date = None
                    if "upload_date" in provenance and provenance["upload_date"]:
                        video_date = provenance["upload_date"]
                    elif "video_title" in provenance:
                        # Try to extract date from video title if available
                        video_title = provenance["video_title"]
                        import re
                        # Look for dates in title like "Tuesday 17th March" or "2020", "2025"
                        date_match = re.search(r'(2020|2021|2022|2023|2024|2025)', video_title)
                        if date_match:
                            year = date_match.group(1)
                            # Create approximate date (mid-year if no specific date)
                            video_date = f"{year}-06-15"
                    
                    # Calculate temporal boost - always apply, but stronger for explicit temporal queries
                    temporal_boost = 1.0
                    if video_date:
                        base_boost = self._calculate_recency_boost(video_date, temporal_analysis["recent_threshold"])
                        if temporal_analysis["is_temporal_query"]:
                            # Full boost for explicit temporal queries
                            temporal_boost = base_boost
                        else:
                            # Lighter boost for all queries (1.0 to 1.5x range)
                            temporal_boost = 1.0 + (base_boost - 1.0) * 0.5
                        
                        # Track recent vs total content
                        total_count += 1
                        if temporal_boost > 1.5:  # Significantly boosted = recent
                            recent_count += 1
                        
                        # Track date range
                        if video_date:
                            if not date_range["newest"] or video_date > date_range["newest"]:
                                date_range["newest"] = video_date
                            if not date_range["oldest"] or video_date < date_range["oldest"]:
                                date_range["oldest"] = video_date
                    
                    # Apply temporal boost to relationship strength
                    enhanced_stmt["temporal_boost"] = temporal_boost
                    enhanced_stmt["boosted_strength"] = enhanced_stmt["relationship_strength"] * temporal_boost
                    enhanced_stmt["video_date"] = video_date
                
                enhanced_statements.append(enhanced_stmt)
            
            # Always sort by temporal-boosted relevance (but lighter boost for non-temporal queries)
            enhanced_statements.sort(
                key=lambda x: (
                    x.get("boosted_strength", x.get("relationship_strength", 0)),
                    x.get("temporal_boost", 1.0),
                    x.get("extracted_at", "")
                ), 
                reverse=True
            )
            
            if temporal_analysis["is_temporal_query"]:
                logger.info(f"🕒 Full temporal sorting applied with {recent_count}/{total_count} recent results")
            else:
                logger.info(f"🕒 Light temporal boost applied with {recent_count}/{total_count} recent results")
            
            enhanced_statements = enhanced_statements[:100]  # Increased limit to capture more relationships
            
            # Build temporal metadata for the LLM
            temporal_metadata = {
                "query_requested_recent": temporal_analysis["is_temporal_query"],
                "recent_content_found": recent_count,
                "total_content_found": total_count,
                "recent_threshold": temporal_analysis["recent_threshold"].isoformat() if temporal_analysis["recent_threshold"] else None,
                "newest_content_date": date_range["newest"],
                "oldest_content_date": date_range["oldest"],
                "detected_keywords": temporal_analysis["detected_keywords"]
            }
            
            result = {
                "query": query,
                "entities": entities_data[:20],  # Limit entities too
                "statements": enhanced_statements,
                "provenance": provenance_details,
                "temporal_analysis": temporal_metadata,
                "summary": f"Found {len(entities_data)} entities and {len(enhanced_statements)} high-relevance statements",
                "search_metadata": {
                    "total_entities": len(entities_data),
                    "total_statements": len(enhanced_statements),
                    "provenance_segments": len(provenance_details),
                    "hops": hops,
                    "limit": limit,
                    "optimization": "focused_high_strength_temporal"
                }
            }
            
            logger.info(f"🎯 Optimized search complete: {len(entities_data)} entities, {len(enhanced_statements)} statements")
            return result
            
        except Exception as e:
            logger.error(f"❌ Structured search failed: {e}")
            return {
                "query": query,
                "entities": [],
                "statements": [],
                "provenance": {},
                "summary": f"Error searching for: {query}",
                "error": str(e)
            }

    def _streamline_results_for_llm(self, full_results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert full search results to minimal structure for LLM."""
        try:
            # Streamline entities
            minimal_entities = []
            for entity in full_results.get("entities", [])[:20]:
                minimal_entity = {
                    "id": entity.get("entity_id"),
                    "name": entity.get("entity_name"),
                    "type": entity.get("entity_type")
                }
                # Only include description if it's meaningful and not too long
                desc = entity.get("entity_description", "")
                if desc and len(desc) < 200 and desc.lower() != "unknown":
                    minimal_entity["description"] = desc[:200]
                minimal_entities.append(minimal_entity)
            
            # Streamline statements with inline provenance
            minimal_statements = []
            provenance_data = full_results.get("provenance", {})
            
            for stmt in full_results.get("statements", [])[:100]:
                minimal_stmt = {
                    "source": stmt.get("source_entity_id"),
                    "target": stmt.get("target_entity_id"),
                    "relationship": stmt.get("relationship_description"),
                    "strength": stmt.get("relationship_strength", 5)
                }
                
                # Try to get provenance from statement first
                if "provenance" in stmt and stmt["provenance"]:
                    prov = stmt["provenance"]
                    # Look for timestamped_url first (with timestamp), then fallback to video_url
                    if "timestamped_url" in prov:
                        minimal_stmt["source_url"] = prov["timestamped_url"]
                    elif "video_url" in prov:
                        minimal_stmt["source_url"] = prov["video_url"]
                    elif "youtube_url" in prov:  # Legacy field support
                        minimal_stmt["source_url"] = prov["youtube_url"]
                    if "video_title" in prov:
                        minimal_stmt["source_title"] = prov["video_title"]
                # Otherwise try to get from provenance_details using segment_id
                elif "provenance_segment_id" in stmt and stmt["provenance_segment_id"] in provenance_data:
                    prov = provenance_data[stmt["provenance_segment_id"]]
                    # Look for timestamped_url first (with timestamp), then fallback to video_url  
                    if "timestamped_url" in prov:
                        minimal_stmt["source_url"] = prov["timestamped_url"]
                    elif "video_url" in prov:
                        minimal_stmt["source_url"] = prov["video_url"]
                    elif "youtube_url" in prov:  # Legacy field support
                        minimal_stmt["source_url"] = prov["youtube_url"]
                    if "video_title" in prov:
                        minimal_stmt["source_title"] = prov["video_title"]
                        
                # Debug log if no URL found but title exists
                if "source_title" in minimal_stmt and "source_url" not in minimal_stmt:
                    logger.debug(f"Statement has title but no URL: {minimal_stmt['relationship'][:50]}...")
                    # Show available provenance fields for debugging
                    if "provenance" in stmt and stmt["provenance"]:
                        prov_keys = list(stmt["provenance"].keys())
                        logger.debug(f"Available provenance fields: {prov_keys}")
                    elif "provenance_segment_id" in stmt and stmt["provenance_segment_id"] in provenance_data:
                        prov_keys = list(provenance_data[stmt["provenance_segment_id"]].keys())
                        logger.debug(f"Available provenance_data fields: {prov_keys}")
                    else:
                        logger.debug(f"No provenance data found for segment_id: {stmt.get('provenance_segment_id', 'NONE')}")
                
                minimal_statements.append(minimal_stmt)
            
            # Simple temporal context
            temporal_analysis = full_results.get("temporal_analysis", {})
            date_range = None
            if temporal_analysis.get("oldest_content_date") and temporal_analysis.get("newest_content_date"):
                oldest_year = temporal_analysis["oldest_content_date"][:4]
                newest_year = temporal_analysis["newest_content_date"][:4]
                date_range = f"{oldest_year}-{newest_year}"
            
            minimal_temporal = {
                "found_recent": temporal_analysis.get("recent_content_found", 0) > 0,
                "date_range": date_range
            }
            
            # Build minimal result
            minimal_result = {
                "query": full_results.get("query"),
                "entities": minimal_entities,
                "statements": minimal_statements,
                "temporal_context": minimal_temporal,
                "summary": f"Found {len(minimal_entities)} entities and {len(minimal_statements)} relationships"
            }
            
            return minimal_result
            
        except Exception as e:
            logger.error(f"Error streamlining results: {e}")
            # Return original if streamlining fails
            return full_results

# In your chatbot.py file...

class ParliamentarySystem:
    """Main system for parliamentary research, integrating agent, querier, and session management."""

    def __init__(self, google_api_key: Optional[str] = None):
        self.google_api_key = google_api_key
        self.querier = ParliamentaryGraphQuerier()
        self.session_manager = MongoSessionManager()
        self.session_graphs: Dict[str, SessionGraphState] = {}
        self.agent = self._create_agent()
        self.runner = InMemoryRunner(agent=self.agent)
        logger.info("✅ ParliamentarySystem initialized with MongoDB session manager")

    def _create_agent(self) -> LlmAgent:
        """Create and configure the LLM agent."""
        search_tool = FunctionTool(
            func=self.querier.get_structured_search_results
        )
        
        planner = BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=0))

        return LlmAgent(
            name="TrackSabha",
            model="gemini-flash-latest",
            description="AI assistant for the Indian Parliament using parliamentary transcripts and provenance",
            planner=planner,
            instruction="""You are TrackSabha, an expert AI assistant for the Indian Parliament.
Your role is to provide accurate, factual, and well-sourced answers based on parliamentary transcripts and data.
The user's input may contain previous conversation history for context. Focus on answering the newest user query at the end.
Always use the provided `get_structured_search_results` tool to find information to answer the user's query.
When you receive search results, synthesize them into a clear, structured JSON response that matches the required format.
Your response should be in markdown format and include:
1.  An introductory summary.
2.  Detailed "Response Cards" for each key finding, with a one-sentence summary and full details.
3.  Direct quotes and links to the source video with timestamps when available.
4.  A list of follow-up questions.
Do not invent information. If the search results do not contain the answer, state that clearly.
""",
            tools=[search_tool],
            generate_content_config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8000,
                response_mime_type="application/json",
            ),
        )

    async def process_query(self, query: str, user_id: str, session_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Process a user query, manage session, and return a structured response."""
        try:
            session = await self.session_manager.ensure_session(user_id, session_id)
            session_id = session.session_id
            session_graph = self._get_session_graph(session_id)
            
            # Log the new user message first
            await self.session_manager.add_message(session_id, "user", query)
            
            # This call will now work correctly because we fixed the session_manager
            chat_history = await self.get_session_history(session_id, limit=10)
            
            # Manually construct a single prompt with history because the runner API is stateless
            history_str_parts = []
            if chat_history:
                history_str_parts.append("Previous conversation history for context:")
                for entry in reversed(chat_history): # Oldest first
                    role = "User" if entry.role == "user" else "Assistant"
                    text_content = " ".join(p.text for p in entry.parts if hasattr(p, 'text'))
                    if text_content:
                        history_str_parts.append(f"{role}: {text_content}")
            
            history_str_parts.append("\nNew User Query:")
            history_str_parts.append(query)
            
            prompt_with_context = "\n".join(history_str_parts)

            # This call is correct: a single positional argument with no keywords.
            agent_response_coro = self.runner.run_async(prompt_with_context)
            
            agent_response_str = await agent_response_coro
            
            try:
                repaired_json_str = repair_json(agent_response_str)
                structured_response_data = json.loads(repaired_json_str)
                structured_response = StructuredResponse(**structured_response_data)
                
                await self.session_manager.add_message(
                    session_id, "assistant", agent_response_str, metadata={"parsed_successfully": True}
                )
                
                return agent_response_str, {
                    "success": True, 
                    "session_id": session_id, 
                    "graph_stats": session_graph.get_stats(),
                    "structured_response": structured_response,
                    "response_type": "structured"
                }

            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning(f"Could not parse agent response as JSON: {e}. Falling back to text. Response was: {agent_response_str}")
                await self.session_manager.add_message(
                    session_id, "assistant", agent_response_str, metadata={"parsed_successfully": False, "error": str(e)}
                )
                return agent_response_str, {
                    "success": True, 
                    "session_id": session_id, 
                    "graph_stats": session_graph.get_stats(),
                    "response_type": "text"
                }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            if session_id:
                await self.session_manager.add_message(session_id, "assistant", f"An error occurred: {e}", metadata={"is_error": True})
            return f"An error occurred: {e}", {"success": False, "session_id": session_id}

    def _get_session_graph(self, session_id: str) -> SessionGraphState:
        """Get or create a session graph."""
        if session_id not in self.session_graphs:
            self.session_graphs[session_id] = SessionGraphState(session_id)
        return self.session_graphs[session_id]

    async def get_session_history(self, session_id: str, limit: int = 10) -> List[Content]:
        """Retrieve chat history for a given session."""
        return await self.session_manager.get_history_as_content(session_id, limit)

    async def get_user_sessions(self, user_id: str, limit: int = 10, include_archived: bool = False) -> List[Dict[str, Any]]:
        """Get user's recent sessions."""
        sessions = await self.session_manager.get_user_sessions(user_id, limit, include_archived)
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat(),
                "last_updated": s.last_updated.isoformat(),
                "message_count": len(s.messages),
                "archived": s.metadata.get("archived", False)
            }
            for s in sessions
        ]

    async def archive_session(self, session_id: str, reason: str) -> bool:
        """Archive a session."""
        success = await self.session_manager.archive_session(session_id, reason)
        if success and session_id in self.session_graphs:
            del self.session_graphs[session_id]  # Remove from memory
        return success

    def visualize_session_graph(self, session_id: str) -> str:
        """Generate an HTML visualization of the session graph."""
        if session_id not in self.session_graphs:
            raise ValueError("Session graph not found")
        
        graph_data = self.session_graphs[session_id].get_json_dump()
        
        # Render the template with graph data
        return templates.get_template("graph_visualization.html").render({
            "request": None,
            "graph_data_json": json.dumps(graph_data, indent=2)
        })

    def close(self):
        """Clean up resources."""
        if self.querier and self.querier.client:
            self.querier.client.close()
        if self.session_manager and self.session_manager.client:
            self.session_manager.client.close()
        logger.info("ParliamentarySystem resources closed.")

def create_system() -> "ParliamentarySystem":
    """Create the parliamentary system."""
    google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not google_api_key:
        logger.warning("GOOGLE_API_KEY or GEMINI_API_KEY not set — continuing without Google credentials. Embeddings and some agent features will be disabled until credentials are provided.")
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
    title="TrackSabha - Hindi Parliament Research API",
    description="TrackSabha: AI-powered indian Parliament research system with MongoDB session persistence and interactive graph visualization",
    version="4.1.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

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
        "message": "TrackSabha - indian Parliament Research API with MongoDB Session Storage and Graph Visualization",
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
    except Exception as e:
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
    except Exception:
        raise

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
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        html_content = parliamentary_system.visualize_session_graph(session_id)
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Failed to visualize session graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        "chatbot:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )