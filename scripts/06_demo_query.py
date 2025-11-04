#!/usr/bin/env python3
"""
MongoDB Graph Query Tool with Vector Search Support

This script queries the MongoDB graph database using both text and vector search,
returning relevant nodes with configurable traversal depth, outputting results as RDF Turtle.

Requirements:
- pymongo
- python-dotenv (optional, for environment variables)
- sentence-transformers (optional, for vector search)

Usage:
    python query_graph.py "query string" [--hops N] [--output file.ttl] [--vector-only] [--text-only]

Examples:
    # Hybrid search (text + local vector model)
    python .\scripts\06_demo_query.py "digital media" --db parliamentary_graph --hops 2

    # Vector-only search using local sentence-transformers model
    python .\scripts\06_demo_query.py "digital media" --vector-only --db parliamentary_graph

    # Use Google GenAI to create the query embedding at runtime (requires google.genai creds)
    python .\scripts\06_demo_query.py "digital media" --vector-only --google-embeddings --db parliamentary_graph
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from datetime import datetime, timezone
import re
import time
import random

try:
    from pymongo import MongoClient, TEXT
    from pymongo.errors import ConnectionFailure
    from dotenv import load_dotenv
    from rdflib import Graph, URIRef, Literal, BNode, Namespace
    from rdflib.namespace import RDF, RDFS, OWL, FOAF, XSD
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install pymongo python-dotenv rdflib")
    sys.exit(1)

# Optional: Import sentence transformers for vector search
try:
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

# Load environment variables
load_dotenv()

class GraphQuerier:
    def __init__(self, connection_string: str = None, database_name: str = "parliamentary_graph", use_google_embeddings: bool = False):
        """
        Initialize the graph querier.
        
        Args:
            connection_string: MongoDB Atlas connection string
            database_name: Name of the MongoDB database
        """
        if connection_string is None:
            connection_string = os.getenv('MONGODB_CONNECTION_STRING')
            
        if not connection_string:
            raise ValueError(
                "MongoDB connection string is required. Set MONGODB_CONNECTION_STRING environment variable."
            )
        
        try:
            self.client = MongoClient(connection_string)
            self.client.admin.command('ping')
            print("✅ Connected to MongoDB Atlas")
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")
        
        self.db = self.client[database_name]
        self.nodes = self.db.nodes
        self.edges = self.db.edges
        self.statements = self.db.statements
        self.videos = self.db.videos
        
        # Embedding configuration
        self.use_google_embeddings = use_google_embeddings
        self.google_client = None

        # Initialize embedding model for vector search (sentence-transformers fallback)
        self.embedding_model = None
        if VECTOR_SEARCH_AVAILABLE:
            try:
                print("🔄 Loading embedding model for vector search...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Vector search enabled (local model)")
            except Exception as e:
                print(f"⚠️  Vector search disabled: {e}")

        # Initialize Google GenAI client if requested
        if self.use_google_embeddings:
            try:
                from google import genai
                self.google_client = genai.Client()
                print("✅ Google GenAI client initialized for query embeddings")
            except Exception as e:
                print(f"⚠️  Failed to initialize Google GenAI client: {e}")
                print("Falling back to local sentence-transformers if available")
                self.google_client = None
    
    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for the search query."""
        # Try Google GenAI first if enabled
        if self.use_google_embeddings and self.google_client is not None:
            # Retry loop to handle transient quota/rate-limit errors
            max_retries = 5
            base_delay = 1.0
            attempt = 0
            result = None
            while attempt <= max_retries:
                try:
                    result = self.google_client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=[query]
                    )
                    break
                except Exception as e:
                    msg = str(e)
                    if '429' in msg or 'RESOURCE_EXHAUSTED' in msg or 'quota' in msg.lower():
                        sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        print(f"⚠️  Rate limit encountered (attempt {attempt+1}/{max_retries}). Sleeping {sleep_time:.1f}s before retry...")
                        time.sleep(sleep_time)
                        attempt += 1
                        continue
                    else:
                        # Non-rate-limit error: surface it and fall back
                        print(f"⚠️  Google embedding generation failed: {e}")
                        result = None
                        break

            # Normalize possible genai response shapes to list of floats
            if result is not None:
                emb_obj = getattr(result, "embeddings", None)
                if emb_obj is None:
                    try:
                        emb_obj = result["embeddings"]
                    except Exception:
                        emb_obj = None

                if emb_obj is None:
                    print("⚠️  Google embedding response missing 'embeddings'")
                else:
                    emb_entry = emb_obj[0] if isinstance(emb_obj, (list, tuple)) and len(emb_obj) > 0 else emb_obj

                    # ContentEmbedding-like objects expose `.values`
                    try:
                        vals = getattr(emb_entry, "values", None)
                        if vals is not None and isinstance(vals, (list, tuple)):
                            return [float(x) for x in vals]
                    except Exception:
                        pass

                    # If it's a flat list/tuple of numbers or numeric-like tuples (index, value)
                    if isinstance(emb_entry, (list, tuple)):
                        normalized = []
                        for x in emb_entry:
                            candidate = x
                            if isinstance(x, tuple) and len(x) >= 2:
                                # Some responses contain (index, value) pairs
                                candidate = x[1]
                            try:
                                normalized.append(float(candidate))
                            except Exception:
                                # skip non-numeric entries
                                continue
                        if len(normalized) > 0:
                            return normalized

                    # If it's a dict, look for common keys
                    if isinstance(emb_entry, dict):
                        for key in ("embedding", "values", "vector", "data", "value"):
                            val = emb_entry.get(key)
                            if isinstance(val, (list, tuple)):
                                try:
                                    return [float(x) for x in val]
                                except Exception:
                                    # try to normalize tuples inside
                                    normalized = []
                                    for x in val:
                                        if isinstance(x, tuple) and len(x) >= 2:
                                            candidate = x[1]
                                        else:
                                            candidate = x
                                        try:
                                            normalized.append(float(candidate))
                                        except Exception:
                                            continue
                                    if len(normalized) > 0:
                                        return normalized

                    # Fallback: try casting to list
                    try:
                        return [float(x) for x in list(emb_entry)]
                    except Exception:
                        print("⚠️  Could not normalize Google embedding response")
                        # fall through to local model fallback

        # Fallback to local sentence-transformers
        if not self.embedding_model:
            return None

        try:
            embedding = self.embedding_model.encode(query)
            try:
                return embedding.tolist()
            except Exception:
                return list(embedding)
        except Exception as e:
            print(f"⚠️  Failed to generate query embedding: {e}")
            return None
    
    def vector_search_nodes(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Perform vector search on node embeddings.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching node documents with similarity scores
        """
        # Allow vector search when either a local embedding model is available
        # or Google GenAI embeddings are explicitly enabled and the client exists.
        if not (self.embedding_model or (self.use_google_embeddings and self.google_client is not None)):
            print("⚠️  Vector search not available (no embedding method configured: local model or Google GenAI client)")
            return []
        
        print(f"🔍 Vector searching for: '{query}'")
        
        # Generate embedding for query
        query_embedding = self.generate_query_embedding(query)
        if not query_embedding:
            return []
        try:
            # MongoDB Atlas Vector Search aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index_1",  # Name of your vector search index
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 2,  # Search more candidates for better results
                        "limit": limit
                    }
                },
                {
                    "$addFields": {
                        "similarity_score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(self.nodes.aggregate(pipeline))
            print(f"📍 Found {len(results)} vector matches")
            return results
            
        except Exception as e:
            print(f"⚠️  Vector search failed: {e}")
            print("   Make sure you have created a vector search index named 'vector_index'")
            return []
    
    def text_search_nodes(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Perform text search on node labels and properties.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching node documents
        """
        print(f"🔍 Text searching for: '{query}'")
        
        # Text search on indexed fields (label, searchable_text)
        text_results = []
        try:
            text_results = list(self.nodes.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit))
        except Exception as e:
            print(f"⚠️  Full-text search failed: {e}")
        
        # Also search using regex on specific fields
        regex_pattern = re.compile(re.escape(query), re.IGNORECASE)
        regex_results = list(self.nodes.find({
            "$or": [
                {"label": {"$regex": regex_pattern}},
                {"local_name": {"$regex": regex_pattern}},
                {"searchable_text": {"$regex": regex_pattern}},
                # Search in specific property values
                {"properties.http://schema.org/name": {"$regex": regex_pattern}},
                {"properties.http://www.w3.org/2000/01/rdf-schema#label": {"$regex": regex_pattern}},
                {"properties.http://xmlns.com/foaf/0.1/name": {"$regex": regex_pattern}}
            ]
        }).limit(limit))
        
        # Combine and deduplicate results
        seen_uris = set()
        combined_results = []
        
        # Add text search results first (they have relevance scores)
        for result in text_results:
            if result["uri"] not in seen_uris:
                seen_uris.add(result["uri"])
                result["search_type"] = "text_index"
                combined_results.append(result)
        
        # Add regex results
        for result in regex_results:
            if result["uri"] not in seen_uris:
                seen_uris.add(result["uri"])
                result["search_type"] = "regex"
                combined_results.append(result)
        
        print(f"📍 Found {len(combined_results)} text matches")
        return combined_results
    
    def hybrid_search_nodes(self, query: str, limit: int = 5, vector_weight: float = 0.7) -> List[Dict]:
        """
        Perform hybrid search combining vector and text search results.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            vector_weight: Weight for vector search results (0.0 to 1.0)
            
        Returns:
            List of matching node documents with combined scores
        """
        print(f"🔍 Hybrid searching for: '{query}'")
        
        # Get results from both search methods
        vector_results = self.vector_search_nodes(query, limit)
        text_results = self.text_search_nodes(query, limit)
        
        # Combine results with weighted scoring
        combined_results = {}
        
        # Add vector search results
        for result in vector_results:
            uri = result["uri"]
            vector_score = result.get("similarity_score", 0.0)
            result["vector_score"] = vector_score
            result["text_score"] = 0.0
            result["hybrid_score"] = vector_score * vector_weight
            result["search_types"] = ["vector"]
            combined_results[uri] = result
        
        # Add/merge text search results
        for result in text_results:
            uri = result["uri"]
            text_score = result.get("score", 0.5)  # Default score for regex matches
            
            if uri in combined_results:
                # Merge with existing vector result
                combined_results[uri]["text_score"] = text_score
                combined_results[uri]["hybrid_score"] = (
                    combined_results[uri]["vector_score"] * vector_weight + 
                    text_score * (1 - vector_weight)
                )
                combined_results[uri]["search_types"].append(result.get("search_type", "text"))
            else:
                # New text-only result
                result["vector_score"] = 0.0
                result["text_score"] = text_score
                result["hybrid_score"] = text_score * (1 - vector_weight)
                result["search_types"] = [result.get("search_type", "text")]
                combined_results[uri] = result
        
        # Sort by hybrid score and return top results
        sorted_results = sorted(
            combined_results.values(), 
            key=lambda x: x["hybrid_score"], 
            reverse=True
        )[:limit]
        
        print(f"📍 Found {len(sorted_results)} hybrid matches")
        return sorted_results
    
    def search_nodes(self, query: str, search_mode: str = "hybrid", limit: int = 5) -> List[Dict]:
        """
        Search for nodes using the specified search mode.
        
        Args:
            query: Search query string
            search_mode: "hybrid", "vector", or "text"
            limit: Maximum number of results
            
        Returns:
            List of matching node documents
        """
        if search_mode == "vector":
            return self.vector_search_nodes(query, limit)
        elif search_mode == "text":
            return self.text_search_nodes(query, limit)
        else:  # hybrid
            return self.hybrid_search_nodes(query, limit)
    
    def get_connected_nodes(self, node_uris: Set[str], hops: int = 1) -> Set[str]:
        """
        Get all nodes connected to the given nodes within specified hops.
        
        Args:
            node_uris: Set of starting node URIs
            hops: Number of hops to traverse
            
        Returns:
            Set of all connected node URIs
        """
        if hops <= 0:
            return node_uris
        
        current_nodes = set(node_uris)
        all_nodes = set(node_uris)
        
        for hop in range(hops):
            print(f"🔗 Traversing hop {hop + 1}/{hops} (current: {len(current_nodes)} nodes)")
            
            if not current_nodes:
                break
            
            # Find all edges connected to current nodes
            connected_edges = self.edges.find({
                "$or": [
                    {"subject": {"$in": list(current_nodes)}},
                    {"object": {"$in": list(current_nodes)}}
                ]
            })
            
            next_nodes = set()
            for edge in connected_edges:
                next_nodes.add(edge["subject"])
                next_nodes.add(edge["object"])
            
            # Only add truly new nodes for next iteration
            new_nodes = next_nodes - all_nodes
            all_nodes.update(next_nodes)
            current_nodes = new_nodes
            
            print(f"   Added {len(new_nodes)} new nodes (total: {len(all_nodes)})")
        
        return all_nodes
    
    def get_subgraph(self, node_uris: Set[str]) -> Dict[str, Any]:
        """
        Extract subgraph containing specified nodes and their connections.
        
        Args:
            node_uris: Set of node URIs to include
            
        Returns:
            Dictionary containing nodes, edges, and statements
        """
        print(f"📊 Extracting subgraph for {len(node_uris)} nodes")
        
        # Get node details
        nodes = list(self.nodes.find({"uri": {"$in": list(node_uris)}}))
        
        # Get edges between these nodes
        edges = list(self.edges.find({
            "subject": {"$in": list(node_uris)},
            "object": {"$in": list(node_uris)}
        }))
        
        # Get ALL reified statements that involve any of these nodes
        # (not just edges between them)
        statements = list(self.statements.find({
            "$or": [
                {"subject": {"$in": list(node_uris)}},
                {"object": {"$in": list(node_uris)}}
            ]
        }))
        
        print(f"   Nodes: {len(nodes)}, Edges: {len(edges)}, Statements: {len(statements)}")
        
        return {
            "nodes": nodes,
            "edges": edges,
            "statements": statements
        }
    
    def query_graph(self, query: str, hops: int = 2, search_mode: str = "hybrid") -> Dict[str, Any]:
        """
        Perform complete graph query with traversal.
        
        Args:
            query: Search query string
            hops: Number of hops to traverse
            search_mode: "hybrid", "vector", or "text"
            
        Returns:
            Subgraph data with search metadata
        """
        # Find initial matching nodes
        initial_nodes = self.search_nodes(query, search_mode)
        
        if not initial_nodes:
            print("❌ No matching nodes found")
            return {"nodes": [], "edges": [], "statements": [], "search_results": []}
        
        # Show top search results
        print(f"\n🎯 Top search matches:")
        for i, node in enumerate(initial_nodes[:5], 1):
            score_info = []
            if "hybrid_score" in node:
                score_info.append(f"hybrid: {node['hybrid_score']:.3f}")
            if "similarity_score" in node:
                score_info.append(f"vector: {node['similarity_score']:.3f}")
            if "score" in node:
                score_info.append(f"text: {node['score']:.3f}")
            
            score_str = f" ({', '.join(score_info)})" if score_info else ""
            search_types = node.get("search_types", ["unknown"])
            print(f"   {i}. {node.get('label', node.get('local_name', 'Unknown'))}{score_str}")
            print(f"      URI: {node['uri']}")
            print(f"      Search: {', '.join(search_types)}")
        
        # Extract URIs from initial results
        initial_uris = {node["uri"] for node in initial_nodes}
        
        # Perform graph traversal
        all_node_uris = self.get_connected_nodes(initial_uris, hops)
        
        # Extract subgraph
        subgraph = self.get_subgraph(all_node_uris)
        
        # Add search results metadata
        subgraph["search_results"] = initial_nodes
        
        return subgraph
    
    def subgraph_to_rdf_graph(self, subgraph: Dict[str, Any]) -> Graph:
        """
        Convert subgraph data to RDFLib Graph for proper deduplication and formatting.
        
        Args:
            subgraph: Subgraph data from query
            
        Returns:
            RDFLib Graph object
        """
        # Create new RDF graph
        graph = Graph()
        
        # Bind prefixes for the Indian Parliament ontology
        LOK = Namespace("http://example.com/Indian-parliament-ontology#")
        SESS = Namespace("http://example.com/Indian-parliament-session/")
        SCHEMA = Namespace("http://schema.org/")
        ORG = Namespace("http://www.w3.org/ns/org#")
        PROV = Namespace("http://www.w3.org/ns/prov#")
        BBP = Namespace("http://example.com/JanSetu.in/")

        graph.bind("lok", LOK)
        graph.bind("bbp", BBP)
        graph.bind("sess", SESS)
        graph.bind("schema", SCHEMA)
        graph.bind("org", ORG)
        graph.bind("prov", PROV)
        graph.bind("foaf", FOAF)
        graph.bind("owl", OWL)
        graph.bind("rdf", RDF)
        graph.bind("rdfs", RDFS)
        graph.bind("xsd", XSD)
        
        # Helper function to convert string to RDF term
        def string_to_rdf_term(value_str: str):
            if value_str.startswith('http://') or value_str.startswith('https://'):
                return URIRef(value_str)
            elif value_str.startswith('_:'):
                # Blank node
                return BNode(value_str[2:])  # Remove '_:' prefix
            else:
                # Try to detect if it's a number
                try:
                    if '.' in value_str:
                        float(value_str)
                        return Literal(value_str, datatype=XSD.decimal)
                    else:
                        int(value_str)
                        return Literal(value_str, datatype=XSD.integer)
                except ValueError:
                    # Check if it's a year (4 digits)
                    if re.match(r'^\d{4}$', value_str):
                        return Literal(value_str, datatype=XSD.gYear)
                    # Check if it's a date
                    elif re.match(r'^\d{4}-\d{2}-\d{2}$', value_str):
                        return Literal(value_str, datatype=XSD.date)
                    else:
                        # It's a string literal
                        return Literal(value_str)
        
        # Add node type declarations and properties
        for node in subgraph["nodes"]:
            node_uri = URIRef(node["uri"])
            
            # Add type declarations
            for node_type in node.get("type", []):
                type_uri = URIRef(node_type)
                graph.add((node_uri, RDF.type, type_uri))
            
            # Add label if available
            if "label" in node:
                graph.add((node_uri, RDFS.label, Literal(str(node["label"]))))
            
            # Add other properties (excluding internal MongoDB fields)
            for prop_uri, prop_values in node.get("properties", {}).items():
                if prop_uri == str(RDF.type):
                    continue  # Already handled above
                
                prop_ref = URIRef(prop_uri)
                
                # Handle both single values and lists
                values = prop_values if isinstance(prop_values, list) else [prop_values]
                
                for value in values:
                    value_term = string_to_rdf_term(str(value))
                    graph.add((node_uri, prop_ref, value_term))
        
        # Add edges (triples)
        for edge in subgraph["edges"]:
            subject = URIRef(edge["subject"])
            predicate = URIRef(edge["predicate"])
            obj = string_to_rdf_term(edge["object"])
            graph.add((subject, predicate, obj))
        
        # Add reified statements with enhanced provenance
        for stmt in subgraph["statements"]:
            if "statement_uri" in stmt and stmt["statement_uri"].startswith("_:"):
                # Use the original blank node ID
                stmt_node = BNode(stmt["statement_uri"][2:])  # Remove '_:' prefix
            else:
                # Generate a new blank node
                stmt_node = BNode(f"stmt_{stmt['statement_id'][:8]}")
            
            # Add reification triples
            graph.add((stmt_node, RDF.type, RDF.Statement))
            graph.add((stmt_node, RDF.subject, URIRef(stmt["subject"])))
            graph.add((stmt_node, RDF.predicate, URIRef(stmt["predicate"])))
            graph.add((stmt_node, RDF.object, string_to_rdf_term(stmt["object"])))
            
            # Add enhanced provenance using prov:wasDerivedFrom structure
            if any(key in stmt for key in ["from_video", "start_offset", "end_offset"]):
                # Create a blank node for the transcript segment
                segment_node = BNode(f"segment_{stmt['statement_id'][:8]}")
                
                # Link statement to segment
                graph.add((stmt_node, PROV.wasDerivedFrom, segment_node))
                
                # Add segment type
                graph.add((segment_node, RDF.type, BBP.TranscriptSegment))
                
                # Add segment properties
                if stmt.get("from_video"):
                    graph.add((segment_node, BBP.fromVideo, URIRef(stmt["from_video"])))
                
                if stmt.get("start_offset") is not None:
                    graph.add((segment_node, BBP.startTimeOffset, Literal(stmt["start_offset"], datatype=XSD.decimal)))
                
                if stmt.get("end_offset") is not None:
                    graph.add((segment_node, BBP.endTimeOffset, Literal(stmt["end_offset"], datatype=XSD.decimal)))
        
        return graph

    def subgraph_to_turtle(self, subgraph: Dict[str, Any]) -> str:
        """
        Convert subgraph data to Turtle format using RDFLib for proper formatting.
        
        Args:
            subgraph: Subgraph data from query
            
        Returns:
            Turtle format string
        """
        # Convert to RDF graph first
        rdf_graph = self.subgraph_to_rdf_graph(subgraph)
        
        # Count statements with provenance
        statements_with_provenance = sum(1 for stmt in subgraph['statements'] 
                                       if any(key in stmt for key in ["start_offset", "end_offset"]))
        
        # Search results summary
        search_summary = ""
        if "search_results" in subgraph:
            search_summary = f"# Initial search results: {len(subgraph['search_results'])}\n"
        
        # Add comment header
        header_comment = f"""# Query results generated at {datetime.now(timezone.utc).isoformat()}Z
{search_summary}# Nodes: {len(subgraph['nodes'])}, Edges: {len(subgraph['edges'])}, Statements: {len(subgraph['statements'])}
# Statements with provenance: {statements_with_provenance}
# Total triples: {len(rdf_graph)}

"""
        
        # Serialize to turtle with proper formatting
        turtle_content = rdf_graph.serialize(format='turtle')
        
        # Combine header and content
        return header_comment + turtle_content
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        stats = {
            "nodes": self.nodes.count_documents({}),
            "edges": self.edges.count_documents({}),
            "statements": self.statements.count_documents({}),
            "videos": self.videos.count_documents({})
        }
        
        # Check for vector search capabilities
        nodes_with_embeddings = self.nodes.count_documents({"embedding": {"$exists": True}})
        stats["nodes_with_embeddings"] = nodes_with_embeddings
        
        return stats
    
    def get_provenance_stats(self) -> Dict[str, Any]:
        """Get statistics about provenance data."""
        total_statements = self.statements.count_documents({})
        statements_with_offsets = self.statements.count_documents({
            "$and": [
                {"start_offset": {"$ne": None}},
                {"end_offset": {"$ne": None}}
            ]
        })
        
        # Sample a few statements with provenance
        sample_statements = list(self.statements.find({
            "start_offset": {"$ne": None}
        }).limit(3))
        
        return {
            "total_statements": total_statements,
            "statements_with_offsets": statements_with_offsets,
            "coverage_percentage": (statements_with_offsets / total_statements * 100) if total_statements > 0 else 0,
            "sample_statements": sample_statements
        }

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Query MongoDB graph with vector and text search")
    parser.add_argument("--db", type=str, default="youtube_data", help="MongoDB database name")

    parser.add_argument("query", help="Search query string")
    parser.add_argument("--hops", type=int, default=2, help="Number of traversal hops (default: 2)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--provenance", action="store_true", help="Show provenance statistics")
    parser.add_argument("--google-embeddings", action="store_true", help="Use Google GenAI to create the query embedding at runtime")
    
    # Search mode options
    search_group = parser.add_mutually_exclusive_group()
    search_group.add_argument("--vector-only", action="store_true", help="Use only vector search")
    search_group.add_argument("--text-only", action="store_true", help="Use only text search")
    # Default is hybrid search
    
    args = parser.parse_args()
    
    # Determine search mode
    if args.vector_only:
        search_mode = "vector"
    elif args.text_only:
        search_mode = "text"
    else:
        search_mode = "hybrid"
    
    try:
        # Initialize querier
        querier = GraphQuerier(database_name=args.db, use_google_embeddings=args.google_embeddings)

        if args.stats:
            stats = querier.get_stats()
            print("📊 Database Statistics:")
            for collection, count in stats.items():
                print(f"  {collection}: {count:,}")
            
        
        if args.provenance:
            prov_stats = querier.get_provenance_stats()
            print("🔍 Provenance Statistics:")
            print(f"Total statements: {prov_stats['total_statements']:,}")
            print(f"Statements with time offsets: {prov_stats['statements_with_offsets']:,}")
            print(f"Coverage: {prov_stats['coverage_percentage']:.1f}%")
            
            if prov_stats['sample_statements']:
                print("\n  Sample statements with provenance:")
                for i, stmt in enumerate(prov_stats['sample_statements'], 1):
                    print(f"    {i}. {stmt.get('subject', 'N/A')} -> {stmt.get('predicate', 'N/A')}")
                    print(f"       Time: {stmt.get('start_offset', 'N/A')}s - {stmt.get('end_offset', 'N/A')}s")
                    print(f"       Video: {stmt.get('source_video', 'N/A')}")
            print()
        
        # Perform query
        print(f"🚀 Querying: '{args.query}' with {args.hops} hops using {search_mode} search")
        subgraph = querier.query_graph(args.query, args.hops, search_mode)
        
        # Generate Turtle output
        turtle_output = querier.subgraph_to_turtle(subgraph)
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(turtle_output, encoding='utf-8')
            print(f"✅ Results saved to: {args.output}")
        else:
            print("\n" + "="*50)
            print("TURTLE OUTPUT:")
            print("="*50)
            print(turtle_output)
        
        # Summary
        statements_with_prov = sum(1 for stmt in subgraph['statements'] 
                                 if any(key in stmt for key in ["start_offset", "end_offset"]))
        
        print(f"\n📋 Query Summary:")
        print(f"  Query: '{args.query}'")
        print(f"  Search mode: {search_mode}")
        print(f"  Hops: {args.hops}")
        print(f"  Initial matches: {len(subgraph.get('search_results', []))}")
        print(f"  Final nodes: {len(subgraph['nodes'])}")
        print(f"  Edges: {len(subgraph['edges'])}")
        print(f"  Statements: {len(subgraph['statements'])}")
        print(f"  Statements with provenance: {statements_with_prov}")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Set MONGODB_CONNECTION_STRING environment variable")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()