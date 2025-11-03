#!/usr/bin/env python3
"""
Backfill node embeddings using Google GenAI (gemini-embedding-001)

This script finds documents in the `nodes` collection that are missing an
`embedding` field and have non-empty `searchable_text`, generates embeddings
via Google GenAI, and updates the documents in-place.

It supports a --dry-run mode that performs no writes or model calls.

Usage examples (PowerShell):
  $env:MONGODB_CONNECTION_STRING = "..."
  uv run .\scripts\05_backfill_embeddings.py --database parliamentary_graph --google-embeddings --embedding-dim 3072 --dry-run --limit 10

Requirements:
  - pymongo
  - python-dotenv
  - google genai Python client (when --google-embeddings and not --dry-run)

"""

import os
import sys
import json
import argparse
from typing import Optional, List
from datetime import datetime, timezone
import re
import time
import random

try:
    from pymongo import MongoClient, ASCENDING
    from pymongo.errors import ConnectionFailure
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages: pip install pymongo python-dotenv")
    sys.exit(1)

# Load environment variables from .env if present
load_dotenv()


def init_google_client() -> Optional[object]:
    try:
        from google import genai
        client = genai.Client()
        print("✅ Google GenAI client initialized")
        return client
    except Exception as e:
        print(f"⚠️  Failed to initialize Google GenAI client: {e}")
        return None


class EmbeddingBackfiller:
    def __init__(self, connection_string: Optional[str], database: str = "parliamentary_graph",
                 use_google: bool = False, embedding_dim: int = 3072, dry_run: bool = False,
                 batch_size: int = 32):
        if connection_string is None:
            connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("MONGODB_CONNECTION_STRING not set")

        try:
            self.client = MongoClient(connection_string)
            self.client.admin.command("ping")
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")

        self.db = self.client[database]
        self.nodes = self.db.nodes
        self.use_google = use_google
        self.embedding_dim = embedding_dim
        self.dry_run = dry_run
        self.batch_size = int(batch_size)

        self.google_client = None
        if self.use_google and not self.dry_run:
            self.google_client = init_google_client()
            if not self.google_client:
                print("Embeddings will be disabled because Google client failed to initialize")
                self.use_google = False

    def find_candidates(self, limit: Optional[int] = None) -> List[dict]:
        """Find nodes missing embedding but with searchable_text."""
        query = {"embedding": {"$exists": False}, "searchable_text": {"$exists": True, "$ne": ""}}
        cursor = self.nodes.find(query, {"_id": 1, "searchable_text": 1}).limit(limit or 0)
        return list(cursor)

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        if self.dry_run:
            return None
        if not self.use_google or not self.google_client or not text:
            return None

        # Clean text
        clean_text = re.sub(r"\s+", " ", text).strip()
        if len(clean_text) < 3:
            return None

        # Retry loop with exponential backoff to handle rate limits (429 / RESOURCE_EXHAUSTED)
        max_retries = 5
        base_delay = 1.0
        attempt = 0
        res = None
        while attempt <= max_retries:
            try:
                res = self.google_client.models.embed_content(model="gemini-embedding-001", contents=[clean_text])
                break
            except Exception as e:
                msg = str(e)
                # Detect rate limit / quota errors heuristically
                if '429' in msg or 'RESOURCE_EXHAUSTED' in msg or 'quota' in msg.lower():
                    sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    print(f"⚠️  Rate limit encountered (attempt {attempt+1}/{max_retries}). Sleeping {sleep_time:.1f}s before retry...")
                    time.sleep(sleep_time)
                    attempt += 1
                    continue
                else:
                    print(f"⚠️  Embedding generation failed: {e}")
                    return None
        if res is None:
            print("⚠️  Embedding generation failed after retries")
            return None

        # Normalize different possible return shapes from genai client
        emb_obj = getattr(res, "embeddings", None)
        if emb_obj is None and isinstance(res, dict) and "embeddings" in res:
            emb_obj = res["embeddings"]

        if emb_obj is None:
            print("⚠️  Embedding response has no 'embeddings' field")
            return None

        # emb_obj may be a list-like containing embedding entries
        if isinstance(emb_obj, (list, tuple)) and len(emb_obj) > 0:
            emb_entry = emb_obj[0]
        else:
            emb_entry = emb_obj

        # emb_entry might be a plain list of numbers, a dict or an object.
        emb_list = None

        # google.genai ContentEmbedding objects expose `values` (list of floats)
        try:
            values_attr = getattr(emb_entry, "values", None)
            if values_attr is not None and isinstance(values_attr, (list, tuple)):
                emb_list = list(values_attr)
        except Exception:
            pass

        # If it's a list/tuple of numbers
        if emb_list is None and isinstance(emb_entry, (list, tuple)) and all(not isinstance(x, (list, tuple, dict)) for x in emb_entry):
            emb_list = list(emb_entry)
        elif emb_list is None and isinstance(emb_entry, dict):
            # common keys to check
            for key in ("embedding", "value", "vector", "data"):
                if key in emb_entry:
                    cand = emb_entry[key]
                    if isinstance(cand, (list, tuple)):
                        emb_list = list(cand)
                        break
            # Sometimes embedding is nested under 'data' -> list of {'embedding': [...]}
            if emb_list is None and "data" in emb_entry and isinstance(emb_entry["data"], (list, tuple)) and len(emb_entry["data"])>0:
                inner = emb_entry["data"][0]
                if isinstance(inner, dict):
                    for key in ("embedding", "value", "vector"):
                        if key in inner and isinstance(inner[key], (list, tuple)):
                            emb_list = list(inner[key])
                            break
        else:
            # Try to access attribute 'embedding'
            try:
                cand = getattr(emb_entry, "embedding", None)
                if isinstance(cand, (list, tuple)):
                    emb_list = list(cand)
            except Exception:
                emb_list = None

        if emb_list is None:
            # As a last resort, try to coerce the entire emb_entry to a list
            try:
                emb_list = list(emb_entry)
            except Exception:
                print(f"⚠️  Could not parse embedding entry type={type(emb_entry)} repr={str(emb_entry)[:200]}")
                return None

        # emb_list may contain numeric values or tuples like (index, value). Normalize to floats.
        normalized = []
        for item in emb_list:
            if isinstance(item, (int, float)):
                normalized.append(float(item))
            elif isinstance(item, (list, tuple)):
                # If tuple looks like (index, value) or similar, try to find a numeric inside
                num = None
                for part in item:
                    if isinstance(part, (int, float)):
                        num = float(part)
                        # prefer the second element if tuple length == 2
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            num = float(item[1])
                            break
                if num is not None:
                    normalized.append(num)
                else:
                    # skip non-numeric
                    continue
            else:
                try:
                    normalized.append(float(item))
                except Exception:
                    # skip items we can't convert
                    continue

        # Validate length
        try:
            if self.embedding_dim and len(normalized) != int(self.embedding_dim):
                print(f"⚠️  Warning: received embedding length {len(normalized)} != expected {self.embedding_dim}")
        except Exception:
            pass

        if not normalized:
            # Debug output to help understand unexpected embedding shapes
            try:
                print(f"⚠️  Debug: embedding entry type={type(emb_entry)} repr={str(emb_entry)[:400]}")
                if emb_list is not None:
                    print(f"    emb_list sample (len={len(emb_list)}): {str(emb_list)[:400]}")
            except Exception:
                pass
            return None

        return normalized

    def backfill(self, limit: Optional[int] = None):
        candidates = self.find_candidates(limit=limit)
        total = len(candidates)
        print(f"Found {total} candidate nodes missing embeddings")
        if total == 0:
            return

        updated = 0
        processed = 0

        for i in range(0, total, self.batch_size):
            batch = candidates[i:i + self.batch_size]
            for doc in batch:
                processed += 1
                nid = doc["_id"]
                text = doc.get("searchable_text", "")
                emb = self.generate_embedding(text)
                if emb is None:
                    print(f"  - Skipping node {nid} (no embedding created)")
                    continue

                if self.dry_run:
                    print(f"  - DRY-RUN: would update node {nid} with embedding length {len(emb)}")
                    updated += 1
                else:
                    try:
                        self.nodes.update_one({"_id": nid}, {"$set": {"embedding": emb, "embedding_generated_at": datetime.now(timezone.utc)}})
                        print(f"  - Updated node {nid} with embedding (len={len(emb)})")
                        updated += 1
                    except Exception as e:
                        print(f"  ⚠️  Failed to update node {nid}: {e}")

        print(f"Backfill complete: processed={processed}, updated={updated}")


def main():
    parser = argparse.ArgumentParser(description="Backfill node embeddings using Google GenAI")
    parser.add_argument("--database", default="parliamentary_graph")
    parser.add_argument("--limit", type=int, help="Limit number of nodes to process")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--google-embeddings", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=3072)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    conn = os.getenv("MONGODB_CONNECTION_STRING")
    if not conn:
        print("MONGODB_CONNECTION_STRING not set in environment")
        sys.exit(1)

    try:
        backfiller = EmbeddingBackfiller(connection_string=conn, database=args.database,
                                         use_google=args.google_embeddings, embedding_dim=args.embedding_dim,
                                         dry_run=args.dry_run, batch_size=args.batch_size)
        backfiller.backfill(limit=args.limit)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
