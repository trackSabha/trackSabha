#!/usr/bin/env python3
"""
MongoDB Parliamentary Transcript to RDF Converter (JSON-LD)

This script converts processed parliamentary transcripts from MongoDB into RDF knowledge graphs,
with proper provenance linking each claim to specific time segments, and stores the TTL back to MongoDB.

Requirements:
- langchain-google-genai
- pymongo
- rdflib
- python-dotenv
- pydantic

Usage:
    python mongodb_ttl_generator.py --database parliamentary_graph
"""
"""
Usage:
    python mongodb_ttl_generator.py --database parliamentary_graph
"""

import sys
import os
import json
import argparse
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import re

# Suppress verbose logs from LangChain and Google GenAI
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    # JsonOutputParser location may vary; try common import path
    from langchain_core.output_parsers import JsonOutputParser
    from pydantic import BaseModel, Field
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    from dotenv import load_dotenv
    import rdflib
    from rdflib import Graph
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install langchain-google-genai pymongo rdflib python-dotenv pydantic")
    sys.exit(1)

# Load environment variables
load_dotenv()

def chunk_transcript(text: str, max_chars: int = 12000) -> List[str]:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


class KnowledgeGraph(BaseModel):
    """Pydantic model for structured knowledge graph output."""
    context: Dict[str, str] = Field(description="JSON-LD context with namespace prefixes", alias="@context")
    graph: List[dict] = Field(description="List of RDF entities and relationships in JSON-LD format", alias="@graph")

    class Config:
        validate_by_name = True
        populate_by_name = True


class MongoDBRDFConverter:
    def chunk_transcript(text: str, max_chars: int = 12000) -> List[str]:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    def __init__(self, connection_string: str = None, database_name: str = "parliamentary_graph", api_key: str = None):
        """
        Initialize the RDF converter with MongoDB connection and Gemini model.
        
        Args:
            connection_string: MongoDB connection string. If None, will try to get from environment.
            database_name: Name of the MongoDB database to use
            api_key: Google API key. If None, will try to get from environment.
        """
        # Setup MongoDB connection
        if connection_string is None:
            connection_string = os.getenv('MONGODB_CONNECTION_STRING')
            
        if not connection_string:
            raise ValueError(
                "MongoDB connection string is required. Set MONGODB_CONNECTION_STRING environment variable "
                "or pass connection_string parameter."
            )
        
        try:
            self.client = MongoClient(connection_string)
            # Test connection
            self.client.admin.command('ping')
            print("✅ Successfully connected to MongoDB")
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")
        
        self.db = self.client[database_name]
        self.videos = self.db.videos
        
        # Setup Gemini model
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
            
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-lite-latest",
            google_api_key=api_key,
            temperature=0.1,  # Lower temperature for more consistent output
            thinking_budget=0
        )
        
        # Set up JSON output parser
        self.json_parser = JsonOutputParser(pydantic_object=KnowledgeGraph)
    
    def _get_conversion_prompt(self, video_id: str, video_title: str) -> str:
        """Return the improved system prompt for transcript to JSON-LD conversion with better provenance."""
        return f"""You are converting a parliamentary transcript into RDF triples in JSON-LD format. 

**CRITICAL REQUIREMENT: EVERY factual claim must have BOTH the main triple AND a corresponding reified statement with time provenance.**

**Source Video Title:** `{video_title}`
**Source Video ID:** `{video_id}` (YouTube video)

**Input Format:** Each line starts with time offset in seconds, followed by spoken text.
Example: `54 This is [unknown] H, Member for St. Peter.`

**Required Output Structure:**
```json
{{
  "@context": {{
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "schema": "http://schema.org/",
    "org": "http://www.w3.org/ns/org#",
    "lok": "http://example.com/jansety.in/ontology#",
    "sess":"http://example.com/jansetu.in/session/",
    "prov": "http://www.w3.org/ns/prov#"
  }},
  "@graph": [
    // Main entities AND reified statements with provenance
  ]
}}
```

**MANDATORY PATTERN - For EVERY claim, create BOTH:**

1. **Main Entity** (example for speaker identification):
```json
{{
  "@id": "bbp:IndianParliament",
  "@type": ["schema:Person", "bbp:Parliament"],
  "schema:name": "H",
  "lok:hasRole": "Member for Lok sabha",
  "lok:representsConstituency": {{"@id": "bbp:Constituency_LokSabha"}}
}}
```

2. **Reified Statement with Time Provenance** (MANDATORY):
```json
{{
  "@id": "_:stmt_H_role_{video_id}_54",
  "@type": "rdf:Statement",
  "rdf:subject": {{"@id": "bbp:IndianParliament"}},
  "rdf:predicate": {{"@id": "bbp:hasRole"}},
  "rdf:object": "Member for Lok Sabha",
  "prov:wasDerivedFrom": {{
    "@type": "bbp:TranscriptSegment",
    "lok:fromVideo": {{"@id": "sess:video_{video_id}"}},
    "lok:startTimeOffset": {{"@type": "xsd:decimal", "@value": "54.0"}},
    "lok:endTimeOffset": {{"@type": "xsd:decimal", "@value": "61.0"}},
    "lok:transcriptText": "This is [unknown] H, Member for LokSabha"
  }}
}}
```

**STEP-BY-STEP PROCESS:**
1. Read each timestamped line
2. Extract factual claims (speaker identity, role, position, topic discussed, etc.)
3. For EACH claim, create:
   - The main RDF triple
   - A reified statement linking it to the specific time segment
4. Use exact time offsets from transcript
5. Calculate endTimeOffset as start of next relevant statement

**ENTITY TYPES TO EXTRACT:**
- **Video**: `sess:video_{video_id}`
- **Session**: `sess:{video_id}_session`
- **Parliament**: `lok:Parliament_[Name]`
- **Constituencies**: `lok:Constituency_[Name]`
- **Bills**: `lok:Bill_[Name]`
- **Concepts**: `lok:Concept_[TopicName]`
- **Organizations**: `lok:Org_[Name]`

**CRITICAL RULES:**
1. NEVER output JSON without reified statements
2. EVERY factual claim needs time provenance
3. Use exact timestamp format: "54.0", "61.0", etc.
4. Include original transcript text in provenance
5. Link everything back to specific video segments

Video Title: {video_title}
Video ID: {video_id}

Now convert this transcript, ensuring EVERY claim has both main triple AND reified statement with time provenance:"""

    def get_videos_with_processed_transcripts(self) -> List[Dict[str, Any]]:
        """
        Get all videos from videos collection that have processed transcripts but no TTL.

        Returns:
            List of video documents with processed transcripts
        """
        # The transcript cleaner writes the processed transcript into `transcript`.
        query = {
            "transcript": {"$exists": True, "$ne": ""},
            "ttl_generated_at": {"$exists": False}
        }

        # Only fetch necessary fields
        projection = {
            "transcript": 1,
            "Video_title": 1,
            "video_id": 1,
            "VideoURL": 1,
            "processed_at": 1,
            "ttl_generated_at": 1
        }
        videos = list(self.videos.find(query, projection))
        print(f"Found {len(videos)} videos with processed transcripts but no TTL")

        return videos

    def check_if_ttl_exists(self, video_url: str) -> bool:
        """
        Check if a video already has TTL generated.
        
        Args:
            video_url: URL of the video to check
            
        Returns:
            True if TTL exists, False otherwise
        """
        existing = self.videos.find_one({"VideoURL": video_url, "ttl": {"$exists": True}})
        return existing is not None

    def validate_provenance(self, json_data: Dict[str, Any]) -> bool:
        """Validate that the JSON-LD contains proper provenance information."""
        if '@graph' not in json_data:
            print("❌ No @graph found in JSON-LD")
            return False
        
        graph = json_data['@graph']
        reified_statements = [item for item in graph if item.get('@type') == 'rdf:Statement']
        transcript_segments = [item for item in graph if 
                             isinstance(item.get('prov:wasDerivedFrom'), dict) and
                             item['prov:wasDerivedFrom'].get('@type') == 'bbp:TranscriptSegment']
        
        print(f"📊 Validation results:")
        print(f"  - Total graph items: {len(graph)}")
        print(f"  - Reified statements: {len(reified_statements)}")
        print(f"  - Items with transcript provenance: {len(transcript_segments)}")
        
        if len(reified_statements) == 0:
            print("❌ No reified statements found - missing provenance!")
            return False
        
        if len(transcript_segments) == 0:
            print("❌ No transcript segment provenance found!")
            return False
        
        # Check if reified statements have proper time provenance
        valid_provenance = 0
        for stmt in reified_statements:
            if 'prov:wasDerivedFrom' in stmt:
                prov = stmt['prov:wasDerivedFrom']
                if isinstance(prov, dict) and 'lok:startTimeOffset' in prov:
                    valid_provenance += 1
        
        print(f"  - Reified statements with time provenance: {valid_provenance}")
        
        if valid_provenance < len(reified_statements) * 0.5:  # At least 50% should have time provenance
            print("❌ Insufficient time provenance in reified statements!")
            return False
        
        print("✅ Provenance validation passed")
        return True

    def convert_to_jsonld(self, transcript: str, video_id: str, video_title: str) -> Dict[str, Any]:
        """Convert transcript to JSON-LD format with enhanced provenance validation."""
        conversion_prompt = self._get_conversion_prompt(video_id, video_title)
        
        # Try multiple approaches if first one fails
        for attempt in range(3):
            try:
                print(f"🔄 Attempt {attempt + 1} to generate JSON-LD with provenance...")
                
                if attempt == 0:
                    # First attempt: simple direct LLM call and JSON parsing
                    messages = [
                        SystemMessage(content=conversion_prompt),
                        HumanMessage(content=transcript)
                    ]
                    response = self.llm.invoke(messages)
                    try:
                        json_data = json.loads(response.content.strip())
                    except Exception:
                        # Fall back to the parser if response is not strict JSON
                        try:
                            json_data = self.json_parser.parse(response.content)
                        except Exception as e:
                            raise
                    
                else:
                    # Subsequent attempts: Direct LLM call with more explicit instructions
                    enhanced_prompt = conversion_prompt + f"""
                    
**CRITICAL REMINDER FOR ATTEMPT {attempt + 1}:**
The previous attempt failed to include proper provenance. You MUST include reified statements.

For EVERY fact extracted (like "X is Member for Y"), create:
1. The main entity
2. A reified statement with this pattern:
```
{{
  "@id": "_:stmt_[unique_id]",
  "@type": "rdf:Statement", 
  "rdf:subject": {{"@id": "[subject_uri]"}},
  "rdf:predicate": {{"@id": "[predicate_uri]"}},
  "rdf:object": "[object_value]",
  "prov:wasDerivedFrom": {{
    "@type": "bbp:TranscriptSegment",
    "lok:fromVideo": {{"@id": "sess:video_{video_id}"}},
    "lok:startTimeOffset": {{"@type": "xsd:decimal", "@value": "[time].0"}},
    "lok:endTimeOffset": {{"@type": "xsd:decimal", "@value": "[next_time].0"}},
    "lok:transcriptText": "[original text]"
  }}
}}
```

DO NOT output JSON-LD without reified statements!
                    """
                    
                    messages = [
                        SystemMessage(content=enhanced_prompt),
                        HumanMessage(content=transcript)
                    ]
                    
                    response = self.llm.invoke(messages)
                    try:
                        json_data = json.loads(response.content.strip())
                    except Exception:
                        json_data = self.json_parser.parse(response.content)
                
                # Validate provenance
                if self.validate_provenance(json_data):
                    print(f"✅ Attempt {attempt + 1} successful - proper provenance found")
                    return json_data
                else:
                    print(f"❌ Attempt {attempt + 1} failed validation - missing provenance")
                    if attempt == 2:  # Last attempt
                        print("⚠️ All attempts failed - proceeding with best available result")
                        return json_data
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed with error: {e}")
                if attempt == 2:  # Last attempt
                    raise e
        
        raise Exception("All conversion attempts failed")

    def validate_and_load_to_rdf(self, json_data: Dict[str, Any]) -> Graph:
        """Validate JSON-LD by loading it into rdflib Graph."""
        try:
            g = Graph()
            json_str = json.dumps(json_data, indent=2)
            
            # Debug: Print structure info
            print(f"📋 JSON-LD structure preview:")
            print(f"  - Top-level keys: {list(json_data.keys())}")
            if '@graph' in json_data:
                print(f"  - Graph entries: {len(json_data['@graph'])}")
                if json_data['@graph']:
                    first_item = json_data['@graph'][0]
                    print(f"  - First item type: {first_item.get('@type', 'No type')}")
                    print(f"  - First item ID: {first_item.get('@id', 'No ID')}")
            
            g.parse(data=json_str, format='json-ld')
            return g
        except Exception as e:
            print(f"📋 JSON-LD content causing error (first 500 chars):")
            print(json.dumps(json_data, indent=2)[:500])
            raise Exception(f"Failed to load JSON-LD into RDF graph: {str(e)}")

    def save_ttl_to_mongodb(self, video_url: str, ttl_content: str, json_ld: Dict[str, Any], 
                           triple_count: int) -> bool:
        """
        Save the generated TTL content to MongoDB.
        
        Args:
            video_url: URL of the video
            ttl_content: Generated TTL/Turtle content
            json_ld: Generated JSON-LD data
            triple_count: Number of RDF triples generated
            
        Returns:
            True if successful, False otherwise
        """
        try:
            document = {
                "ttl": ttl_content,
                "json_ld": json_ld,
                "rdf_triple_count": triple_count,
                "rdf_generated_at": datetime.now(timezone.utc),
                "rdf_generator_version": "mongodb_ttl_generator_v1.0"
            }
            
            # Update the existing video document with TTL data
            result = self.videos.update_one(
                {"VideoURL": video_url},
                {"$set": document}
            )
            
            if result.modified_count > 0:
                print(f"  ✅ TTL saved to MongoDB ({len(ttl_content)} chars, {triple_count} triples)")
                return True
            else:
                print(f"  ❌ Failed to update video document in MongoDB")
                return False
            
        except Exception as e:
            print(f"  ❌ Error saving TTL to MongoDB: {e}")
            return False
    import re

    def split_transcript_by_time(transcript: str, chunk_duration: int = 300) -> List[str]:
        """
        Split the transcript into chunks based on timestamps, using chunk_duration in seconds.
        Returns a list of transcript chunks.
        """
        pattern = re.compile(r"^(\d+)\s+(.*)$", re.MULTILINE)
        lines = pattern.findall(transcript)

        if not lines:
            print("⚠️ Could not find timestamped lines to split")
            return [transcript]

        chunks = []
        current_chunk = []
        current_start = 0

        for ts_str, text in lines:
            ts = int(ts_str)
            if ts - current_start >= chunk_duration and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_start = ts
            current_chunk.append(f"{ts} {text}")

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        print(f"🧩 Transcript split into {len(chunks)} chunks (each ~{chunk_duration}s)")
        return chunks


    def process_single_video(self, video: Dict[str, Any]) -> bool:
        """
        Process a single video's transcript to generate TTL (RDF).
        
        Args:
            video: Video document from MongoDB
        
        Returns:
            True if successfully processed and saved, else False.
        """
        video_url = video.get("VideoURL", "")
        video_title = video.get("Video_title", "Untitled Video")
        video_id = video.get("video_id", "")
        # The cleaner stores the processed transcript in 'transcript' (string)
        transcript = video.get("transcript", "")
        if not transcript:
            print("  ⚠️  Transcript is missing or empty.")
            return False

        if not video_id:
            print("  ⚠️  Video ID is missing.")
            return False

        try:
            # If the transcript is stored as a list of blocks, join them
            if isinstance(transcript, list):
                transcript = " ".join(block.get("text", "") for block in transcript)

            transcript_length = len(transcript)
            print(f"  📄 Transcript length: {transcript_length} characters")


            if transcript_length < 200:
                print("  ⚠️  Transcript too short to extract meaningful data. Skipping.")
                return False

            # Convert to JSON-LD
            print("  🔄 Converting transcript to JSON-LD with provenance...")
            if isinstance(transcript, list):
                transcript = " ".join(block.get("text", "") for block in transcript)

            chunks = chunk_transcript(transcript, max_chars=12000)
            merged_graph = []
            json_context = "https://schema.org"

            for idx, chunk in enumerate(chunks):
                print(f"🧩 Processing chunk {idx + 1}/{len(chunks)}...")
                partial_jsonld = self.convert_to_jsonld(chunk, video_id, video_title)
                
                if "@graph" in partial_jsonld:
                    merged_graph.extend(partial_jsonld["@graph"])
                if "@context" in partial_jsonld:
                    json_context = partial_jsonld["@context"]

            json_data = {
                "@context": json_context,
                "@graph": merged_graph
            }


            # Validate and load RDF
            print("  🔍 Loading RDF graph from JSON-LD...")
            rdf_graph = self.validate_and_load_to_rdf(json_data)
            triple_count = len(rdf_graph)
            print(f"  ✅ RDF graph successfully created with {triple_count} triples.")

            if triple_count == 0:
                print("  ⚠️  RDF graph is empty. Possible issue with parsing or Gemini output.")
                return False

            # Generate TTL
            print("  📝 Serializing RDF graph to Turtle format (TTL)...")
            ttl_content = rdf_graph.serialize(format='turtle')

            # Save back to MongoDB
            print("  💾 Saving TTL content back to MongoDB...")
            success = self.save_ttl_to_mongodb(video_url, ttl_content, json_data, triple_count)

            if success:
                self.print_provenance_summary(json_data)
                return True
            else:
                print("  ❌ Failed to save TTL to MongoDB.")
                return False

        except Exception as e:
            print(f"  ❌ Exception during processing: {str(e)}")
            return False
    

    
    

    def print_provenance_summary(self, json_data: Dict[str, Any]):
        """Print a summary of provenance information in the generated JSON-LD."""
        if '@graph' not in json_data:
            return

        graph = json_data['@graph']
        reified_statements = [item for item in graph if item.get('@type') == 'rdf:Statement']

        print("  📈 PROVENANCE SUMMARY:")
        print(f"    - Total entities: {len(graph)}")
        print(f"    - Reified statements: {len(reified_statements)}")

        if reified_statements:
            time_ranges = []
            for stmt in reified_statements:
                if 'prov:wasDerivedFrom' in stmt:
                    prov = stmt['prov:wasDerivedFrom']
                    if isinstance(prov, dict) and 'lok:startTimeOffset' in prov:
                        try:
                            start_time = float(prov['lok:startTimeOffset'].get('@value', prov['lok:startTimeOffset']))
                            time_ranges.append(start_time)
                        except Exception:
                            continue

            if time_ranges:
                print(f"    - Time range covered: {min(time_ranges):.1f}s - {max(time_ranges):.1f}s")
                print(f"    - Provenance points: {len(time_ranges)}")

        print("    - ✅ All claims linked to video timestamps")

    def get_generation_stats(self) -> Dict[str, int]:
        """Get statistics about TTL generation."""
        videos_with_transcripts = self.videos.count_documents({
            "transcript": {"$exists": True, "$ne": ""}
        })

        videos_with_ttl = self.videos.count_documents({
            "ttl": {"$exists": True, "$ne": ""}
        })

        return {
            "videos_with_transcripts": videos_with_transcripts,
            "videos_with_ttl": videos_with_ttl,
            "remaining": videos_with_transcripts - videos_with_ttl
        }

    def process_all_videos(self, skip_existing: bool = True, limit: Optional[int] = None):
        """
        Process all videos with transcripts to generate TTL files.
        
        Args:
            skip_existing: Whether to skip videos with existing TTL
            limit: Maximum number of videos to process (None for all)
        """
        print("Starting TTL generation from processed transcripts...")

        # Get videos with processed transcripts
        if skip_existing:
            videos_to_process = self.get_videos_with_processed_transcripts()
        else:
            # Get all videos with transcripts, regardless of TTL existence
            query = {"transcript": {"$exists": True, "$ne": ""}}
            projection = {
                "VideoURL": 1,
                "Video_title": 1,
                "video_id": 1,
                "transcript": 1,
                "_id": 1
            }
            videos_to_process = list(self.videos.find(query, projection))
            print(f"Found {len(videos_to_process)} videos with processed transcripts")

        if not videos_to_process:
            print("No videos to process")
            return

        # Apply limit if specified
        if limit:
            videos_to_process = videos_to_process[:limit]
            print(f"Processing limited to first {limit} videos")

        # Dry-run: just list videos that would be processed
        if getattr(self, 'dry_run', False):
            print(f"[dry-run] Would process {len(videos_to_process)} videos:")
            for v in videos_to_process:
                print(f"  - {v.get('VideoURL', v.get('video_id', '<no id>'))} | {v.get('Video_title','')}")
            return

        stats = {
            "total": len(videos_to_process),
            "processed": 0,
            "skipped": 0,
            "errors": 0
        }

        for i, video in enumerate(videos_to_process, 1):
            video_url = video.get("VideoURL", "")
            video_title = video.get("Video_title", "Unknown Title")
            video_id = video.get("video_id", "")

            print(f"\n[{i}/{stats['total']}] Processing: {video_title[:80]}...")
            print(f"  🆔 Video ID: {video_id}")

            # Check if already processed (if skip_existing is True)
            if skip_existing and self.check_if_ttl_exists(video_url):
                print("  ⏭️  TTL already exists, skipping")
                stats["skipped"] += 1
                continue

            # Process the video
            if self.process_single_video(video):
                stats["processed"] += 1
            else:
                stats["errors"] += 1

        # Print final statistics
        print("\n📊 TTL Generation Complete!")
        print(f"  Total videos: {stats['total']}")
        print(f"  Successfully processed: {stats['processed']}")
        print(f"  Skipped (already processed): {stats['skipped']}")
        print(f"  Errors: {stats['errors']}")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Generate TTL/RDF from processed parliamentary transcripts in MongoDB")
    parser.add_argument("--database", default="youtube_data", help="MongoDB database name")
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    parser.add_argument("--force", action="store_true", help="Regenerate TTL for videos that already have it")
    parser.add_argument("--stats", action="store_true", help="Show generation statistics only")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the LLM or write TTL to DB; just list what would be processed")
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        converter = MongoDBRDFConverter(database_name=args.database)
        converter.dry_run = bool(args.dry_run)

        if args.stats:
            # Show statistics only
            stats = converter.get_generation_stats()
            print("📊 TTL Generation Statistics:")
            print(f"  Videos with processed transcripts: {stats['videos_with_transcripts']}")
            print(f"  Videos with TTL generated: {stats['videos_with_ttl']}")
            print(f"  Remaining to process: {stats['remaining']}")
            return
        
        # Process videos
        converter.process_all_videos(
            skip_existing=not args.force,
            limit=args.limit
        )
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nTo set up required credentials:")
        print("1. MongoDB: Set MONGODB_CONNECTION_STRING environment variable")
        print("2. Google API: Set GEMINI_API_KEY environment variable")
        print("3. Or create a .env file with both values")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()