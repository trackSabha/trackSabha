#!/usr/bin/env python3
"""
YouTube Data to MongoDB Uploader

This script loads YouTube video data from JSON files and saves them to MongoDB
for search and analysis, with support for full-text search and efficient querying.

Requirements:
- pymongo
- python-dotenv (optional, for environment variables)
- dateutil

Usage:
    python youtube_uploader.py <json_file.json>
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from urllib.parse import urlparse
import re

try:
    from pymongo import MongoClient, ASCENDING, TEXT, DESCENDING
    from pymongo.errors import ConnectionFailure, DuplicateKeyError, BulkWriteError
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install pymongo python-dotenv python-dateutil")
    sys.exit(1)

try:
    from dateutil.parser import parse as parse_date
except ImportError:
    print("Warning: python-dateutil not available. Date parsing may be limited.")
    parse_date = None

# Load environment variables
load_dotenv()

class YouTubeToMongoUploader:
    def __init__(self, connection_string: str = None, database_name: str = "youtube_data"):
        """
        Initialize the YouTube to MongoDB uploader.
        
        Args:
            connection_string: MongoDB connection string. If None, will try to get from environment.
            database_name: Name of the MongoDB database to use
        """
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
        self.setup_collections()
    
    def setup_collections(self):
        """Set up MongoDB collections with appropriate indexes."""
        
        # Raw videos collection
        self.videos = self.db.raw_videos
        
        # Create indexes for efficient querying
        indexes_to_create = [
            # Index on Channel ID for channel-based queries
            {"keys": [("Channel_Id", ASCENDING)], "name": "channel_id_idx"},
            
            # Index on published date for time-based queries
            {"keys": [("published_Date", DESCENDING)], "name": "published_date_idx"},
            
            # Index on views (converted to numeric) for sorting
            {"keys": [("views_numeric", DESCENDING)], "name": "views_numeric_idx"},
            
            # Text index for full-text search on title and description
            {"keys": [("Video_title", TEXT), ("Description", TEXT), ("Channel_Name", TEXT)], "name": "text_search_idx"},
            
            # Index on category for filtering
            {"keys": [("category", ASCENDING)], "name": "category_idx"},
            
            # Index on hasTranscript for filtering videos with transcripts
            {"keys": [("hasTranscript", ASCENDING)], "name": "transcript_idx"},
            
            # Compound index for common query patterns
            {"keys": [("Channel_Id", ASCENDING), ("published_Date", DESCENDING)], "name": "channel_date_idx"},
        ]
        
        # Try to create unique index on VideoURL, but handle existing null values
        try:
            # First, let's create a partial unique index that only applies to non-null VideoURL values
            self.videos.create_index(
                [("VideoURL", ASCENDING)], 
                unique=True, 
                name="video_url_unique",
                partialFilterExpression={"VideoURL": {"$ne": None}}
            )
        except Exception as e:
            print(f"Warning: Could not create index video_url_unique: {e}")
        
        for index_spec in indexes_to_create:
            try:
                self.videos.create_index(
                    index_spec["keys"], 
                    unique=index_spec.get("unique", False),
                    name=index_spec["name"]
                )
            except Exception as e:
                # Index might already exist
                if "already exists" not in str(e):
                    print(f"Warning: Could not create index {index_spec['name']}: {e}")
        
        print("✅ Collections and indexes set up successfully")
    
    def extract_numeric_views(self, views_str: str) -> Optional[int]:
        """Extract numeric value from views string like '1348 views'."""
        if not views_str:
            return None
        
        # Remove 'views' and any commas, then extract number
        numeric_part = re.sub(r'[^\d]', '', views_str)
        try:
            return int(numeric_part) if numeric_part else None
        except ValueError:
            return None
    
    def parse_duration(self, duration_str: str) -> Optional[int]:
        """Parse duration string like '200:24' to total seconds."""
        if not duration_str:
            return None
        
        try:
            parts = duration_str.split(':')
            if len(parts) == 2:
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            else:
                return None
        except (ValueError, AttributeError):
            return None
    
    def parse_published_date(self, date_str: str) -> Optional[datetime]:
        """Parse published date string to datetime object."""
        if not date_str:
            return None
        
        if parse_date:
            try:
                return parse_date(date_str)
            except Exception:
                pass
        
        # Fallback parsing for ISO format
        try:
            # Handle timezone format like '2024-10-08T17:05:23-07:00'
            if 'T' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except Exception:
            pass
        
        return None
    
    def process_transcript(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process transcript data and extract useful information."""
        if not transcript_data:
            return {}
        
        processed = {
            "has_transcript": transcript_data.get("hasTranscript", False),
            "format": transcript_data.get("format"),
            "language": transcript_data.get("language"),
            "is_translated": transcript_data.get("isTranslated", False),
            "is_auto_generated": transcript_data.get("isAutoGenerated", False),
            "segment_count": transcript_data.get("segmentCount", 0)
        }
        
        # Extract and process transcript content
        formatted_content = transcript_data.get("formattedContent", "")
        if formatted_content:
            # Extract text from XML-like transcript format
            text_content = re.sub(r'<[^>]+>', '', formatted_content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            processed["transcript_text"] = text_content
            processed["transcript_length"] = len(text_content)
            processed["word_count"] = len(text_content.split()) if text_content else 0
        
        return processed
    
    def clean_and_enhance_video_data(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and enhance video data with additional processed fields."""
        cleaned = video_data.copy()
        
        # Ensure VideoURL is not None or empty - this is critical for the unique index
        video_url = video_data.get("VideoURL")
        if not video_url or video_url.strip() == "":
            # Generate a unique identifier for videos without URLs
            video_id = video_data.get("video_id") or f"unknown_{datetime.now().isoformat()}"
            cleaned["VideoURL"] = f"unknown://video/{video_id}"
            print(f"Warning: Video has no URL, assigning: {cleaned['VideoURL']}")
        
        # Add processed numeric fields
        cleaned["views_numeric"] = self.extract_numeric_views(video_data.get("Views"))
        cleaned["duration_seconds"] = self.parse_duration(video_data.get("Runtime"))
        cleaned["published_datetime"] = self.parse_published_date(video_data.get("published_Date"))
        
        # Process transcript data
        transcript_info = self.process_transcript(video_data.get("transcript", {}))
        cleaned["transcript_info"] = transcript_info
        
        # Extract video ID from URL
        video_url = cleaned.get("VideoURL", "")
        if "watch?v=" in video_url:
            video_id = video_url.split("watch?v=")[-1].split("&")[0]
            cleaned["video_id"] = video_id
        
        # Add processing metadata
        cleaned["processed_at"] = datetime.now(timezone.utc)
        cleaned["data_source"] = "youtube_scraper"
        
        # Create searchable text field combining title, description, and channel
        searchable_parts = []
        for field in ["Video_title", "Description", "Channel_Name"]:
            if video_data.get(field):
                searchable_parts.append(str(video_data[field]))
        
        if transcript_info.get("transcript_text"):
            # Add first 1000 characters of transcript for search
            searchable_parts.append(transcript_info["transcript_text"][:1000])
        
        cleaned["searchable_text"] = " ".join(searchable_parts)
        
        return cleaned
    
    def load_json_file(self, json_file: str) -> List[Dict[str, Any]]:
        """Load and parse JSON file containing YouTube data."""
        print(f"Loading JSON file: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both single video object and list of videos
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise ValueError("JSON file must contain a video object or list of videos")
            
            print(f"✅ Loaded {len(data)} video records from JSON file")
            return data
        except Exception as e:
            raise Exception(f"Error loading JSON file: {e}")
    
    def upload_videos(self, videos_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Upload video data to MongoDB with bulk operations for efficiency."""
        print(f"Processing {len(videos_data)} videos...")
        
        processed_videos = []
        stats = {"processed": 0, "inserted": 0, "updated": 0, "errors": 0}
        
        # Process each video
        for video_data in videos_data:
            try:
                cleaned_video = self.clean_and_enhance_video_data(video_data)
                
                # Validate that required fields are present and not None
                if not cleaned_video.get("VideoURL"):
                    print(f"Warning: Skipping video with no VideoURL: {video_data}")
                    stats["errors"] += 1
                    continue
                
                processed_videos.append(cleaned_video)
                stats["processed"] += 1
            except Exception as e:
                print(f"Warning: Error processing video {video_data.get('VideoURL', 'unknown')}: {e}")
                stats["errors"] += 1
        
        if not processed_videos:
            print("No videos to upload after processing")
            return stats
        
        # Process videos one by one to handle errors gracefully
        print(f"Uploading {len(processed_videos)} processed videos...")
        
        for video in processed_videos:
            try:
                # Use update_one with upsert instead of bulk operations for better error handling
                result = self.videos.update_one(
                    {"VideoURL": video["VideoURL"]},
                    {"$set": video},
                    upsert=True
                )
                
                if result.upserted_id:
                    stats["inserted"] += 1
                elif result.modified_count > 0:
                    stats["updated"] += 1
                    
            except Exception as e:
                print(f"Error: {e}")
                stats["errors"] += 1
                continue
        
        print(f"✅ Upload complete:")
        print(f"   Processed: {stats['processed']} videos")
        print(f"   Inserted: {stats['inserted']} new videos")
        print(f"   Updated: {stats['updated']} existing videos")
        if stats["errors"] > 0:
            print(f"   Errors: {stats['errors']} videos failed processing")
        
        return stats
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the raw_videos collection."""
        total_videos = self.videos.count_documents({})
        videos_with_transcripts = self.videos.count_documents({"hasTranscript": True})
        
        # Get channel distribution
        pipeline = [
            {"$group": {"_id": "$Channel_Name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        top_channels = list(self.videos.aggregate(pipeline))
        
        # Get date range
        date_range = list(self.videos.aggregate([
            {"$match": {"published_datetime": {"$exists": True}}},
            {"$group": {
                "_id": None,
                "earliest": {"$min": "$published_datetime"},
                "latest": {"$max": "$published_datetime"}
            }}
        ]))
        
        stats = {
            "total_videos": total_videos,
            "videos_with_transcripts": videos_with_transcripts,
            "transcript_percentage": round((videos_with_transcripts / total_videos * 100), 2) if total_videos > 0 else 0,
            "top_channels": top_channels,
            "date_range": date_range[0] if date_range else None
        }
        
        return stats

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Upload YouTube video data to MongoDB")
    parser.add_argument("json_file", help="Path to the JSON file containing YouTube video data")
    parser.add_argument("--database", default="youtube_data", help="MongoDB database name (default: youtube_data)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.json_file).exists():
        print(f"Error: Input file '{args.json_file}' does not exist.")
        sys.exit(1)
    
    try:
        # Initialize uploader
        uploader = YouTubeToMongoUploader(database_name=args.database)
        
        # Load and upload data
        videos_data = uploader.load_json_file(args.json_file)
        upload_stats = uploader.upload_videos(videos_data)
        
        # Show collection statistics
        collection_stats = uploader.get_collection_stats()
        print(f"\n📊 Collection Statistics:")
        print(f"  Total videos in collection: {collection_stats['total_videos']:,}")
        print(f"  Videos with transcripts: {collection_stats['videos_with_transcripts']:,} ({collection_stats['transcript_percentage']}%)")
        
        if collection_stats["top_channels"]:
            print(f"  Top channels by video count:")
            for channel in collection_stats["top_channels"][:5]:
                print(f"    {channel['_id']}: {channel['count']} videos")
        
        if collection_stats["date_range"]:
            date_range = collection_stats["date_range"]
            if date_range["earliest"] and date_range["latest"]:
                print(f"  Date range: {date_range['earliest'].strftime('%Y-%m-%d')} to {date_range['latest'].strftime('%Y-%m-%d')}")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nTo set up MongoDB connection:")
        print("1. Create a MongoDB Atlas cluster or local MongoDB instance")
        print("2. Get your connection string")
        print("3. Set environment variable: export MONGODB_CONNECTION_STRING='your-connection-string'")
        print("4. Or create a .env file with: MONGODB_CONNECTION_STRING=your-connection-string")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()