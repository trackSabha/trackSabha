"""
YouTube Transcript Scraper

This script uses the Apify platform to fetch YouTube video transcripts
using the pintostudio/youtube-transcript-scraper actor.

Requirements:
- apify-client
- python-dotenv (optional, for environment variables)

Usage:
    python youtube_transcript_scraper.py <video_id>
"""

import sys
import os
import json
from pathlib import Path
import argparse
import re
import requests

try:
    from apify_client import ApifyClient
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install apify-client python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

class YouTubeTranscriptScraper:
    def __init__(self, api_token: str = None):
        """
        Initialize the YouTube transcript scraper.
        
        Args:
            api_token: Apify API token. If None, will try to get from environment.
        """
        if api_token is None:
            api_token = os.getenv('APIFY_API_TOKEN')
            
        if not api_token:
            raise ValueError(
                "Apify API token is required. Set APIFY_API_TOKEN environment variable "
                "or pass api_token parameter."
            )
        
        self.client = ApifyClient(api_token)
        self.actor_id = "pintostudio/youtube-transcript-scraper"
        # number of retries for transient failures
        self.max_retries = int(os.getenv('YT_TRANSCRIPT_MAX_RETRIES', '3'))
        # backoff factor in seconds (will multiply by attempt number)
        self.backoff_factor = float(os.getenv('YT_TRANSCRIPT_BACKOFF', '1.0'))
    
    def extract_video_id(self, video_input: str) -> str:
        """
        Extract video ID from YouTube URL or return if already a video ID.
        
        Args:
            video_input: YouTube URL or video ID
            
        Returns:
            Clean video ID
        """
        # If it's already a video ID (11 characters), return as is
        if len(video_input) == 11 and not ('/' in video_input or '.' in video_input):
            return video_input
        
        # Extract from various YouTube URL formats
        if 'youtu.be/' in video_input:
            # Short URL format: https://youtu.be/VIDEO_ID
            video_id = video_input.split('youtu.be/')[1].split('?')[0].split('&')[0]
        elif 'youtube.com/watch' in video_input:
            # Standard URL format: https://www.youtube.com/watch?v=VIDEO_ID
            if 'v=' in video_input:
                video_id = video_input.split('v=')[1].split('&')[0]
            else:
                raise ValueError("Could not extract video ID from YouTube URL")
        elif 'youtube.com/embed/' in video_input:
            # Embed URL format: https://www.youtube.com/embed/VIDEO_ID
            video_id = video_input.split('embed/')[1].split('?')[0]
        else:
            # Assume it's already a video ID
            video_id = video_input
        
        # Clean any remaining parameters
        video_id = video_id.split('?')[0].split('&')[0]
        
        if len(video_id) != 11:
            raise ValueError(f"Invalid YouTube video ID: {video_id} (should be 11 characters)")
        
        return video_id
    
    def scrape_transcript(self, video_id: str) -> dict:
        """
        Scrape transcript for a YouTube video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Transcript data from Apify
        """
        print(f"Starting transcript scrape for video: {video_id}")
        
        # Prepare the actor input - use videoUrl instead of videoIds
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        run_input = {
            "videoUrl": video_url,
            "language": "en",  # Default to English, can be made configurable
            "format": "json"
        }
        
        try:
            # Run the actor
            print("Running Apify actor...")
            run = self.client.actor(self.actor_id).call(run_input=run_input)
            
            # Fetch results
            print("Fetching results...")
            results = []
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                results.append(item)
            
            if not results:
                raise Exception("No transcript data returned from Apify")
            
            print(f"Successfully scraped transcript with {len(results)} items")
            return results[0]  # Return the first (and likely only) result
            
        except Exception as e:
            raise Exception(f"Error scraping transcript: {str(e)}")
    
    def save_transcript(self, video_id: str, transcript_data: dict) -> str:
        """
        Save transcript data to JSON file.
        
        Args:
            video_id: YouTube video ID
            transcript_data: Transcript data to save
            
        Returns:
            Path to saved file
        """
        # ensure transcripts directory exists
        transcripts_dir = Path(__file__).resolve().parents[1] / 'transcripts'
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        output_file = transcripts_dir / f"{video_id}.json"

        print(f"Saving transcript to: {output_file}")

        try:
            # If file exists and contains success: false, overwrite it. Otherwise overwrite normally.
            if output_file.exists():
                try:
                    with open(output_file, 'r', encoding='utf-8') as fr:
                        existing = json.load(fr)
                        if isinstance(existing, dict) and existing.get('data', {}).get('success') is False:
                            print(f"Overwriting previous failed transcript file: {output_file}")
                        else:
                            print(f"Overwriting existing transcript file: {output_file}")
                except Exception:
                    # if file is not valid json or cannot be read, just overwrite
                    print(f"Existing file present but could not be read as JSON. Overwriting: {output_file}")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
            file_size = Path(output_file).stat().st_size
            print(f"Transcript saved successfully ({file_size} bytes)")
            return output_file
            
        except Exception as e:
            raise Exception(f"Error saving transcript: {str(e)}")
    
    def process_video(self, video_input: str) -> str:
        """
        Process a YouTube video to extract and save its transcript.
        
        Args:
            video_input: YouTube URL or video ID
            
        Returns:
            Path to saved transcript file
        """
        # Extract clean video ID
        video_id = self.extract_video_id(video_input)
        print(f"Processing video ID: {video_id}")
        
        # Check if transcript already exists in transcripts/ and skip if success:true
        transcripts_dir = Path(__file__).resolve().parents[1] / 'transcripts'
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = transcripts_dir / f"{video_id}.json"
        if output_file_path.exists():
            try:
                with open(output_file_path, 'r', encoding='utf-8') as fr:
                    existing = json.load(fr)
                    if isinstance(existing, dict) and existing.get('data', {}).get('success') is False:
                        # overwrite failed file
                        print(f"Existing transcript has success:false — will attempt to re-download and overwrite: {output_file_path}")
                    else:
                        print(f"Transcript already exists and appears successful: {output_file_path}. Skipping download.")
                        return str(output_file_path)
            except Exception:
                # if unreadable, proceed to overwrite
                print(f"Existing transcript file present but unreadable; will attempt to overwrite: {output_file_path}")
        
    # Scrape transcript with retries for transient/unsuccessful responses
        attempt = 0
        last_exception = None
        transcript_data = None
        while attempt < self.max_retries:
            attempt += 1
            try:
                transcript_data = self.scrape_transcript(video_id)

                # If Apify/actor returns a structure that indicates failure, handle retry
                # Be defensive: transcript_data may be a list, dict with data:list, or dict with data: {success: false}
                if isinstance(transcript_data, dict) and isinstance(transcript_data.get('data'), dict) and transcript_data['data'].get('success') is False:
                    msg = transcript_data.get('data', {}).get('message', '')
                    print(f"Attempt {attempt}/{self.max_retries}: actor returned success=false: {msg}")
                    last_exception = Exception(f"actor returned success=false: {msg}")
                    # Save or overwrite failed file immediately so we have a record (will be overwritten on success)
                    try:
                        self.save_transcript(video_id, transcript_data)
                    except Exception as e:
                        print(f"Failed to write failed transcript file: {e}")
                    # backoff before retrying
                    if attempt < self.max_retries:
                        sleep_time = self.backoff_factor * attempt
                        print(f"Retrying after {sleep_time:.1f}s...")
                        import time
                        time.sleep(sleep_time)
                        continue
                    else:
                        break

                # If we get here, transcript_data is assumed good
                break

            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * attempt
                    print(f"Retrying after {sleep_time:.1f}s...")
                    import time
                    time.sleep(sleep_time)
                else:
                    break

        if transcript_data is None and last_exception is not None:
            raise last_exception

        # Save to file
        saved_file = self.save_transcript(video_id, transcript_data)

        # Display summary: support multiple transcript shapes returned by actor
        segments = None
        if isinstance(transcript_data, list):
            segments = transcript_data
        elif isinstance(transcript_data, dict):
            if isinstance(transcript_data.get('data'), list):
                segments = transcript_data.get('data')
            elif isinstance(transcript_data.get('transcript'), list):
                segments = transcript_data.get('transcript')

        if isinstance(segments, list):
            total_segments = len(segments)
            print(f"Transcript contains {total_segments} segments")

            if total_segments > 0:
                first_segment = segments[0]
                last_segment = segments[-1]

                if isinstance(first_segment, dict) and isinstance(last_segment, dict) and 'start' in first_segment and 'start' in last_segment and 'dur' in last_segment:
                    try:
                        duration = float(last_segment['start']) + float(last_segment['dur'])
                        print(f"Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
                    except (ValueError, TypeError):
                        pass
        
        return saved_file


def fetch_playlist_video_ids(playlist_input):
    """
    Fetch all video IDs from a YouTube playlist URL or ID using web scraping (no API key required).
    Args:
        playlist_input: Playlist URL or ID
    Returns:
        List of video IDs
    """
    # Extract playlist ID
    playlist_id = None
    if 'list=' in playlist_input:
        playlist_id = playlist_input.split('list=')[1].split('&')[0]
    elif re.match(r'^[A-Za-z0-9_-]{13,}$', playlist_input):
        playlist_id = playlist_input
    else:
        raise ValueError("Invalid playlist input. Provide a playlist URL or ID.")

    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
    print(f"Fetching playlist: {playlist_url}")
    resp = requests.get(playlist_url)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch playlist page: {resp.status_code}")
    # Find all video IDs in the page
    video_ids = re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', resp.text)
    # Remove duplicates
    video_ids = list(dict.fromkeys(video_ids))
    if not video_ids:
        raise Exception("No video IDs found in playlist.")
    print(f"Found {len(video_ids)} videos in playlist.")
    return video_ids

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="YouTube Transcript Scraper")
    parser.add_argument("input", help="YouTube video ID/URL or playlist ID/URL")
    parser.add_argument("--playlist", action="store_true", help="Treat input as a playlist and extract all videos")
    args = parser.parse_args()

    try:
        scraper = YouTubeTranscriptScraper()
        if args.playlist:
            video_ids = fetch_playlist_video_ids(args.input)
            for idx, vid in enumerate(video_ids, 1):
                print(f"\nProcessing video {idx}/{len(video_ids)}: {vid}")
                try:
                    output_file = scraper.process_video(vid)
                    print(f"✅ Transcript saved to: {output_file}")
                except Exception as e:
                    print(f"❌ Error processing {vid}: {e}")
        else:
            output_file = scraper.process_video(args.input)
            print(f"\n✅ Success! Transcript saved to: {output_file}")
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nTo set up Apify API token:")
        print("1. Sign up at: https://apify.com/")
        print("2. Get your API token from: https://console.apify.com/account/integrations")
        print("3. Set environment variable: export APIFY_API_TOKEN='your-api-token'")
        print("4. Or create a .env file with: APIFY_API_TOKEN=your-api-token")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()