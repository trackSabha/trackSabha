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
        output_file = f"{video_id}.json"
        
        print(f"Saving transcript to: {output_file}")
        
        try:
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
        
        # Check if transcript already exists
        output_file = f"{video_id}.json"
        if Path(output_file).exists():
            print(f"Transcript file already exists: {output_file}")
            response = input("Overwrite existing file? (y/N): ").strip().lower()
            if response != 'y':
                print("Skipping download.")
                return output_file
        
        # Scrape transcript
        transcript_data = self.scrape_transcript(video_id)
        
        # Save to file
        saved_file = self.save_transcript(video_id, transcript_data)
        
        # Display summary
        if 'transcript' in transcript_data and isinstance(transcript_data['transcript'], list):
            total_segments = len(transcript_data['transcript'])
            print(f"Transcript contains {total_segments} segments")
            
            if total_segments > 0:
                first_segment = transcript_data['transcript'][0]
                last_segment = transcript_data['transcript'][-1]
                
                if 'start' in first_segment and 'start' in last_segment and 'dur' in last_segment:
                    try:
                        duration = float(last_segment['start']) + float(last_segment['dur'])
                        print(f"Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
                    except (ValueError, TypeError):
                        pass
        
        return saved_file

def main():
    """Main function to run the script."""
    if len(sys.argv) != 2:
        print("Usage: python youtube_transcript_scraper.py <video_id_or_url>")
        print("\nExamples:")
        print("python youtube_transcript_scraper.py dR-eoAEvPH4")
        print("python youtube_transcript_scraper.py https://www.youtube.com/watch?v=dR-eoAEvPH4")
        print("python youtube_transcript_scraper.py https://youtu.be/dR-eoAEvPH4")
        print("\nOutput will be saved as: <video_id>.json")
        sys.exit(1)
    
    video_input = sys.argv[1]
    
    try:
        # Initialize scraper
        scraper = YouTubeTranscriptScraper()
        
        # Process the video
        output_file = scraper.process_video(video_input)
        
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