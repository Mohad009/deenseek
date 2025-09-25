#!/usr/bin/env python3
"""
YouTube Playlist Downloader for Su'al Ahl al-Dhikr Series
Downloads videos as MP3 audio files from playlist and renames them with video IDs only
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path

def check_dependencies():
    """Check if yt-dlp is installed"""
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        print("✓ yt-dlp is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ yt-dlp is not installed. Please install it first:")
        print("pip install yt-dlp")
        sys.exit(1)

def get_playlist_info(playlist_url):
    """Extract playlist information including video IDs, titles, and playlist name"""
    print("🔍 Extracting playlist information...")
    
    # First get playlist metadata
    playlist_cmd = [
        'yt-dlp',
        '--dump-single-json',
        '--flat-playlist',
        playlist_url
    ]
    
    try:
        result = subprocess.run(playlist_cmd, capture_output=True, text=True, check=True)
        playlist_data = json.loads(result.stdout)
        
        playlist_title = playlist_data.get('title', 'Unknown_Playlist')
        print(f"📋 Playlist: {playlist_title}")
        
        # Extract videos info
        videos_info = []
        for entry in playlist_data.get('entries', []):
            if entry:
                videos_info.append({
                    'id': entry.get('id'),
                    'title': entry.get('title', 'Unknown'),
                    'url': f"https://www.youtube.com/watch?v={entry.get('id')}"
                })
        
        print(f"✓ Found {len(videos_info)} videos in playlist")
        return videos_info, playlist_title
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error extracting playlist info: {e}")
        return [], "Unknown_Playlist"

def extract_hijri_date(playlist_title):
    """Extract Hijri date from playlist title to use as folder name"""
    # Look for Hijri date patterns in the title
    # Common patterns: "1445هـ", "1445 هـ", "رمضان 1445", etc.
    hijri_patterns = [
        r'(\d{4})\s*هـ',  # 1445هـ or 1445 هـ
        r'(\d{4})\s*ه',   # Alternative heh
        r'(\d{4})',       # Just the year if no هـ found
    ]
    
    for pattern in hijri_patterns:
        match = re.search(pattern, playlist_title)
        if match:
            year = match.group(1)
            # Look for month names
            month_patterns = {
                'محرم': 'muharram',
                'صفر': 'safar', 
                'ربيع الأول': 'rabi_al_awwal',
                'ربيع الآخر': 'rabi_al_thani',
                'جمادى الأولى': 'jumada_al_awwal',
                'جمادى الآخرة': 'jumada_al_thani',
                'رجب': 'rajab',
                'شعبان': 'shaban',
                'رمضان': 'ramadan',
                'شوال': 'shawwal',
                'ذو القعدة': 'dhul_qadah',
                'ذو الحجة': 'dhul_hijjah'
            }
            
            for arabic_month, english_month in month_patterns.items():
                if arabic_month in playlist_title:
                    return f"{english_month}_{year}h"
            
            return f"{year}h"
    
    # Fallback: clean up the playlist title
    clean_title = re.sub(r'[^\w\s-]', '', playlist_title)
    clean_title = re.sub(r'\s+', '_', clean_title.strip())
    return clean_title[:50]  # Limit length

def download_video(video_info, download_dir):
    """Download a single video as MP3 and rename it with video ID"""
    video_id = video_info['id']
    video_url = video_info['url']
    
    print(f"🎵 Downloading audio: {video_info['title'][:50]}...")
    
    # Check if file already exists
    output_file = os.path.join(download_dir, f"{video_id}.mp3")
    if os.path.exists(output_file):
        print(f"⚠️  File {video_id}.mp3 already exists, skipping...")
        return True
    
    # Download as MP3 with video ID as filename
    output_filename = f"{video_id}.%(ext)s"
    
    cmd = [
        'yt-dlp',
        '--extract-audio',           # Extract audio only
        '--audio-format', 'mp3',     # Convert to MP3
        '--audio-quality', '192K',   # Good quality for speech
        '--output', os.path.join(download_dir, output_filename),
        '--no-playlist',             # Download single video
        video_url
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Downloaded: {video_id}.mp3")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading {video_id}: {e}")
        return False

def main():
    """Main function to download playlist"""
    playlist_url = "https://www.youtube.com/watch?v=zwDDYosdjHg&list=PLOnIdxOBLlJxajYFGbZKPIdDvQa6fdDmh"
    
    print("🎵 Su'al Ahl al-Dhikr Audio Downloader")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Get playlist information
    videos_info, playlist_title = get_playlist_info(playlist_url)
    
    if not videos_info:
        print("❌ No videos found in playlist")
        return
    
    # Create download directory based on Hijri date
    folder_name = extract_hijri_date(playlist_title)
    download_dir = folder_name
    
    Path(download_dir).mkdir(exist_ok=True)
    print(f"📁 Download directory: {download_dir}")
    print(f"📋 Based on playlist: {playlist_title}")
    
    # Download each video as MP3
    successful_downloads = 0
    failed_downloads = 0
    
    for i, video_info in enumerate(videos_info, 1):
        print(f"\n[{i}/{len(videos_info)}] Processing audio...")
        
        if download_video(video_info, download_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Download Summary:")
    print(f"✓ Successful: {successful_downloads}")
    print(f"❌ Failed: {failed_downloads}")
    print(f"🎵 MP3 files saved to: {os.path.abspath(download_dir)}")
    
    if failed_downloads > 0:
        print("\n⚠️  Some downloads failed. You can re-run the script to retry.")

if __name__ == "__main__":
    main()