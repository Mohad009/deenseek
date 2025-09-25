#!/usr/bin/env python3
"""
Simple Audio Transcriber - Transcription only
"""

import os
import json
from pathlib import Path
import argparse

def check_whisper_module():
    """Check if Whisper module is available"""
    try:
        import whisper
        print(f"✓ Whisper module is available")
        available_models = whisper.available_models()
        print(f"   Available models: {', '.join(available_models)}")
        return True
    except ImportError as e:
        print(f"❌ Whisper module not found: {e}")
        print("💡 Install with: pip install openai-whisper")
        return False

def merge_short_segments(segments, min_duration=15.0, max_duration=120.0):
    """Merge segments to create longer, more meaningful chunks"""
    if not segments:
        return segments
    
    merged = []
    current_segment = segments[0].copy()
    
    for next_segment in segments[1:]:
        current_duration = current_segment["end"] - current_segment["start"]
        combined_duration = next_segment["end"] - current_segment["start"]
        
        # Merge if current is too short or combined duration is reasonable
        if (current_duration < min_duration and combined_duration <= max_duration):
            current_segment["end"] = next_segment["end"]
            current_segment["text"] += " " + next_segment["text"].strip()
        else:
            merged.append(current_segment)
            current_segment = next_segment.copy()
    
    # Don't forget the last segment
    merged.append(current_segment)
    return merged

def transcribe_audio(audio_file, output_dir, language='ar', model='turbo'):
    """
    Transcribe audio using Whisper Python API
    """
    try:
        import whisper
        
        video_id = Path(audio_file).stem
        print(f"🎤 Transcribing: {os.path.basename(audio_file)}")
        
        # Create output directory for this audio
        audio_dir = Path(output_dir) / video_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"   Loading {model} model...")
        whisper_model = whisper.load_model(model)
        
        # Transcribe with optimized settings for longer segments
        print(f"   Transcribing audio...")
        result = whisper_model.transcribe(
            str(audio_file),
            language=language,
            verbose=False,
            condition_on_previous_text=True,
            no_speech_threshold=0.7,
            compression_ratio_threshold=2.2
        )
        
        # Merge short segments for better database performance
        original_count = len(result["segments"])
        merged_segments = merge_short_segments(
            result["segments"], 
            min_duration=15.0,  # Minimum 15 seconds
            max_duration=120.0  # Maximum 2 minutes
        )
        
        print(f"   Segments: {original_count} → {len(merged_segments)} (merged)")
        
        # Save custom JSON format only
        base_name = f"{video_id}"
        
        # Create custom JSON structure
        custom_result = {
            "video_link": video_id,
            "segment": [
                {
                    "start": round(segment["start"], 2),
                    "end": round(segment["end"], 2),
                    "text": segment["text"].strip()
                }
                for segment in merged_segments
            ]
        }
        
        # Save custom JSON format
        json_file = audio_dir / f"{base_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(custom_result, f, ensure_ascii=False, indent=4)
        
        print(f"✓ Transcription completed for {video_id}")
        print(f"   File saved: {base_name}.json")
        print(f"   Final segments: {len(merged_segments)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error transcribing {audio_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_folder(input_folder, output_folder="transcriptions", language='ar', model='turbo'):
    """Process all audio files in folder"""
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"❌ Input folder not found: {input_folder}")
        return
    
    # Find audio files (MP3, WAV, M4A, etc.)
    audio_extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac', '*.aac']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(ext))
    
    if not audio_files:
        print(f"❌ No audio files found in: {input_folder}")
        return
    
    print(f"📁 Found {len(audio_files)} audio files")
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        audio_id = audio_file.stem
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_id}")
        
        # Check if already processed
        existing_transcript = output_path / audio_id / f"{audio_id}.json"
        if existing_transcript.exists():
            print(f"⚠️  Already transcribed: {audio_id}")
            successful += 1
            continue
        
        # Transcribe
        if transcribe_audio(audio_file, output_path, language, model):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Processing Summary:")
    print(f"✓ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output: {output_path.absolute()}")

def main():
    parser = argparse.ArgumentParser(description='Simple Audio Transcriber')
    parser.add_argument('input_folder', help='Folder with audio files')
    parser.add_argument('-o', '--output', default='transcriptions', help='Output folder')
    parser.add_argument('-l', '--language', default='ar', help='Language code (e.g., ar, en, fr)')
    parser.add_argument('--model', default='turbo', 
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'turbo'], 
                       help='Whisper model')
    
    args = parser.parse_args()
    
    print("🎤 Simple Audio Transcriber")
    print("=" * 50)
    print(f"📂 Input: {args.input_folder}")
    print(f"📂 Output: {args.output}")
    print(f"🌍 Language: {args.language}")
    print(f"🤖 Model: {args.model}")
    print("=" * 50)
    
    # Check dependencies
    if not check_whisper_module():
        print("\n❌ Missing Whisper module. Please install it first.")
        return
    
    # Process files
    process_folder(args.input_folder, args.output, args.language, args.model)

if __name__ == "__main__":
    main()