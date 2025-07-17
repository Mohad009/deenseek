import json
import os
import re

def extract_segments_from_transcript(input_file_path, output_file_path=None):
    """
    Extract video link and segments from transcript JSON file.
    
    Args:
        input_file_path (str): Path to the input transcript JSON file
        output_file_path (str): Path for output file (optional, will auto-generate if not provided)
    
    Returns:
        dict: Extracted data in the desired format
    """
    
    # Read the input JSON file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Extract video ID from filename
    filename = os.path.basename(input_file_path)
    # Extract video ID (assuming format: videoId_transcript.json)
    video_id_match = re.search(r'([^_\\]+)_transcript\.json', filename)
    video_id = video_id_match.group(1) if video_id_match else filename.replace('_transcript.json', '')
    
    # Create video link (assuming YouTube)
    video_link = f"https://www.youtube.com/watch?v={video_id}"
    
    # Extract segments with only start, end, and text
    segments = []
    if 'segments' in data:
        for segment in data['segments']:
            extracted_segment = {
                "start": segment.get("start"),
                "end": segment.get("end"), 
                "text": segment.get("text")
            }
            segments.append(extracted_segment)
    
    # Create output structure
    output_data = {
        "video_link": video_link,
        "segments": segments
    }
    
    # Generate output filename if not provided
    if output_file_path is None:
        base_name = os.path.splitext(input_file_path)[0]
        output_file_path = f"{base_name}_extracted.json"
    
    # Write output JSON file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=2)
    
    print(f"Extracted data saved to: {output_file_path}")
    print(f"Video Link: {video_link}")
    print(f"Number of segments: {len(segments)}")
    
    return output_data

def process_all_transcripts_in_directory(directory_path):
    """
    Process all transcript JSON files in a directory.
    
    Args:
        directory_path (str): Path to directory containing transcript files
    """
    
    for filename in os.listdir(directory_path):
        if filename.endswith('_transcript.json'):
            input_path = os.path.join(directory_path, filename)
            print(f"\nProcessing: {filename}")
            extract_segments_from_transcript(input_path)

# Example usage
if __name__ == "__main__":
    # Process the current file
    input_file = r"c:\Users\mralh\Desktop\transcriptions\ipl5umkF5l0\ipl5umkF5l0_transcript.json"
    
    # Extract segments from the current file
    result = extract_segments_from_transcript(input_file)
    
    # Uncomment the line below to process all transcript files in the transcriptions directory
    # process_all_transcripts_in_directory(r"c:\Users\mralh\Desktop\transcriptions")
