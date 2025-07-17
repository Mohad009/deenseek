import json
from pathlib import Path

def quick_convert(input_file):
    """Quick conversion of a single JSON file"""
    
    # Get video link from filename
    video_link = Path(input_file).stem
    
    # Read and process the file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create new structure
    result = {
        'video_link': video_link,
        'segment': [
            {
                'start': seg.get('start'),
                'end': seg.get('end'),
                'text': seg.get('text')
            }
            for seg in data.get('segments', [])
        ]
    }
    
    # Save converted file
    output_file = input_file.replace('.json', '_converted.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"Converted file saved as: {output_file}")
    return result

# Usage
converted_data = quick_convert("ipl5umkF5l0_transcript.json")