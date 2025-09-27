import json
import os
from glob import glob
from tqdm import tqdm

def transform_file(input_file, output_dir):
    """Transform a transcription file with segments into flat array format for direct ES insertion"""
    try:
        # Read the original file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create an array of individual documents
        documents = []
        
        # Process each segment
        for i, segment in enumerate(data.get('segment', [])):
            doc = {
                'video_link': data.get('video_link', ''),
                'segment_index': i,
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', '')
            }
            documents.append(doc)
        
        # Create output filename
        base_filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"es_{base_filename}")
        
        # Write the transformed file (flat array format)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        return output_file
    
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return None

def main():
    # Directories
    input_dir = "transcriptions_1445h_v3 - Copy"
    output_dir = "transcriptions_1445h_v4"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = glob(os.path.join(input_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file with progress bar
    processed_files = []
    for file_path in tqdm(json_files, desc="Transforming files"):
        output_file = transform_file(file_path, output_dir)
        if output_file:
            processed_files.append(output_file)
    
    print(f"\nSuccessfully processed {len(processed_files)} files")
    print(f"Transformed files are in: {output_dir}")
    

if __name__ == "__main__":
    main()