from flask import Flask, request, jsonify, render_template
from elasticsearch import Elasticsearch
import os

app = Flask(__name__)

# Elasticsearch connection
es = Elasticsearch(
    "",
    api_key=""
)

# Ensure templates directory exists
os.makedirs('templates', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search_term = data.get('query', '')
    size = data.get('size', 50)  # Default to 50 results
    
    if not search_term:
        return jsonify({'results': [], 'total': 0})
    
    # Vector search using KNN with gate-arabert-v1-doc model
    retriever_object = {
        "standard": {
            "query": {
                "knn": {
                    "field": "vector",
                    "num_candidates": min(size * 2, 100),
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": "",  # Using specified Arabic model
                            "model_text": search_term
                        }
                    }
                }
            }
        }
    }
    
    try:
        # Using vector search
        response = es.search(
            index="gate_transcription",
            retriever=retriever_object,
            size=min(size, 1000)
        )
        
        # Get total count (this is approximate for kNN searches)
        total_count = response['hits']['total']['value']
        
        results = []
        
        for hit in response['hits']['hits']:
            video_link = hit['_source']['video_link']
            start_time = int(hit['_source']['start'])
            
            # Create YouTube link with timestamp
            if 'youtube.com/watch?v=' in video_link:
                video_id = video_link.split('v=')[1].split('&')[0]
            elif 'youtu.be/' in video_link:
                video_id = video_link.split('youtu.be/')[-1].split('?')[0]
            else:
                video_id = video_link
            
            youtube_link_with_time = f"https://www.youtube.com/watch?v={video_id}&t={start_time}s"
            
            results.append({
                'text': hit['_source']['text'],
                'start': hit['_source']['start'],
                'end': hit['_source']['end'],
                'video_link': video_link,
                'youtube_link_with_time': youtube_link_with_time,
                'score': hit.get('_score', 0)  # Include the relevance score
            })
        
        return jsonify({
            'results': results,
            'total': total_count,
            'returned': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'results': [], 'total': 0}), 500



if __name__ == '__main__':
    app.run()