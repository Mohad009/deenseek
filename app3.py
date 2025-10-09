from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os
import time

def format_time(seconds):
    """Convert seconds to MM:SS format using built-in time functions"""
    if not seconds or not isinstance(seconds, (int, float)):
        return "00:00"
    
    # Convert to int to ensure we have whole seconds
    seconds = int(float(seconds))
    
    # Use time.strftime with time.gmtime
    return time.strftime("%M:%S", time.gmtime(seconds))

app = Flask(__name__)
CORS(app)
load_dotenv()

# Configure for production or development
is_production = os.environ.get('ENVIRONMENT', 'development') == 'production'

# Elasticsearch connection
es = Elasticsearch(
    os.getenv("ElasticURL"),
    api_key=os.getenv("ElasticAPIKey"),
   
    retry_on_timeout=True,
    max_retries=3
)

# Ensure templates directory exists
os.makedirs('templates', exist_ok=True)
@app.route('/fast', methods=['GET'])
def debug():
    import sys
    import platform
    
    try:
        # Test Elasticsearch
        es_status = "Not connected"
        es_version = "Unknown"
        try:
            info = es.info()
            es_status = "Connected"
            es_version = info.get("version", {}).get("number", "Unknown")
        except Exception as es_error:
            es_status = f"Error: {str(es_error)}"
        
        # Check environment variables
        env_vars = {
            "MODEL_ID": os.getenv("MODEL_ID", "not set"),
            "ENVIRONMENT": os.environ.get("ENVIRONMENT", "not set"),
            "PORT": os.environ.get("PORT", "not set"),
            "ElasticURL": os.getenv("ElasticURL", "not set")[:10] + "..." if os.getenv("ElasticURL") else "not set",
            "ElasticAPIKey": "***" if os.getenv("ElasticAPIKey") else "not set"
        }
        
        # Check python environment
        python_info = {
            "version": sys.version,
            "platform": platform.platform(),
            "modules": [m.__name__ for m in sys.modules.values() if hasattr(m, "__name__")][:20]  # First 20 modules
        }
        
        return jsonify({
            "status": "ok",
            "elasticsearch": {
                "status": es_status,
                "version": es_version
            },
            "environment_variables": env_vars,
            "python_info": python_info
        })
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
    

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
                            "model_id": os.getenv("MODEL_ID"),  # Using specified Arabic model
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
            formatted_start_time = format_time(start_time)
            formatted_end_time = format_time(hit['_source'].get('end'))
            results.append({
                'text': hit['_source']['text'],
                'start': formatted_start_time,
                'end': formatted_end_time,
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
    # Use production settings when on Render or other production environments
    if is_production:
        # Let gunicorn handle the app in production
        pass
    else:
        # Use development server locally
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))