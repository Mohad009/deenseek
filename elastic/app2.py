from flask import Flask, request, jsonify, render_template
from elasticsearch import Elasticsearch, exceptions as es_exceptions
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Elasticsearch connection with error handling
es_host = ""
api_key = ""

def get_elasticsearch_client():
    try:
        es_client = Elasticsearch(
            es_host,
            api_key=api_key
        )
        # Verify connection
        if not es_client.ping():
            logger.error("Failed to connect to Elasticsearch - ping failed")
            return None
        return es_client
    except Exception as e:
        logger.error(f"Failed to establish Elasticsearch connection: {str(e)}")
        return None

# Initialize connection
es = get_elasticsearch_client()

# Ensure templates directory exists
os.makedirs('templates', exist_ok=True)

@app.route('/')
def index():
    # Check if ES is available
    global es
    if es is None or not es.ping():
        # Try to reconnect
        es = get_elasticsearch_client()
    
    # Still proceed with template rendering even if ES is down
    # The UI can show appropriate error messages
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Check if ES is available
    global es
    if es is None or not es.ping():
        # Try to reconnect
        es = get_elasticsearch_client()
        if es is None:
            logger.error("Elasticsearch is unavailable during search request")
            return jsonify({
                'error': 'Database connection unavailable. Please try again later.',
                'results': [],
                'total': 0
            }), 503  # Service Unavailable
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided', 'results': [], 'total': 0}), 400
        
        search_term = data.get('query', '').strip()
        size = data.get('size', 50)
        
        # Validate parameters
        if not search_term:
            return jsonify({'error': 'Search query is required', 'results': [], 'total': 0}), 400
        
        if not isinstance(size, int) or size < 1:
            size = 50  # Default if invalid
        
        # Vector search using KNN with gate-arabert-v1-doc model
        retriever_object = {
            "standard": {
                "query": {
                    "knn": {
                        "field": "vector",
                        "num_candidates": min(size * 2, 100),
                        "query_vector_builder": {
                            "text_embedding": {
                                "model_id": "",
                                "model_text": search_term
                            }
                        }
                    }
                }
            }
        }
        
        # Add timeout to prevent long-running queries
        start_time = time.time()
        response = es.search(
            index="gate_transcription",
            retriever=retriever_object,
            size=min(size, 1000),
            request_timeout=30  # 30-second timeout
        )
        query_time = time.time() - start_time
        logger.info(f"Search completed in {query_time:.2f}s for query: '{search_term}'")
        
        # Get total count (this is approximate for kNN searches)
        total_count = response['hits']['total']['value']
        
        results = []
        
        for hit in response['hits']['hits']:
            source = hit.get('_source', {})
            
            # Safely get values with fallbacks
            video_link = source.get('video_link', '')
            start_time = int(source.get('start', 0))
            
            # Create YouTube link with timestamp
            youtube_link_with_time = ''
            if video_link:
                try:
                    if 'youtube.com/watch?v=' in video_link:
                        video_id = video_link.split('v=')[1].split('&')[0]
                    elif 'youtu.be/' in video_link:
                        video_id = video_link.split('youtu.be/')[-1].split('?')[0]
                    else:
                        video_id = video_link
                    
                    youtube_link_with_time = f"https://www.youtube.com/watch?v={video_id}&t={start_time}s"
                except Exception as e:
                    logger.warning(f"Failed to process video link: {str(e)}")
            
            results.append({
                'text': source.get('text', ''),
                'start': source.get('start', 0),
                'end': source.get('end', 0),
                'video_link': video_link,
                'youtube_link_with_time': youtube_link_with_time,
                'score': hit.get('_score', 0)
            })
        
        return jsonify({
            'results': results,
            'total': total_count,
            'returned': len(results),
            'query_time': f"{query_time:.2f}s"
        })
    
    except es_exceptions.ConnectionError as e:
        logger.error(f"Elasticsearch connection error: {str(e)}")
        return jsonify({
            'error': 'Could not connect to search service. Please try again later.',
            'results': [],
            'total': 0
        }), 503
    
    except es_exceptions.NotFoundError as e:
        logger.error(f"Index not found: {str(e)}")
        return jsonify({
            'error': 'Search index not found.',
            'results': [],
            'total': 0
        }), 404
    
    except es_exceptions.AuthenticationException as e:
        logger.error(f"Authentication error: {str(e)}")
        return jsonify({
            'error': 'Authentication error with search service.',
            'results': [],
            'total': 0
        }), 503
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({
            'error': f"An unexpected error occurred: {str(e)}",
            'results': [],
            'total': 0
        }), 500


# Add a health check endpoint
@app.route('/health')
def health_check():
    health_status = {
        'status': 'UP',
        'elasticsearch': 'DOWN',
        'details': {}
    }
    
    # Check Elasticsearch
    try:
        if es and es.ping():
            health_status['elasticsearch'] = 'UP'
            # Get cluster health
            cluster_health = es.cluster.health()
            health_status['details']['elasticsearch'] = {
                'status': cluster_health.get('status'),
                'nodes': cluster_health.get('number_of_nodes'),
                'version': es.info().get('version', {}).get('number', 'unknown')
            }
        else:
            health_status['status'] = 'DEGRADED'
    except Exception as e:
        health_status['status'] = 'DEGRADED'
        health_status['details']['elasticsearch_error'] = str(e)
    
    status_code = 200 if health_status['status'] == 'UP' else 503
    return jsonify(health_status), status_code


if __name__ == '__main__':
    # Check if Elasticsearch is available at startup
    if es is None:
        logger.warning("Starting application without Elasticsearch connection. Search functionality will be unavailable.")
    app.run(debug=True)