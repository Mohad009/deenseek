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
# is_production = os.environ.get('ENVIRONMENT', 'development') == 'production'

# Elasticsearch connection
es = Elasticsearch(
    os.getenv("ElasticURL"),
    api_key=os.getenv("ElasticAPIKey"),
   
    retry_on_timeout=True,
    max_retries=3
)





# Ensure templates directory exists
os.makedirs('templates', exist_ok=True)

    
# @app.route('/response',methods=['GET'])
# def model_status():
#     try:
#         resp = es.inference.get(
#         task_type="text_embedding",
#         inference_id=os.getenv("MODEL_ID"),
#                         )
#         return jsonify(resp), 500
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

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
            index=os.getenv("ES_INDEX"),
            retriever=retriever_object,
            size=min(size, 1000)
        )
        # Collect matched doc ids and group_ids from initial vector hits
        hits = response.get('hits', {}).get('hits', [])
        matching_doc_ids = set()
        group_ids = set()
        initial_scores = {}
        for hit in hits:
            src = hit.get('_source', {})
            doc_id = src.get('doc_id')
            if doc_id:
                matching_doc_ids.add(doc_id)
                initial_scores[doc_id] = hit.get('_score', 0)
            if src.get('group_id'):
                group_ids.add(src.get('group_id'))

        if not group_ids:
            return jsonify({'results': [], 'total': 0, 'returned': 0})

        # Fetch all documents that belong to the matched groups so we can display full context
        fetch_size = max(100, len(group_ids) * 10)
        fetch_body = {
            "query": {"terms": {"group_id": list(group_ids)}},
            "size": fetch_size,
            "sort": [
                {"group_id": {"order": "asc"}},
                {"sequence": {"order": "asc"}}
            ]
        }

        fetch_resp = es.search(index=os.getenv("ES_INDEX"), body=fetch_body)
        groups = {}
        for doc in fetch_resp.get('hits', {}).get('hits', []):
            src = doc.get('_source', {})
            gid = src.get('group_id')
            if not gid:
                continue
            item = {
                'doc_id': src.get('doc_id'),
                'is_follow_up': src.get('is_follow_up', False),
                'sequence': src.get('sequence', 0),
                'question': src.get('question', ''),
                'answer': src.get('answer', ''),
                'start': src.get('start', 0),
                'end': src.get('end', 0),
                'video_link': src.get('video_link', ''),
                'is_match': src.get('doc_id') in matching_doc_ids,
                'score': initial_scores.get(src.get('doc_id'), 0)
            }
            groups.setdefault(gid, []).append(item)

        # Build results grouped and sorted by sequence
        results = []
        for gid, items in groups.items():
            items_sorted = sorted(items, key=lambda x: x.get('sequence', 0))
            formatted_items = []
            for it in items_sorted:
                start_time = int(it.get('start') or 0)
                video_link = it.get('video_link') or ''
                if 'youtube.com/watch?v=' in video_link:
                    vid = video_link.split('v=')[1].split('&')[0]
                elif 'youtu.be/' in video_link:
                    vid = video_link.split('youtu.be/')[-1].split('?')[0]
                else:
                    vid = video_link
                youtube_link_with_time = f"https://www.youtube.com/watch?v={vid}&t={start_time}s"
                formatted_items.append({
                    'doc_id': it.get('doc_id'),
                    'is_follow_up': it.get('is_follow_up'),
                    'question': it.get('question'),
                    'answer': it.get('answer'),
                    'start': format_time(start_time),
                    'end': format_time(it.get('end')),
                    'video_link': it.get('video_link'),
                    'youtube_link_with_time': youtube_link_with_time,
                    'is_match': it.get('is_match'),
                    'score': it.get('score')
                })
            results.append({'group_id': gid, 'items': formatted_items})

        total_groups = len(results)
        return jsonify({'results': results, 'total': total_groups, 'returned': len(results)})
    
    except Exception as e:
        return jsonify({'error': str(e), 'results': [], 'total': 0}), 500



if __name__ == '__main__':
    app.run()