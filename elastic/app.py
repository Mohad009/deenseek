from flask import Flask, request, jsonify, render_template_string
from elasticsearch import Elasticsearch
import json
import numpy as np

# Import AraBERT search engine
try:
    from arabert_search import AraBERTSearchEngine
    ARABERT_AVAILABLE = True
    print("AraBERT integration loaded successfully!")
except ImportError as e:
    print(f"AraBERT not available: {e}")
    ARABERT_AVAILABLE = False

app = Flask(__name__)

# Elasticsearch connection
es = Elasticsearch('http://localhost:9200', basic_auth=('elastic', 'Fe1odvmZ'))

# Initialize AraBERT search engine if available
if ARABERT_AVAILABLE:
    try:
        arabert_engine = AraBERTSearchEngine()
        print("AraBERT search engine initialized")
    except Exception as e:
        print(f"Error initializing AraBERT: {e}")
        ARABERT_AVAILABLE = False
        arabert_engine = None
else:
    arabert_engine = None

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <title>البحث في المحاضرات الصوتية</title>
    <meta charset="UTF-8">
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 40px; 
            direction: rtl; 
            text-align: right;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .search-box { 
            margin-bottom: 30px; 
            text-align: center;
        }
        input[type="text"] { 
            width: 400px; 
            padding: 15px; 
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 25px;
            margin-left: 10px;
            outline: none;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            border-color: #3498db;
        }
        select {
            padding: 15px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 25px;
            margin-left: 10px;
            outline: none;
            background: white;
        }
        button { 
            padding: 15px 30px; 
            background: #3498db; 
            color: white; 
            border: none; 
            cursor: pointer;
            font-size: 16px;
            border-radius: 25px;
            transition: background-color 0.3s;
        }
        button:hover {
            background: #2980b9;
        }
        .load-more-btn {
            background: #27ae60;
            margin: 20px auto;
            display: block;
        }
        .load-more-btn:hover {
            background: #219a52;
        }
        .result { 
            border: 1px solid #ddd; 
            margin: 15px 0; 
            padding: 20px; 
            border-radius: 8px;
            background: #fff;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .video-link { 
            color: #e74c3c; 
            text-decoration: none; 
            font-weight: bold;
            display: inline-block;
            margin-top: 10px;
            padding: 8px 15px;
            background: #fff5f5;
            border-radius: 5px;
            border: 1px solid #e74c3c;
            transition: all 0.3s;
        }
        .video-link:hover {
            background: #e74c3c;
            color: white;
        }
        .time { 
            color: #7f8c8d; 
            font-size: 14px;
            margin: 10px 0;
        }
        .text-content {
            font-size: 16px;
            line-height: 1.6;
            color: #2c3e50;
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            border-right: 4px solid #3498db;
        }
        .score {
            background: #f39c12;
            color: white;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 12px;
            display: inline-block;
            margin-right: 10px;
        }
        .search-mode {
            background: #9b59b6;
            color: white;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 12px;
            display: inline-block;
            margin-right: 10px;
        }
        .result-number {
            background: #3498db;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 14px;
            display: inline-block;
            margin-bottom: 10px;
        }
        .results-info {
            text-align: center;
            margin: 20px 0;
            color: #7f8c8d;
            font-size: 16px;
        }
        .no-results {
            text-align: center;
            color: #7f8c8d;
            font-size: 18px;
            margin: 50px 0;
        }
        .loading {
            text-align: center;
            color: #3498db;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 البحث في المحاضرات الصوتية</h1>
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="ابحث في النصوص... مثال: صلاة السفر">
            <select id="searchMode">
                <option value="enhanced">بحث محسّن (AraBERT)</option>
                <option value="basic">بحث أساسي</option>
                <option value="semantic">بحث دلالي</option>
            </select>
            <select id="resultLimit">
                <option value="10">10 نتائج</option>
                <option value="20">20 نتيجة</option>
                <option value="50" selected>50 نتيجة</option>
                <option value="100">100 نتيجة</option>
                <option value="200">200 نتيجة</option>
            </select>
            <button onclick="search()">بحث</button>
        </div>
        <div id="resultsInfo" class="results-info"></div>
        <div id="results"></div>
        <button id="loadMoreBtn" class="load-more-btn" onclick="loadMore()" style="display: none;">
            تحميل المزيد من النتائج
        </button>
    </div>

    <script>
        let isSearching = false;
        let currentQuery = '';
        let currentResults = [];
        let displayedCount = 0;
        let totalResults = 0;
        const LOAD_MORE_INCREMENT = 20;
        
        function search() {
            if (isSearching) return;
            
            const query = document.getElementById('searchInput').value.trim();
            const limit = parseInt(document.getElementById('resultLimit').value);
            const searchMode = document.getElementById('searchMode').value;
            
            if (!query) {
                alert('يرجى إدخال كلمة للبحث');
                return;
            }
            
            currentQuery = query;
            displayedCount = 0;
            isSearching = true;
            
            const resultsDiv = document.getElementById('results');
            const resultsInfo = document.getElementById('resultsInfo');
            const loadMoreBtn = document.getElementById('loadMoreBtn');
            
            resultsDiv.innerHTML = '<div class="loading">🔄 جاري البحث...</div>';
            resultsInfo.innerHTML = '';
            loadMoreBtn.style.display = 'none';
            
            fetch('/search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query, size: limit, mode: searchMode})
            })
            .then(response => response.json())
            .then(data => {
                currentResults = data.results || [];
                totalResults = data.total || currentResults.length;
                
                resultsDiv.innerHTML = '';
                resultsInfo.innerHTML = `تم العثور على ${totalResults} نتيجة باستخدام ${getSearchModeText(searchMode)}`;
                
                if (currentResults.length > 0) {
                    displayResults(Math.min(LOAD_MORE_INCREMENT, currentResults.length));
                    
                    if (currentResults.length > LOAD_MORE_INCREMENT) {
                        loadMoreBtn.style.display = 'block';
                    }
                } else {
                    resultsDiv.innerHTML = '<div class="no-results">❌ لم يتم العثور على نتائج لهذا البحث</div>';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                resultsDiv.innerHTML = '<div class="no-results">❌ حدث خطأ أثناء البحث</div>';
            })
            .finally(() => {
                isSearching = false;
            });
        }
        
        function getSearchModeText(mode) {
            switch(mode) {
                case 'enhanced': return 'البحث المحسّن';
                case 'semantic': return 'البحث الدلالي';
                case 'basic': return 'البحث الأساسي';
                default: return 'البحث المحسّن';
            }
        }
        
        function displayResults(count) {
            const resultsDiv = document.getElementById('results');
            const startIndex = displayedCount;
            const endIndex = Math.min(startIndex + count, currentResults.length);
            
            for (let i = startIndex; i < endIndex; i++) {
                const result = currentResults[i];
                const div = document.createElement('div');
                div.className = 'result';
                
                let scoreDisplay = '';
                if (result.score) {
                    scoreDisplay = `<span class="score">النقاط: ${result.score.toFixed(2)}</span>`;
                }
                
                div.innerHTML = `
                    <div class="result-number">النتيجة ${i + 1}</div>
                    ${scoreDisplay}
                    <div class="text-content">${result.text}</div>
                    <div class="time">⏰ الوقت: من ${result.start} ثانية إلى ${result.end} ثانية</div>
                    <a href="${result.youtube_link_with_time}" target="_blank" class="video-link">
                        🎥 مشاهدة الفيديو من هذه النقطة
                    </a>
                `;
                resultsDiv.appendChild(div);
            }
            
            displayedCount = endIndex;
            
            // Hide load more button if all results are displayed
            if (displayedCount >= currentResults.length) {
                document.getElementById('loadMoreBtn').style.display = 'none';
            }
        }
        
        function loadMore() {
            displayResults(LOAD_MORE_INCREMENT);
        }
        
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                search();
            }
        });
        
        // Focus on search input when page loads
        document.getElementById('searchInput').focus();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search_term = data.get('query', '')
    size = data.get('size', 50)  # Default to 50 results
    mode = data.get('mode', 'enhanced')  # Default to enhanced search
    
    if not search_term:
        return jsonify({'results': [], 'total': 0})
    
    try:
        if mode == 'enhanced' and ARABERT_AVAILABLE and arabert_engine:
            # Use enhanced search with AraBERT
            results = enhanced_search(search_term, size)
        elif mode == 'semantic' and ARABERT_AVAILABLE and arabert_engine:
            # Use semantic search with embeddings
            results = semantic_search(search_term, size)
        else:
            # Fallback to basic search
            results = basic_search(search_term, size)
        
        return jsonify({
            'results': results['hits'],
            'total': results['total'],
            'returned': len(results['hits']),
            'mode': mode
        })
    
    except Exception as e:
        print(f"Search error: {e}")
        # Fallback to basic search on error
        try:
            results = basic_search(search_term, size)
            return jsonify({
                'results': results['hits'],
                'total': results['total'],
                'returned': len(results['hits']),
                'mode': 'basic_fallback'
            })
        except Exception as e2:
            return jsonify({'error': str(e2), 'results': [], 'total': 0}), 500

def basic_search(search_term, size=50):
    """Basic Elasticsearch search"""
    query = {
        "match": {
            "text": search_term
        }
    }
    
    # Get total count first
    count_response = es.count(index='transcription', query=query)
    total_count = count_response['count']
    
    # Get the actual results
    response = es.search(index='transcription', query=query, size=min(size, 1000))
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
            'score': hit['_score']
        })
    
    return {
        'hits': results,
        'total': total_count
    }

def enhanced_search(search_term, size=50):
    """Enhanced search using AraBERT and multiple Elasticsearch strategies"""
    
    # Create enhanced query
    query = arabert_engine.create_enhanced_query(search_term)
    
    # Try enhanced index first, fallback to basic index
    index_name = 'transcription_with_embeddings'
    if not es.indices.exists(index=index_name):
        index_name = 'transcription'
    
    # Get total count
    count_response = es.count(index=index_name, query=query)
    total_count = count_response['count']
    
    # Get results
    response = es.search(
        index=index_name, 
        query=query, 
        size=min(size, 1000),
        sort=[{"_score": {"order": "desc"}}]
    )
    
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
            'score': hit['_score']
        })
    
    return {
        'hits': results,
        'total': total_count
    }

def semantic_search(search_term, size=50):
    """Semantic search using AraBERT embeddings"""
    
    # Check if embeddings index exists
    index_name = 'transcription_with_embeddings'
    if not es.indices.exists(index=index_name):
        print("Embeddings index not found, falling back to enhanced search")
        return enhanced_search(search_term, size)
    
    try:
        # Get query embedding
        query_embedding = arabert_engine.get_embedding(search_term)
        
        # Create hybrid query (semantic + lexical)
        query = arabert_engine.create_hybrid_query(search_term, query_embedding)
        
        # Get total count (approximate for script_score queries)
        basic_query = arabert_engine.create_enhanced_query(search_term)
        count_response = es.count(index=index_name, query=basic_query)
        total_count = count_response['count']
        
        # Get results
        response = es.search(
            index=index_name,
            query=query,
            size=min(size, 1000),
            sort=[{"_score": {"order": "desc"}}]
        )
        
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
                'score': hit['_score']
            })
        
        return {
            'hits': results,
            'total': total_count
        }
        
    except Exception as e:
        print(f"Semantic search error: {e}")
        return enhanced_search(search_term, size)

@app.route('/api/search')
def api_search():
    search_term = request.args.get('q', '')
    size = int(request.args.get('size', 50))
    
    if not search_term:
        return jsonify({'results': [], 'total': 0})
    
    query = {
        "match": {
            "text": search_term
        }
    }
    
    try:
        count_response = es.count(index='transcription', query=query)
        total_count = count_response['count']
        
        response = es.search(index='transcription', query=query, size=min(size, 1000))
        results = []
        
        for hit in response['hits']['hits']:
            results.append({
                'text': hit['_source']['text'],
                'start': hit['_source']['start'],
                'end': hit['_source']['end'],
                'video_link': hit['_source']['video_link']
            })
        
        return jsonify({
            'results': results,
            'total': total_count,
            'returned': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'results': [], 'total': 0}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)