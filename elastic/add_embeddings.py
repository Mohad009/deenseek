from elasticsearch import Elasticsearch
from arabert_search import AraBERTSearchEngine
import json
from tqdm import tqdm
import time

def add_embeddings_to_index():
    """Add AraBERT embeddings to existing documents"""
    
    print("Connecting to Elasticsearch...")
    es = Elasticsearch('http://localhost:9200', basic_auth=('elastic', 'Fe1odvmZ'))
    
    print("Initializing AraBERT search engine...")
    arabert_engine = AraBERTSearchEngine()
    
    # Check if original index exists
    if not es.indices.exists(index='transcription'):
        print("Error: Original 'transcription' index not found!")
        return
    
    # Update the index mapping to include dense vector field
    mapping = {
        "mappings": {
            "properties": {
                "text": {
                    "type": "text", 
                    "analyzer": "arabic"
                },
                "processed_text": {
                    "type": "text", 
                    "analyzer": "arabic"
                },
                "text_embedding": {
                    "type": "dense_vector",
                    "dims": 768,  # AraBERT embedding dimension
                    "index": True,
                    "similarity": "cosine"
                },
                "start": {"type": "float"},
                "end": {"type": "float"},
                "video_link": {"type": "keyword"}
            }
        },
        "settings": {
            "analysis": {
                "analyzer": {
                    "arabic": {
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "arabic_normalization",
                            "arabic_stem"
                        ]
                    }
                }
            }
        }
    }
    
    new_index_name = 'transcription_with_embeddings'
    
    try:
        # Delete existing index if it exists
        if es.indices.exists(index=new_index_name):
            print(f"Deleting existing index: {new_index_name}")
            es.indices.delete(index=new_index_name)
        
        # Create new index with embeddings
        es.indices.create(index=new_index_name, body=mapping)
        print(f"Created new index: {new_index_name}")
    except Exception as e:
        print(f"Error creating index: {e}")
        return
    
    # Get total document count
    try:
        total_response = es.count(index='transcription')
        total_docs = total_response['count']
        print(f"Total documents to process: {total_docs}")
    except Exception as e:
        print(f"Error counting documents: {e}")
        return
    
    if total_docs == 0:
        print("No documents found in original index!")
        return
    
    # Process documents in batches
    batch_size = 50
    processed_count = 0
    batch_docs = []
    
    try:
        # Use scroll to process all documents
        response = es.search(
            index='transcription',
            query={"match_all": {}},
            size=batch_size,
            scroll='10m'
        )
        
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        
        print("Starting document processing...")
        with tqdm(total=total_docs, desc="Processing documents") as pbar:
            
            while hits:
                for hit in hits:
                    try:
                        doc = hit['_source']
                        text = doc.get('text', '')
                        
                        if not text:
                            continue
                        
                        # Get AraBERT embedding
                        embedding = arabert_engine.get_embedding(text)
                        
                        # Add processed text
                        processed_text = arabert_engine.preprocess_arabic_text(text)
                        
                        # Create new document with embedding
                        new_doc = {
                            'text': text,
                            'processed_text': processed_text,
                            'text_embedding': embedding.tolist(),
                            'start': doc.get('start', 0),
                            'end': doc.get('end', 0),
                            'video_link': doc.get('video_link', '')
                        }
                        
                        batch_docs.append({
                            'index': {
                                '_index': new_index_name,
                                '_id': hit['_id']
                            }
                        })
                        batch_docs.append(new_doc)
                        
                        processed_count += 1
                        pbar.update(1)
                        
                        # Bulk index every batch_size documents
                        if len(batch_docs) >= (batch_size * 2):  # batch_size * 2 (index + doc)
                            try:
                                bulk_response = es.bulk(body=batch_docs)
                                if bulk_response['errors']:
                                    print(f"Bulk indexing errors: {bulk_response['errors']}")
                                batch_docs = []
                                time.sleep(0.1)  # Small delay to avoid overwhelming ES
                            except Exception as e:
                                print(f"Error in bulk indexing: {e}")
                                batch_docs = []
                    
                    except Exception as e:
                        print(f"Error processing document {hit.get('_id', 'unknown')}: {e}")
                        continue
                
                # Get next batch
                try:
                    response = es.scroll(scroll_id=scroll_id, scroll='10m')
                    hits = response['hits']['hits']
                except Exception as e:
                    print(f"Error scrolling: {e}")
                    break
        
        # Index remaining documents
        if batch_docs:
            try:
                bulk_response = es.bulk(body=batch_docs)
                if bulk_response['errors']:
                    print(f"Final bulk indexing errors: {bulk_response['errors']}")
            except Exception as e:
                print(f"Error in final bulk indexing: {e}")
    
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Clear scroll
        try:
            if 'scroll_id' in locals():
                es.clear_scroll(scroll_id=scroll_id)
        except:
            pass
    
    print(f"Finished processing {processed_count} documents")
    
    # Verify the new index
    try:
        time.sleep(2)  # Wait for indexing to complete
        es.indices.refresh(index=new_index_name)
        final_count = es.count(index=new_index_name)['count']
        print(f"New index '{new_index_name}' contains {final_count} documents")
        
        if final_count > 0:
            print("Successfully created enhanced index with AraBERT embeddings!")
            print(f"You can now use index '{new_index_name}' for enhanced search.")
        else:
            print("Warning: No documents were indexed successfully.")
            
    except Exception as e:
        print(f"Error verifying new index: {e}")

def test_enhanced_search():
    """Test the enhanced search functionality"""
    print("\n" + "="*50)
    print("Testing Enhanced Search")
    print("="*50)
    
    es = Elasticsearch('http://localhost:9200', basic_auth=('elastic', 'Fe1odvmZ'))
    arabert_engine = AraBERTSearchEngine()
    
    # Test queries
    test_queries = ["صلاة", "صوم", "حج", "زكاة"]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        
        try:
            # Test basic search
            basic_query = {"match": {"text": query}}
            basic_response = es.search(index='transcription', query=basic_query, size=3)
            print(f"Basic search results: {basic_response['hits']['total']['value']}")
            
            # Test enhanced search
            enhanced_query = arabert_engine.create_enhanced_query(query)
            enhanced_response = es.search(index='transcription_with_embeddings', query=enhanced_query, size=3)
            print(f"Enhanced search results: {enhanced_response['hits']['total']['value']}")
            
            # Test with embeddings if available
            if arabert_engine.model is not None:
                embedding = arabert_engine.get_embedding(query)
                hybrid_query = arabert_engine.create_hybrid_query(query, embedding)
                hybrid_response = es.search(index='transcription_with_embeddings', query=hybrid_query, size=3)
                print(f"Hybrid search results: {hybrid_response['hits']['total']['value']}")
            
        except Exception as e:
            print(f"Error testing query '{query}': {e}")

if __name__ == "__main__":
    print("AraBERT Elasticsearch Integration")
    print("="*40)
    
    choice = input("Choose action:\n1. Add embeddings to existing data\n2. Test enhanced search\n3. Both\nEnter choice (1-3): ")
    
    if choice in ['1', '3']:
        add_embeddings_to_index()
    
    if choice in ['2', '3']:
        test_enhanced_search()
    
    print("\nDone!")