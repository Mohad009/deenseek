#!/usr/bin/env python3
"""
Setup script for AraBERT-Enhanced Elasticsearch Search
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Successfully installed all requirements!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_elasticsearch():
    """Check if Elasticsearch is running"""
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch('http://localhost:9200', basic_auth=('elastic', 'Fe1odvmZ'))
        info = es.info()
        print(f"✅ Elasticsearch is running: {info['version']['number']}")
        return True
    except Exception as e:
        print(f"❌ Cannot connect to Elasticsearch: {e}")
        print("Make sure Elasticsearch is running on localhost:9200")
        return False

def check_index():
    """Check if transcription index exists"""
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch('http://localhost:9200', basic_auth=('elastic', 'Fe1odvmZ'))
        
        if es.indices.exists(index='transcription'):
            count = es.count(index='transcription')['count']
            print(f"✅ Transcription index found with {count} documents")
            return True
        else:
            print("❌ Transcription index not found")
            print("Make sure your data is indexed in 'transcription' index")
            return False
    except Exception as e:
        print(f"❌ Error checking index: {e}")
        return False

def test_arabert():
    """Test AraBERT loading"""
    try:
        from arabert_search import AraBERTSearchEngine
        engine = AraBERTSearchEngine()
        print("✅ AraBERT loaded successfully!")
        
        # Test embedding generation
        test_text = "صلاة الفجر"
        embedding = engine.get_embedding(test_text)
        print(f"✅ Generated embedding for '{test_text}': shape {embedding.shape}")
        return True
    except Exception as e:
        print(f"❌ Error loading AraBERT: {e}")
        print("AraBERT will be disabled, but basic search will still work")
        return False

def main():
    print("=" * 60)
    print("AraBERT-Enhanced Elasticsearch Search Setup")
    print("=" * 60)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("Setup failed at requirements installation")
        return False
    
    print("\n" + "-" * 40)
    
    # Step 2: Check Elasticsearch
    if not check_elasticsearch():
        print("Setup failed: Elasticsearch not available")
        return False
    
    print("\n" + "-" * 40)
    
    # Step 3: Check if data index exists
    if not check_index():
        print("Warning: No transcription data found")
        print("You'll need to index your data first")
    
    print("\n" + "-" * 40)
    
    # Step 4: Test AraBERT
    arabert_works = test_arabert()
    
    print("\n" + "=" * 60)
    print("Setup Summary:")
    print("=" * 60)
    print("✅ Requirements installed")
    print("✅ Elasticsearch connected")
    if arabert_works:
        print("✅ AraBERT loaded successfully")
        print("\nNext steps:")
        print("1. Run 'python add_embeddings.py' to add embeddings to your data")
        print("2. Run 'python app.py' to start the search application")
    else:
        print("⚠️  AraBERT not available (will use basic search)")
        print("\nNext steps:")
        print("1. Run 'python app.py' to start the search application")
    
    print("\nAccess the search interface at: http://localhost:5000")
    return True

if __name__ == "__main__":
    main()