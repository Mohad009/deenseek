from transformers import AutoTokenizer, AutoModel
import torch
import re
import numpy as np
from typing import List, Dict

class AraBERTSearchEngine:
    def __init__(self):
        """Initialize AraBERT model for Arabic text processing and embeddings"""
        self.model_name = "aubmindlab/bert-base-arabertv2"
        print("Loading AraBERT model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            print("AraBERT model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load AraBERT model: {e}")
            print("Fallback: Using basic text processing only")
            self.tokenizer = None
            self.model = None
        
        # Arabic synonyms dictionary for query expansion
        self.synonyms = {
            'صلاة': ['صلوات', 'صلاه', 'الصلاة', 'فريضة'],
            'سفر': ['سفار', 'رحلة', 'سفره', 'مسافر'],
            'حج': ['حجة', 'حجج', 'الحج', 'حاج'],
            'صوم': ['صيام', 'صائم', 'الصوم', 'رمضان'],
            'زكاة': ['زكوات', 'الزكاة', 'صدقة'],
            'وضوء': ['طهارة', 'تطهر', 'الوضوء'],
            'قرآن': ['القرآن', 'كتاب الله', 'المصحف'],
            'سنة': ['سنن', 'حديث', 'نبوي'],
            'دعاء': ['أدعية', 'ذكر', 'دعوة'],
            'جنازة': ['موت', 'وفاة', 'دفن'],
            'نكاح': ['زواج', 'عقد', 'خطبة'],
            'طلاق': ['فسخ', 'انفصال', 'خلع'],
            'بيع': ['شراء', 'تجارة', 'معاملة'],
            'ربا': ['فائدة', 'ريبة', 'حرام'],
            'جهاد': ['قتال', 'غزو', 'مجاهد'],
            'علم': ['تعلم', 'فقه', 'معرفة'],
            'ذنب': ['معصية', 'خطأ', 'إثم'],
            'توبة': ['استغفار', 'ندم', 'رجوع'],
            'جنة': ['فردوس', 'نعيم', 'خلود'],
            'نار': ['عذاب', 'جهنم', 'عقاب']
        }
    
    def preprocess_arabic_text(self, text: str) -> str:
        """Clean and normalize Arabic text"""
        if not text:
            return ""
        
        # Remove diacritics (تشكيل)
        text = re.sub(r'[\u064B-\u0652]', '', text)
        
        # Normalize Arabic letters
        text = re.sub(r'[أإآ]', 'ا', text)  # Normalize alef variants
        text = re.sub(r'ة', 'ه', text)      # Normalize teh marbuta
        text = re.sub(r'ي', 'ى', text)      # Normalize yeh variants
        
        # Remove extra whitespace and punctuation
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)  # Keep only Arabic and spaces
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text.strip()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get AraBERT embedding for text"""
        if not self.model or not self.tokenizer:
            # Fallback: return zero vector if model not available
            return np.zeros(768)
        
        text = self.preprocess_arabic_text(text)
        if not text:
            return np.zeros(768)
        
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            return embeddings.numpy().flatten()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(768)
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        query = self.preprocess_arabic_text(query)
        expanded_terms = [query]
        
        # Split query into words and find synonyms
        words = query.split()
        for word in words:
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)
        
        return unique_terms
    
    def create_enhanced_query(self, search_term: str) -> Dict:
        """Create enhanced Elasticsearch query with multiple strategies"""
        expanded_terms = self.expand_query(search_term)
        
        should_queries = []
        
        # 1. Exact phrase match (highest boost)
        should_queries.append({
            "match_phrase": {
                "text": {
                    "query": search_term,
                    "boost": 5.0
                }
            }
        })
        
        # 2. Match with original term
        should_queries.append({
            "match": {
                "text": {
                    "query": search_term,
                    "boost": 3.0,
                    "operator": "and"
                }
            }
        })
        
        # 3. Fuzzy match for typos
        should_queries.append({
            "fuzzy": {
                "text": {
                    "value": search_term,
                    "fuzziness": "AUTO",
                    "boost": 2.0
                }
            }
        })
        
        # 4. Match with synonyms
        for i, term in enumerate(expanded_terms[1:], 1):  # Skip original term
            boost_value = 2.5 if i <= 3 else 1.5  # Higher boost for first few synonyms
            should_queries.append({
                "match": {
                    "text": {
                        "query": term,
                        "boost": boost_value
                    }
                }
            })
        
        # 5. Cross-field search if processed_text exists
        should_queries.append({
            "multi_match": {
                "query": search_term,
                "fields": ["text^3", "processed_text^2"],
                "type": "best_fields",
                "boost": 2.8
            }
        })
        
        # 6. Wildcard search for partial matches
        should_queries.append({
            "wildcard": {
                "text": {
                    "value": f"*{search_term}*",
                    "boost": 1.2
                }
            }
        })
        
        # 7. Boolean query with individual words
        if len(search_term.split()) > 1:
            should_queries.append({
                "bool": {
                    "must": [
                        {"match": {"text": word}} 
                        for word in search_term.split()
                    ],
                    "boost": 2.2
                }
            })
        
        return {
            "bool": {
                "should": should_queries,
                "minimum_should_match": 1
            }
        }
    
    def create_semantic_query(self, search_term: str, query_embedding: np.ndarray) -> Dict:
        """Create semantic search query using dense vectors"""
        return {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'text_embedding') + 1.0",
                    "params": {"query_vector": query_embedding.tolist()}
                }
            }
        }
    
    def create_hybrid_query(self, search_term: str, query_embedding: np.ndarray = None) -> Dict:
        """Create hybrid query combining semantic and lexical search"""
        lexical_query = self.create_enhanced_query(search_term)
        
        if query_embedding is None or not hasattr(query_embedding, 'tolist'):
            # If no embeddings available, use only lexical search
            return lexical_query
        
        # Combine semantic and lexical search
        return {
            "bool": {
                "should": [
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'text_embedding') + 1.0",
                                "params": {"query_vector": query_embedding.tolist()}
                            },
                            "boost": 3.0
                        }
                    },
                    lexical_query
                ],
                "minimum_should_match": 1
            }
        }