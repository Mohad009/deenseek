#!/usr/bin/env python3
"""
Arabic Transcription Quality Checker using AraBERT
For Su'al Ahl al-Dhikr project
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ArabicTranscriptionQualityChecker:
    def __init__(self, model_name='aubmindlab/bert-base-arabertv2'):
        """Initialize with AraBERT model"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            print(f"✓ Loaded AraBERT model: {model_name}")
            
        except ImportError:
            print("❌ Required packages not found. Install with:")
            print("pip install transformers torch sentence-transformers")
            raise
    
    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """Get AraBERT embedding for a sentence"""
        import torch
        
        # Clean and prepare text
        text = self.clean_arabic_text(text)
        if not text.strip():
            return np.zeros(768)  # AraBERT hidden size
        
        # Tokenize and encode
        inputs = self.tokenizer(text, return_tensors='pt', 
                               truncation=True, max_length=512, 
                               padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.flatten()
    
    def clean_arabic_text(self, text: str) -> str:
        """Clean Arabic text for better processing"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-Arabic characters (keep basic punctuation)
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\s\.\?\!\,\:\;]', ' ', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\?\!]{2,}', '?', text)
        
        return text.strip()
    
    def calculate_coherence_score(self, segments: List[str]) -> List[float]:
        """Calculate coherence between consecutive segments"""
        if len(segments) < 2:
            return [1.0] * len(segments)
        
        embeddings = [self.get_sentence_embedding(seg) for seg in segments]
        coherence_scores = []
        
        for i in range(len(embeddings)):
            if i == 0:
                # First segment - compare with next
                similarity = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                coherence_scores.append(similarity)
            elif i == len(embeddings) - 1:
                # Last segment - compare with previous
                similarity = cosine_similarity([embeddings[i]], [embeddings[i-1]])[0][0]
                coherence_scores.append(similarity)
            else:
                # Middle segments - average similarity with neighbors
                sim_prev = cosine_similarity([embeddings[i]], [embeddings[i-1]])[0][0]
                sim_next = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                coherence_scores.append((sim_prev + sim_next) / 2)
        
        return coherence_scores
    
    def detect_quality_issues(self, transcript_data: Dict) -> Dict:
        """Detect quality issues in transcription"""
        issues = {
            'low_coherence_segments': [],
            'short_segments': [],
            'repetitive_segments': [],
            'unusual_characters': [],
            'statistics': {}
        }
        
        segments = transcript_data.get('segments', [])
        if not segments:
            return issues
        
        # Extract text segments
        texts = [seg.get('text', '').strip() for seg in segments]
        
        # 1. Coherence analysis
        print("   Analyzing coherence...")
        coherence_scores = self.calculate_coherence_score(texts)
        
        # Flag low coherence (bottom 10%)
        coherence_threshold = np.percentile(coherence_scores, 10)
        for i, score in enumerate(coherence_scores):
            if score < coherence_threshold and score < 0.5:  # Also absolute threshold
                issues['low_coherence_segments'].append({
                    'segment_id': i,
                    'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                    'coherence_score': float(score),
                    'timestamp': f"{segments[i].get('start', 0):.1f}s"
                })
        
        # 2. Short segments (likely incomplete)
        for i, text in enumerate(texts):
            if len(text.split()) < 3 and len(text) > 0:  # Very short but not empty
                issues['short_segments'].append({
                    'segment_id': i,
                    'text': text,
                    'word_count': len(text.split()),
                    'timestamp': f"{segments[i].get('start', 0):.1f}s"
                })
        
        # 3. Repetitive segments
        for i in range(len(texts) - 1):
            similarity = cosine_similarity(
                [self.get_sentence_embedding(texts[i])],
                [self.get_sentence_embedding(texts[i + 1])]
            )[0][0]
            
            if similarity > 0.95 and len(texts[i]) > 10:  # Very similar and not too short
                issues['repetitive_segments'].append({
                    'segment_ids': [i, i + 1],
                    'similarity': float(similarity),
                    'text1': texts[i][:50] + '...',
                    'text2': texts[i + 1][:50] + '...'
                })
        
        # 4. Unusual characters
        for i, text in enumerate(texts):
            unusual_chars = re.findall(r'[^\u0600-\u06FF\u0750-\u077F\s\.\?\!\,\:\;\(\)]', text)
            if unusual_chars:
                issues['unusual_characters'].append({
                    'segment_id': i,
                    'characters': list(set(unusual_chars)),
                    'text_sample': text[:100] + '...'
                })
        
        # Statistics
        issues['statistics'] = {
            'total_segments': len(segments),
            'avg_coherence': float(np.mean(coherence_scores)),
            'min_coherence': float(np.min(coherence_scores)),
            'avg_segment_length': float(np.mean([len(t.split()) for t in texts if t])),
            'empty_segments': sum(1 for t in texts if not t.strip())
        }
        
        return issues
    
    def process_transcription_file(self, json_file: Path) -> Dict:
        """Process a single transcription JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            print(f"🔍 Analyzing: {json_file.name}")
            
            # Detect issues
            quality_report = self.detect_quality_issues(transcript_data)
            
            # Add metadata
            quality_report['file_info'] = {
                'filename': json_file.name,
                'duration': transcript_data.get('duration', 0),
                'language': transcript_data.get('language', 'unknown'),
                'full_text_length': len(transcript_data.get('text', ''))
            }
            
            return quality_report
            
        except Exception as e:
            print(f"❌ Error processing {json_file}: {e}")
            return {}
    
    def generate_quality_report(self, transcription_folder: str, output_file: str = "quality_report.json"):
        """Generate quality report for all transcriptions"""
        folder = Path(transcription_folder)
        json_files = list(folder.rglob("*_transcript.json"))
        
        if not json_files:
            print(f"❌ No transcript JSON files found in {folder}")
            return
        
        print(f"📊 Processing {len(json_files)} transcription files...")
        
        all_reports = {}
        summary_stats = {
            'total_files': len(json_files),
            'processed_files': 0,
            'total_issues': 0,
            'avg_coherence': 0,
            'files_with_issues': 0
        }
        
        for json_file in json_files:
            video_id = json_file.stem.replace('_transcript', '')
            report = self.process_transcription_file(json_file)
            
            if report:
                all_reports[video_id] = report
                summary_stats['processed_files'] += 1
                
                # Count issues
                issue_count = (len(report.get('low_coherence_segments', [])) +
                              len(report.get('short_segments', [])) +
                              len(report.get('repetitive_segments', [])) +
                              len(report.get('unusual_characters', [])))
                
                summary_stats['total_issues'] += issue_count
                if issue_count > 0:
                    summary_stats['files_with_issues'] += 1
                
                summary_stats['avg_coherence'] += report.get('statistics', {}).get('avg_coherence', 0)
        
        # Finalize summary
        if summary_stats['processed_files'] > 0:
            summary_stats['avg_coherence'] /= summary_stats['processed_files']
        
        # Save report
        final_report = {
            'summary': summary_stats,
            'detailed_reports': all_reports,
            'recommendations': self.generate_recommendations(all_reports)
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Quality report saved: {output_path.absolute()}")
        self.print_summary(summary_stats)
        
        return final_report
    
    def generate_recommendations(self, reports: Dict) -> List[str]:
        """Generate recommendations based on quality analysis"""
        recommendations = []
        
        total_files = len(reports)
        if total_files == 0:
            return recommendations
        
        # Calculate overall statistics
        files_with_low_coherence = sum(1 for r in reports.values() 
                                     if len(r.get('low_coherence_segments', [])) > 0)
        files_with_repetition = sum(1 for r in reports.values() 
                                  if len(r.get('repetitive_segments', [])) > 0)
        
        avg_coherence = np.mean([r.get('statistics', {}).get('avg_coherence', 0) 
                                for r in reports.values()])
        
        if files_with_low_coherence > total_files * 0.3:
            recommendations.append("⚠️ Over 30% of files have low coherence segments. Consider manual review of flagged segments.")
        
        if files_with_repetition > total_files * 0.2:
            recommendations.append("🔄 Many files have repetitive segments. Review transcription settings or audio quality.")
        
        if avg_coherence < 0.6:
            recommendations.append("📉 Overall coherence is low. Consider using a better transcription model or preprocessing audio.")
        
        recommendations.append("✅ Focus manual review on segments flagged with low coherence scores.")
        recommendations.append("🎯 Prioritize files with multiple issue types for manual checking.")
        
        return recommendations
    
    def print_summary(self, stats: Dict):
        """Print summary statistics"""
        print("\n" + "=" * 60)
        print("📊 TRANSCRIPTION QUALITY SUMMARY")
        print("=" * 60)
        print(f"📁 Total files processed: {stats['processed_files']}/{stats['total_files']}")
        print(f"🚨 Files with issues: {stats['files_with_issues']}")
        print(f"📈 Average coherence: {stats['avg_coherence']:.3f}")
        print(f"⚠️ Total issues found: {stats['total_issues']}")
        print("=" * 60)

def main():
    # Initialize quality checker
    checker = ArabicTranscriptionQualityChecker()
    
    # Process transcriptions
    transcription_folder = input("Enter transcription folder path: ").strip()
    output_file = input("Enter output report file (default: quality_report.json): ").strip()
    
    if not output_file:
        output_file = "quality_report.json"
    
    # Generate quality report
    report = checker.generate_quality_report(transcription_folder, output_file)
    
    print(f"\n🎉 Quality analysis complete!")
    print(f"📄 Review the detailed report in: {output_file}")

if __name__ == "__main__":
    main()