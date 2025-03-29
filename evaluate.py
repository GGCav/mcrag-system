import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

class MCRAGEvaluator:
    def __init__(self, results_dir, ground_truth_file=None):
        self.results_dir = results_dir
        self.ground_truth = None
        
        if ground_truth_file:
            self.load_ground_truth(ground_truth_file)
    
    def load_ground_truth(self, file_path):
        """Load ground truth data for evaluation"""
        self.ground_truth = pd.read_csv(file_path)
    
    def evaluate_relevance(self, results_file):
        """Evaluate relevance of retrieved chunks"""
        with open(results_file, 'r') as f:
            result = json.load(f)
        
        query = result["query"]
        retrieved_chunks = result["retrieved_chunks"]
        
        # In a real implementation, this would compare against ground truth
        # For demo purposes, we'll use a simple keyword matching approach
        query_keywords = self._extract_keywords(query)
        
        relevance_scores = []
        for chunk in retrieved_chunks:
            chunk_text = chunk["content"]
            chunk_keywords = self._extract_keywords(chunk_text)
            
            # Calculate overlap between query and chunk keywords
            overlap = len(set(query_keywords) & set(chunk_keywords))
            relevance = overlap / max(len(query_keywords), 1)
            
            relevance_scores.append(relevance)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        return avg_relevance
    
    def _extract_keywords(self, text):
        """Extract relevant keywords from text (simplified)"""
        # In production, use a better keyword extraction approach
        text = text.lower()
        # Remove punctuation
        for char in ".,;:!?()[]{}\"'":
            text = text.replace(char, " ")
        
        words = text.split()
        # Remove stopwords (simplified)
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with"}
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords
    
    def compare_retrieval_modes(self, query, system, modes=["random", "top", "diversity", "class"], k=5):
        """Compare different retrieval modes for the same query"""
        results = {}
        
        for mode in modes:
            result = system.process_query(query, mode=mode, k=k)
            avg_relevance = self.evaluate_relevance(result)
            results[mode] = {
                "relevance": avg_relevance,
                "recommendation": result["recommendation"]
            }
        
        return results
    
    def plot_mode_comparison(self, comparison_results):
        """Plot comparison of retrieval modes"""
        modes = list(comparison_results.keys())
        relevance_scores = [comparison_results[mode]["relevance"] for mode in modes]
        
        plt.figure(figsize=(10, 6))
        plt.bar(modes, relevance_scores)
        plt.ylabel("Average Relevance Score")
        plt.xlabel("Retrieval Mode")
        plt.title("Comparison of Retrieval Modes")
        plt.ylim(0, 1)
        
        # Add values on top of bars
        for i, score in enumerate(relevance_scores):
            plt.text(i, score + 0.02, f"{score:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig("mode_comparison.png")
        plt.close()

# Example usage
if __name__ == "__main__":
    evaluator = MCRAGEvaluator("./results")
    # Add example evaluation code here