from typing import List
from .MembershipOracle import MembershipOracle
from rnn_adapters import GenericRNNInterface


class GenericRNNMembershipOracle(MembershipOracle):
    """
    A membership oracle that can work with any RNN implementing GenericRNNInterface.
    This provides a unified interface for the L* algorithm to work with any type of RNN.
    """
    
    def __init__(self, rnn: GenericRNNInterface):
        """
        Initialize with any RNN implementing the GenericRNNInterface.
        
        Args:
            rnn: Any RNN implementing GenericRNNInterface
        """
        self.rnn = rnn
        self.query_count = 0
    
    def query(self, word: str) -> bool:
        """
        Query the RNN for membership decision.
        
        Args:
            word: String to query
            
        Returns:
            bool: True if the RNN accepts the word, False otherwise
        """
        self.query_count += 1
        return self.rnn.query(word)
    
    def get_query_count(self) -> int:
        """Get the total number of queries made."""
        return self.query_count
    
    def reset_query_count(self):
        """Reset the query counter."""
        self.query_count = 0
    
    def get_confidence(self, word: str) -> float:
        """
        Get the RNN's confidence score for a word.
        
        Args:
            word: String to query
            
        Returns:
            float: Confidence score between 0 and 1
        """
        return self.rnn.get_confidence(word)
    
    def get_alphabet(self) -> List[str]:
        """
        Get the alphabet that this RNN can handle.
        
        Returns:
            List[str]: List of characters/tokens
        """
        return self.rnn.get_alphabet()


class AnalyzingGenericRNNMembershipOracle(GenericRNNMembershipOracle):
    """
    Extended version with detailed logging and analysis capabilities.
    """
    
    def __init__(self, rnn: GenericRNNInterface):
        super().__init__(rnn)
        self.query_log = []
    
    def query(self, word: str) -> bool:
        """Query with detailed logging."""
        result = super().query(word)
        confidence = self.get_confidence(word)
        
        # Log the query with metadata
        self.query_log.append({
            'word': word,
            'result': result,
            'confidence': confidence,
            'query_number': self.query_count,
            'rnn_type': type(self.rnn).__name__
        })
        
        return result
    
    def get_query_log(self) -> List[dict]:
        """Return the complete query log."""
        return self.query_log.copy()
    
    def print_query_statistics(self):
        """Print detailed statistics about the queries."""
        if not self.query_log:
            print("No queries have been made yet.")
            return
        
        total_queries = len(self.query_log)
        accepted = sum(1 for q in self.query_log if q['result'])
        rejected = total_queries - accepted
        
        avg_confidence = sum(q['confidence'] for q in self.query_log) / total_queries
        min_confidence = min(q['confidence'] for q in self.query_log)
        max_confidence = max(q['confidence'] for q in self.query_log)
        
        print(f"\n=== Query Statistics ===")
        print(f"RNN Type: {self.query_log[0]['rnn_type']}")
        print(f"Total queries: {total_queries}")
        print(f"Accepted: {accepted} ({accepted/total_queries*100:.1f}%)")
        print(f"Rejected: {rejected} ({rejected/total_queries*100:.1f}%)")
        print(f"Confidence - Avg: {avg_confidence:.3f}, Min: {min_confidence:.3f}, Max: {max_confidence:.3f}")
        
        # Show distribution of confidence scores
        high_conf = sum(1 for q in self.query_log if q['confidence'] > 0.8)
        medium_conf = sum(1 for q in self.query_log if 0.2 <= q['confidence'] <= 0.8)
        low_conf = sum(1 for q in self.query_log if q['confidence'] < 0.2)
        
        print(f"Confidence distribution:")
        print(f"  High (>0.8): {high_conf} ({high_conf/total_queries*100:.1f}%)")
        print(f"  Medium (0.2-0.8): {medium_conf} ({medium_conf/total_queries*100:.1f}%)")
        print(f"  Low (<0.2): {low_conf} ({low_conf/total_queries*100:.1f}%)")
        
        # Show example queries
        print(f"\nExample queries:")
        for i, q in enumerate(self.query_log[:10]):
            print(f"  '{q['word']}' -> {q['result']} (conf: {q['confidence']:.3f})")
        
        if len(self.query_log) > 10:
            print(f"  ... and {len(self.query_log) - 10} more")
    
    def get_uncertainty_strings(self, threshold: float = 0.1) -> List[str]:
        """
        Get strings where the RNN is uncertain (confidence close to 0.5).
        
        Args:
            threshold: How close to 0.5 to consider uncertain
            
        Returns:
            List of uncertain strings
        """
        uncertain = []
        for q in self.query_log:
            if abs(q['confidence'] - 0.5) <= threshold:
                uncertain.append(q['word'])
        return uncertain
    
    def analyze_patterns(self):
        """Analyze patterns in the queries and results."""
        if not self.query_log:
            return
        
        print(f"\n=== Pattern Analysis ===")
        
        # Analyze by string length
        length_stats = {}
        for q in self.query_log:
            length = len(q['word'])
            if length not in length_stats:
                length_stats[length] = {'total': 0, 'accepted': 0, 'confidences': []}
            
            length_stats[length]['total'] += 1
            if q['result']:
                length_stats[length]['accepted'] += 1
            length_stats[length]['confidences'].append(q['confidence'])
        
        print("Acceptance by string length:")
        for length in sorted(length_stats.keys()):
            stats = length_stats[length]
            acc_rate = stats['accepted'] / stats['total'] * 100
            avg_conf = sum(stats['confidences']) / len(stats['confidences'])
            print(f"  Length {length}: {stats['accepted']}/{stats['total']} ({acc_rate:.1f}%) accepted, avg confidence: {avg_conf:.3f}")
        
        # Find uncertain strings
        uncertain = self.get_uncertainty_strings()
        if uncertain:
            print(f"\nUncertain strings (confidence ~0.5): {uncertain[:5]}")
            if len(uncertain) > 5:
                print(f"  ... and {len(uncertain) - 5} more") 