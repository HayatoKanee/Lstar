from DummyRNN import DummyRNN
from MembershipOracle import MembershipOracle
from typing import List


class RNNMembershipOracle(MembershipOracle):
    """
    A membership oracle that queries a trained RNN instead of using regex patterns.
    This enables the L* algorithm to learn from neural network models.
    """
    
    def __init__(self, rnn: DummyRNN):
        """
        Initialize with a pre-trained RNN.
        
        Args:
            rnn: A trained DummyRNN instance
        """
        self.rnn = rnn
        self.query_count = 0  # Track number of queries for analysis
    
    def query(self, word: str) -> bool:
        """
        Return True iff `word` is accepted by the RNN.
        
        Args:
            word: The string to query
            
        Returns:
            bool: True if the RNN accepts the word, False otherwise
        """
        self.query_count += 1
        return self.rnn.query(word)
    
    def get_query_count(self) -> int:
        """Return the total number of queries made to the RNN."""
        return self.query_count
    
    def reset_query_count(self):
        """Reset the query counter."""
        self.query_count = 0
    
    def get_confidence(self, word: str) -> float:
        """
        Get the RNN's confidence score for the given word.
        
        Args:
            word: The string to query
            
        Returns:
            float: Confidence score between 0 and 1
        """
        return self.rnn.get_confidence(word)


class AnalyzingRNNMembershipOracle(RNNMembershipOracle):
    """
    Extended RNN membership oracle that provides additional analysis capabilities.
    """
    
    def __init__(self, rnn: DummyRNN):
        super().__init__(rnn)
        self.query_log = []  # Log all queries for analysis
    
    def query(self, word: str) -> bool:
        """Query with logging for analysis."""
        result = super().query(word)
        confidence = self.get_confidence(word)
        
        # Log the query
        self.query_log.append({
            'word': word,
            'result': result,
            'confidence': confidence,
            'query_number': self.query_count
        })
        
        return result
    
    def get_query_log(self) -> List[dict]:
        """Return the complete query log."""
        return self.query_log.copy()
    
    def print_query_statistics(self):
        """Print statistics about the queries made."""
        if not self.query_log:
            print("No queries have been made yet.")
            return
        
        total_queries = len(self.query_log)
        accepted = sum(1 for q in self.query_log if q['result'])
        rejected = total_queries - accepted
        
        avg_confidence = sum(q['confidence'] for q in self.query_log) / total_queries
        
        print(f"\n=== Query Statistics ===")
        print(f"Total queries: {total_queries}")
        print(f"Accepted: {accepted} ({accepted/total_queries*100:.1f}%)")
        print(f"Rejected: {rejected} ({rejected/total_queries*100:.1f}%)")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # Show some example queries
        print(f"\nExample queries:")
        for i, q in enumerate(self.query_log[:10]):  # Show first 10
            print(f"  '{q['word']}' -> {q['result']} (conf: {q['confidence']:.3f})")
        
        if len(self.query_log) > 10:
            print(f"  ... and {len(self.query_log) - 10} more") 