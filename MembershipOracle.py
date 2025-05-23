import re
from abc import ABC, abstractmethod


class MembershipOracle(ABC):
    """
    Abstract base class for membership oracles.
    A membership oracle determines whether a given string belongs to a target language.
    """
    
    @abstractmethod
    def query(self, word: str) -> bool:
        """
        Return True iff `word` is in the target language.
        
        Args:
            word: The string to query
            
        Returns:
            bool: True if the word is accepted, False otherwise
        """
        pass


class RegexMembershipOracle(MembershipOracle):
    """
    A membership oracle that uses regular expressions to define the target language.
    """
    
    def __init__(self, pattern: str):
        """
        Initialize with a regex pattern.
        
        Args:
            pattern: Regular expression pattern defining the target language
        """
        self._re = re.compile(pattern)
        self.pattern = pattern

    def query(self, word: str) -> bool:
        """Return True iff `word` matches the regex pattern."""
        return bool(self._re.fullmatch(word))
    
    def get_pattern(self) -> str:
        """Return the regex pattern."""
        return self.pattern 