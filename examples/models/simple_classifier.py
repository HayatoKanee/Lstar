#!/usr/bin/env python3
"""
Example custom classifiers for use with lstar_extract.py

These functions can be used with the command-line tool:
    python lstar_extract.py --model-path examples/simple_classifier.py --model-type function --function-name palindrome_classifier --alphabet "abc" --output palindrome_dfa.png
"""


def palindrome_classifier(string: str) -> float:
    """
    Classifier that recognizes palindromes.
    
    Args:
        string: Input string to classify
        
    Returns:
        float: Probability between 0 and 1
    """
    if len(string) <= 1:
        return 0.9  # Empty string and single chars are palindromes
    
    is_palindrome = string == string[::-1]
    return 0.85 if is_palindrome else 0.15


def starts_with_a(string: str) -> float:
    """
    Classifier that accepts strings starting with 'a'.
    
    Args:
        string: Input string to classify
        
    Returns:
        float: Probability between 0 and 1
    """
    if len(string) == 0:
        return 0.1  # Empty string doesn't start with 'a'
    
    starts_a = string[0].lower() == 'a'
    return 0.9 if starts_a else 0.1


def even_length_classifier(string: str) -> float:
    """
    Classifier that accepts strings of even length.
    
    Args:
        string: Input string to classify
        
    Returns:
        float: Probability between 0 and 1
    """
    is_even = len(string) % 2 == 0
    return 0.9 if is_even else 0.1


def contains_substring_classifier(string: str) -> float:
    """
    Classifier that accepts strings containing "ab".
    
    Args:
        string: Input string to classify
        
    Returns:
        float: Probability between 0 and 1
    """
    contains_ab = "ab" in string
    return 0.85 if contains_ab else 0.15


def balanced_parentheses_classifier(string: str) -> float:
    """
    Classifier for balanced parentheses (simplified version).
    
    Args:
        string: Input string to classify
        
    Returns:
        float: Probability between 0 and 1
    """
    if not string:
        return 0.9  # Empty string is balanced
    
    balance = 0
    for char in string:
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
            if balance < 0:
                return 0.1  # More closing than opening
    
    is_balanced = balance == 0
    return 0.9 if is_balanced else 0.1


def binary_divisible_by_3(string: str) -> float:
    """
    Classifier that accepts binary strings representing numbers divisible by 3.
    
    Args:
        string: Binary string to classify
        
    Returns:
        float: Probability between 0 and 1
    """
    if not string or not all(c in '01' for c in string):
        return 0.0  # Invalid binary string
    
    try:
        number = int(string, 2)
        is_divisible = number % 3 == 0
        return 0.9 if is_divisible else 0.1
    except ValueError:
        return 0.0


def vowel_count_classifier(string: str) -> float:
    """
    Classifier that accepts strings with at least 2 vowels.
    
    Args:
        string: Input string to classify
        
    Returns:
        float: Probability between 0 and 1
    """
    vowels = set('aeiouAEIOU')
    vowel_count = sum(1 for char in string if char in vowels)
    
    has_enough_vowels = vowel_count >= 2
    return 0.8 if has_enough_vowels else 0.2


def alternating_pattern_classifier(string: str) -> float:
    """
    Classifier that accepts strings with alternating 'a' and 'b' pattern.
    
    Args:
        string: Input string to classify
        
    Returns:
        float: Probability between 0 and 1
    """
    if len(string) <= 1:
        return 0.9 if string in ['', 'a', 'b'] else 0.1
    
    # Check if it follows 'ab' or 'ba' alternating pattern
    pattern1 = all(string[i] == 'ab'[i % 2] for i in range(len(string)))
    pattern2 = all(string[i] == 'ba'[i % 2] for i in range(len(string)))
    
    is_alternating = pattern1 or pattern2
    return 0.85 if is_alternating else 0.15 