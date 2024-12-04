# src/utils/metrics.py
import re
import string
import collections

def normalize_text(text):
    """Remove articles and punctuation, and normalize whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(text))))

def compute_exact_match(prediction, reference):
    return normalize_text(prediction) == normalize_text(reference)

def compute_f1(prediction, reference):
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    
    if not prediction_tokens or not reference_tokens:
        return int(prediction_tokens == reference_tokens)
    
    common = collections.Counter(prediction_tokens) & collections.Counter(reference_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1