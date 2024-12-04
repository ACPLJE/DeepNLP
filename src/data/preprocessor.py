# src/data/preprocessor.py
import torch 
from transformers import PreTrainedTokenizer
from typing import Dict, Any, List, Union

class Preprocessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: Dict[str, Any]):
        """
        Preprocessor for question answering tasks
        
        Args:
            tokenizer: Tokenizer for text processing
            config: Configuration dictionary containing preprocessing settings
        """
        self.tokenizer = tokenizer
        self.config = config
        
    def preprocess_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[Any]]:
        """
        Preprocess a batch of examples
        
        Args:
            examples: Dictionary containing questions, contexts, and answers
            
        Returns:
            Dictionary containing processed features
        """
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]
        
        # Tokenize questions and contexts
        tokenized = self.tokenizer(
            questions,
            contexts,
            max_length=self.config["max_seq_length"],
            stride=self.config["doc_stride"],
            padding=self.config.get("padding", "max_length"),
            truncation=self.config.get("truncation", "only_second"),
            return_overflowing_tokens=self.config.get("return_overflowing_tokens", True),
            return_offsets_mapping=self.config.get("return_offsets_mapping", True)
        )
        
        # Process start and end positions
        start_positions = []
        end_positions = []
        
        for i, offset in enumerate(tokenized.offset_mapping):
            # Find the start and end of the context
            sequence_ids = tokenized.sequence_ids(i)
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1
            while sequence_ids[context_end] != 1:
                context_end -= 1
                
            # If no answer, set positions to CLS token
            if len(examples["answers"][i]["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue
                
            # Get the start and end positions in context
            start_char = examples["answers"][i]["answer_start"][0]
            end_char = start_char + len(examples["answers"][i]["text"][0])
            
            # Find token positions
            token_start_index = 0
            token_end_index = 0
            
            for idx, (start, end) in enumerate(offset):
                if start <= start_char and end > start_char:
                    token_start_index = idx
                if start < end_char and end >= end_char:
                    token_end_index = idx
                    break
                    
            # Adjust if answer is cut off
            if token_start_index <= context_start:
                token_start_index = context_start
            if token_end_index >= context_end:
                token_end_index = context_end
                
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)
            
        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        
        # Remove offset mapping if not needed
        if not self.config.get("return_offsets_mapping", True):
            tokenized.pop("offset_mapping")
            
        return tokenized
        
    def convert_to_features(self, question: str, context: str) -> Dict[str, torch.Tensor]:
        """
        Convert a single QA pair to features
        
        Args:
            question: Question text
            context: Context text
            
        Returns:
            Dictionary containing model input features
        """
        # Tokenize
        features = self.tokenizer(
            question,
            context,
            max_length=self.config["max_seq_length"],
            stride=self.config["doc_stride"],
            padding=self.config.get("padding", "max_length"),
            truncation=self.config.get("truncation", "only_second"),
            return_tensors="pt"
        )
        
        return features