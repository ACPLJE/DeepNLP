# src/data/dataset.py
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class QADataset(Dataset):
    """Dataset class for Question Answering tasks."""

    def __init__(self, data_path, split, tokenizer, max_seq_length=384,
                 doc_stride=128, max_query_length=64):
        """
        Initialize QA Dataset.
        
        Args:
            data_path (str): Directory containing dataset files
            split (str): Dataset split ('train' or 'validation')
            tokenizer: Tokenizer for text processing
            max_seq_length (int): Maximum sequence length
            doc_stride (int): Stride for document windowing
            max_query_length (int): Maximum query length
        """
        self.data_path = data_path
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        
        # Load dataset
        self.examples = self._load_and_process_data()
        
    def _load_and_process_data(self):
        """Load and process the dataset."""
        # Determine file path based on split
        file_name = 'train-v1.1.json' if self.split == 'train' else 'dev-v1.1.json'
        file_path = f"{self.data_path}/{file_name}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        examples = []
        for article in dataset['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    
                    # Tokenize question and context
                    tokenized = self.tokenizer(
                        question,
                        context,
                        max_length=self.max_seq_length,
                        stride=self.doc_stride,
                        truncation='only_second',
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        padding='max_length'
                    )
                    
                    # Process answers for training
                    if self.split == 'train':
                        answer = qa['answers'][0]  # Take first answer
                        start_char = answer['answer_start']
                        answer_text = answer['text']
                        
                        # Convert character positions to token positions
                        for i, tokens in enumerate(tokenized.encodings):
                            start_token = tokens.char_to_token(start_char)
                            end_token = tokens.char_to_token(start_char + len(answer_text) - 1)
                            
                            if start_token is not None and end_token is not None:
                                example = {
                                    'input_ids': tokenized['input_ids'][i],
                                    'attention_mask': tokenized['attention_mask'][i],
                                    'start_positions': start_token,
                                    'end_positions': end_token,
                                    'overflow_to_sample_mapping': tokenized['overflow_to_sample_mapping'][i]
                                }
                                # Add token_type_ids if available
                                if 'token_type_ids' in tokenized:
                                    example['token_type_ids'] = tokenized['token_type_ids'][i]
                                else:
                                    example['token_type_ids'] = [0] * len(tokenized['input_ids'][i])
                                examples.append(example)
                    else:
                        # For validation, store original text for evaluation
                        example = {
                            'input_ids': tokenized['input_ids'][0],
                            'attention_mask': tokenized['attention_mask'][0],
                            'example_id': qa['id'],
                            'context': context,
                            'question': question,
                            'answers': qa['answers']
                        }
                        # Add token_type_ids if available
                        if 'token_type_ids' in tokenized:
                            example['token_type_ids'] = tokenized['token_type_ids'][0]
                        else:
                            example['token_type_ids'] = [0] * len(tokenized['input_ids'][0])
                        examples.append(example)
        
        return examples
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get a single example from the dataset."""
        return self.examples[idx]
    
    def collate_fn(self, batch):
        """
        Collate function for batching examples.
        
        Args:
            batch: List of examples to batch
            
        Returns:
            dict: Batched examples with PyTorch tensors
        """
        batch_dict = {}
        
        # Convert to tensors and pad sequences
        for key in batch[0].keys():
            if key in ['input_ids', 'attention_mask', 'token_type_ids']:
                batch_dict[key] = torch.tensor([example[key] for example in batch])
            elif key in ['start_positions', 'end_positions'] and self.split == 'train':
                batch_dict[key] = torch.tensor([example[key] for example in batch])
            else:
                # Keep non-tensor data (like example_id, context, etc.) as lists
                batch_dict[key] = [example[key] for example in batch]
        
        return batch_dict