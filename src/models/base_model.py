# src/models/base_model.py
import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel
class BaseModel(nn.Module):
    """Base model class for question answering"""
    
    def __init__(self, base_model, hidden_size, num_labels=2):
        """
        Initialize the QA model.
        
        Args:
            base_model: Pretrained transformer model
            hidden_size (int): Size of hidden layers
            num_labels (int): Number of output labels (typically 2 for start/end)
        """
        super().__init__()
        self.base_model = base_model
        self.qa_outputs = nn.Linear(hidden_size, num_labels)
        
      

    def forward(self, input_ids, attention_mask, token_type_ids=None, 
                start_positions=None, end_positions=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            start_positions: Ground truth start positions (optional)
            end_positions: Ground truth end positions (optional)
            
        Returns:
            dict: Model outputs containing logits and loss if training
        """
        # Get base model outputs - DistilBERT 처리 추가
        if isinstance(self.base_model, DistilBertModel):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        
        sequence_output = outputs[0]  # Last hidden state
        
        # Get logits for start and end positions
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits
        }
        
        # Calculate loss during training
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs['loss'] = total_loss
            
        return outputs