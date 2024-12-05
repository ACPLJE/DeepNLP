# src/models/base_model.py
import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel
class BaseModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.qa_outputs = nn.Linear(hidden_size, num_labels)
    
        self.config = base_model.config
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                start_positions=None, end_positions=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            **kwargs
        )
        
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }
        
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs['loss'] = total_loss
            
        return outputs