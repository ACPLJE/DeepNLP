# src/models/context_aware_distillation.py
import torch
import torch.nn as nn
import math
from .continuous_token_representation import ContinuousTokenRepresentation

class ContextAwareDistillationModel(nn.Module):

    def __init__(self, teacher_model, student_model, config):  
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.config = config.get('model', {})
    
        required_keys = ['hidden_size', 'dropout_rate']
        missing_keys = [key for key in required_keys if key not in self.config]
        
        if missing_keys:
            raise KeyError(f"Missing required configuration keys in model config: {missing_keys}. Available keys: {list(self.config.keys())}")
        self.continuous_token_rep = ContinuousTokenRepresentation(
            self.student.config.vocab_size,
            self.config['hidden_size'],
            self.config['dropout_rate']
        )
        # Token to Query transformation
        self.token_to_query = nn.Linear(self.config['hidden_size'], self.config['hidden_size'])
        self.sequence_pooling = nn.Sequential(
            nn.Linear(self.config['hidden_size'], self.config['hidden_size']),
            nn.Tanh(),
            nn.Linear(self.config['hidden_size'], 1)
        )
        # Sequence to key/value transformations
        self.sequence_to_key = nn.Linear(self.config['hidden_size'], self.config['hidden_size'])
        self.sequence_to_value = nn.Linear(self.config['hidden_size'], self.config['hidden_size'])
        
        # Normalization and dropout for context
        self.context_layer_norm = nn.LayerNorm(self.config['hidden_size'])
        self.context_dropout = nn.Dropout(self.config['dropout_rate'])
        
        # Sequence context processing
        self.sequence_context = nn.Sequential(
            nn.Linear(self.config['hidden_size'], self.config['hidden_size'] * 2),
            nn.LayerNorm(self.config['hidden_size'] * 2),
            nn.ReLU(),
            nn.Dropout(self.config['dropout_rate']), 
            nn.Linear(self.config['hidden_size'] * 2, self.config['hidden_size'])
        )
        
        # Teacher projection
        self.teacher_projection = nn.Linear(
            self.teacher.config.hidden_size,
            self.config['hidden_size']
        )
        
        # Output layers
        self.qa_outputs = nn.Linear(self.config['hidden_size'], 2)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Teacher forward pass
        teacher_outputs = self.teacher(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        teacher_hidden = teacher_outputs['hidden_states'][-1]
        teacher_projected = self.teacher_projection(teacher_hidden)
        
        # Student forward with continuous token representation
        continuous_tokens = self.continuous_token_rep(input_ids)
        student_outputs = self.student(
            inputs_embeds=continuous_tokens,
            attention_mask=attention_mask,

            output_hidden_states=True
        )
        student_hidden = student_outputs['hidden_states'][-1]
        
        # Token-level context processing
        # 1. Transform tokens to queries
        token_queries = self.token_to_query(student_hidden)
        
        # 2. Create sequence-level representation from teacher
        sequence_weights = self.sequence_pooling(teacher_projected)
        sequence_weights = torch.softmax(sequence_weights, dim=1)
        sequence_repr = torch.sum(teacher_projected * sequence_weights, dim=1, keepdim=True)
        
        # 3. Transform sequence representation to keys and values
        sequence_keys = self.sequence_to_key(sequence_repr)
        sequence_values = self.sequence_to_value(sequence_repr)
        
        # 4. Compute attention scores
        attention_scores = torch.matmul(token_queries, sequence_keys.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(token_queries.size(-1))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(),
                float('-inf')
            )
        
        # 5. Get attention probabilities and compute context
        attention_probs = torch.softmax(attention_scores, dim=-1)
        token_context_output = torch.matmul(attention_probs, sequence_values)
        
        # Apply normalization and dropout
        token_context_output = self.context_layer_norm(token_context_output)
        token_context_output = self.context_dropout(token_context_output)
        
        # Sequence-level context
        sequence_mask = attention_mask.unsqueeze(-1).float()
        sequence_context = self.sequence_context(
            (student_hidden * sequence_mask).sum(1) / sequence_mask.sum(1)
        ).unsqueeze(1)
        
        # Combine contexts
        combined_context = token_context_output + sequence_context
        
        # QA outputs
        logits = self.qa_outputs(combined_context)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        return {
            'start_logits': start_logits.squeeze(-1),
            'end_logits': end_logits.squeeze(-1),
            'student_hidden': student_hidden,
            'teacher_hidden': teacher_hidden,
            'token_context': token_context_output,
            'sequence_context': sequence_context,
            'attention_weights': attention_probs  # for analysis
        }

    def compute_loss(self, start_positions, end_positions, start_logits, end_logits):
        # Ensure all tensors are on the same device as the model
        device = start_logits.device
        start_positions = start_positions.to(device)
        end_positions = end_positions.to(device)
        
        start_loss = nn.CrossEntropyLoss()(start_logits, start_positions)
        end_loss = nn.CrossEntropyLoss()(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss