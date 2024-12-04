# src/models/context_aware_distillation.py
import torch
import torch.nn as nn
from .continuous_token_representation import ContinuousTokenRepresentation

class ContextAwareDistillationModel(nn.Module):
    def __init__(self, teacher_model, student_model, config):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        
        # Token representation
        self.continuous_token_rep = ContinuousTokenRepresentation(
            self.student.config.vocab_size,
            self.config.hidden_size,
            self.config.dropout_rate
        )
        
        # Context layers
        self.token_context = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout_rate
        )
        
        self.sequence_context = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
            nn.LayerNorm(self.config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        )
        
        # Projections
        self.teacher_projection = nn.Linear(
            self.teacher.config.hidden_size,
            self.config.hidden_size
        )
        
        # Output layers
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Teacher forward pass
        teacher_outputs = self.teacher(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        teacher_hidden = teacher_outputs.hidden_states[-1]
        teacher_projected = self.teacher_projection(teacher_hidden)
        
        # Student forward with continuous token representation
        continuous_tokens = self.continuous_token_rep(input_ids)
        student_outputs = self.student(
            inputs_embeds=continuous_tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        student_hidden = student_outputs.hidden_states[-1]
        
        # Token-level context
        token_context_output, _ = self.token_context(
            query=student_hidden.permute(1, 0, 2),
            key=teacher_projected.permute(1, 0, 2),
            value=teacher_projected.permute(1, 0, 2),
            key_padding_mask=~attention_mask.bool()
        )
        token_context_output = token_context_output.permute(1, 0, 2)
        
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
            'sequence_context': sequence_context
        }

