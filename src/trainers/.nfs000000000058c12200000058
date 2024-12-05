from src.trainers.base_trainer import BaseTrainer
from src.models.context_aware_distillation import ContextAwareDistillationModel
from src.utils.logger import setup_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class ContextAwareDistillationTrainer(BaseTrainer):
    def __init__(self, model, optimizer, train_dataloader, eval_dataloader, config, 
                 teacher_model=None, student_model=None):
        super().__init__(model, optimizer, train_dataloader, eval_dataloader, config)
        self.teacher_model = teacher_model
        self.student_model = student_model
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if teacher_model is not None:
            self.teacher_model = teacher_model.to(self.device)
        if student_model is not None:
            self.student_model = student_model.to(self.device)
    
  
        # Initialize context-aware distillation model
        self.context_aware_model = ContextAwareDistillationModel(
            teacher_model=teacher_model,
            student_model=student_model,
            config=config
        )
        
        # Set the model in parent class
        self.model = self.context_aware_model.to(self.device)

        
        self.optimizer = optimizer
  
        
        # Setup loss functions
        self.setup_loss_functions()
        
    def setup_loss_functions(self):
        """Initialize loss functions for different components"""
        self.qa_loss_fct = nn.CrossEntropyLoss()
        self.kl_loss_fct = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss_fct = nn.MSELoss()
        
    def _training_step(self, batch):
        self.optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        token_type_ids = batch.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
            
        start_positions = batch['start_positions'].to(self.device)
        end_positions = batch['end_positions'].to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Calculate losses...
        qa_loss = self.model.compute_loss(
            start_positions=start_positions,
            end_positions=end_positions,
            start_logits=outputs['start_logits'],
            end_logits=outputs['end_logits']
        )
        
        token_context_loss = self.mse_loss_fct(
            outputs['token_context'],
            outputs['teacher_hidden']
        )
        
        sequence_context_loss = self.mse_loss_fct(
            outputs['sequence_context'],
            outputs['teacher_hidden'].mean(dim=1, keepdim=True)
        )
        
        # Combine losses
        total_loss = (
            self.config['training']['qa_loss_weight'] * qa_loss +
            self.config['training']['token_context_weight'] * token_context_loss +
            self.config['training']['sequence_context_weight'] * sequence_context_loss
        )
        
        # Backward pass
        total_loss.backward()
        
        # Clip gradients
        if self.config['training']['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
        
        # Update weights
        self.optimizer.step()
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
            
        return total_loss  
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total_loss': 0,
            'qa_loss': 0,
            'token_context_loss': 0,
            'sequence_context_loss': 0
        }
        
        train_iterator = tqdm(
            self.train_dataloader,
            desc=f'Training Epoch {epoch + 1}',
            disable=not self.config.training.show_progress_bar
        )
        
        for step, batch in enumerate(train_iterator):
            step_losses = self._training_step(batch)
            
            # Update epoch losses
            for key in epoch_losses:
                epoch_losses[key] += step_losses[key]
                
            # Update progress bar
            if step % self.config.training.logging_steps == 0:
                train_iterator.set_postfix({
                    k: f'{v/(step+1):.4f}' for k, v in epoch_losses.items()
                })
                
        # Calculate average losses for the epoch
        num_steps = len(self.train_dataloader)
        return {k: v/num_steps for k, v in epoch_losses.items()}
        
    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        total_qa_loss = 0
        all_start_preds = []
        all_end_preds = []
        all_start_labels = []
        all_end_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc='Evaluating'):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,  # Use device-moved tensors
                    attention_mask=attention_mask,  # Use device-moved tensors
                    token_type_ids=token_type_ids  # Use device-moved tensors
                )
            
    
                
                if 'start_positions' in batch and 'end_positions' in batch:
                    start_positions = batch['start_positions'].to(self.device)
                    end_positions = batch['end_positions'].to(self.device)
                    
                    qa_loss = self.model.compute_loss(
                        start_positions,
                        end_positions,
                        outputs['start_logits'],
                        outputs['end_logits']
                    )
                    total_qa_loss += qa_loss.item()
                
                # Get predictions
                start_preds = torch.argmax(outputs['start_logits'], dim=1)
                end_preds = torch.argmax(outputs['end_logits'], dim=1)
                
                all_start_preds.extend(start_preds.cpu().numpy())
                all_end_preds.extend(end_preds.cpu().numpy())
                
                if 'start_positions' in batch:
                    all_start_labels.extend(start_positions.cpu().numpy())
                    all_end_labels.extend(end_positions.cpu().numpy())
                    
        # Calculate metrics
        metrics = {}
        if all_start_labels:  # If we have labels
            exact_matches = sum(
                1 for (sp, ep, sl, el) in zip(
                    all_start_preds, all_end_preds,
                    all_start_labels, all_end_labels
                )
                if sp == sl and ep == el
            )
            metrics['exact_match'] = 100.0 * exact_matches / len(all_start_labels)
            metrics['qa_loss'] = total_qa_loss / len(self.eval_dataloader)
            
        return metrics