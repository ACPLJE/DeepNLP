# src/trainers/distillation_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from transformers import BertModel, DistilBertModel
from tqdm import tqdm
from src.models.context_aware_distillation import ContextAwareDistillationModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from src.utils.logger import setup_logger
from tqdm import tqdm

class DistillationTrainer(BaseTrainer):
    def __init__(self, 
                 teacher_model, 
                 student_model,
                 train_loader,
                 val_loader,
                 optimizer,
                 criterion,
                 device,
                 config):
        super().__init__(
            model=student_model,
            config=config,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            logger=setup_logger('distillation_trainer')
        )
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.student_model = student_model
        self.setup_loss_functions()
        self.distillation_model = ContextAwareDistillationModel(
        teacher_model=self.teacher_model,
        student_model=self.student_model,
        config=config
        )
    def _training_step(self, batch):
        self.optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        token_type_ids = batch['token_type_ids'].to(self.device) if 'token_type_ids' in batch else None
        
        # context-aware distillation model 사용
        outputs = self.distillation_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        if 'start_positions' in batch and 'end_positions' in batch:
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)
            loss = self.distillation_model.compute_loss(
                start_positions, 
                end_positions,
                outputs['start_logits'],
                outputs['end_logits']
            )
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()

    def setup_loss_functions(self):
        self.qa_loss_fct = nn.CrossEntropyLoss()
        self.kl_loss_fct = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss_fct = nn.MSELoss()
        
    def compute_distillation_loss(self, student_logits, teacher_logits, attention_mask):
        """Compute KL divergence loss between student and teacher logits"""
        # Temperature scaling
        temp = self.config.training.temperature
        student_logits = student_logits / temp
        teacher_logits = teacher_logits / temp
        
        # Compute KL divergence
        kl_loss = self.kl_loss_fct(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1)
        )
        return kl_loss * (temp ** 2)
        
    def compute_representation_loss(self, student_hidden, teacher_hidden, attention_mask):
        """Compute MSE loss between student and teacher hidden states"""
        mask = attention_mask.unsqueeze(-1).expand_as(student_hidden)
        masked_student = student_hidden * mask
        masked_teacher = teacher_hidden * mask
        return self.mse_loss_fct(masked_student, masked_teacher)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        train_iterator = tqdm(self.train_dataloader, desc=f'Training Epoch {epoch + 1}')
        
        for step, batch in enumerate(train_iterator):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            
            # Calculate losses
            qa_loss = (self.qa_loss_fct(outputs['start_logits'], batch['start_positions']) + 
                      self.qa_loss_fct(outputs['end_logits'], batch['end_positions'])) / 2
                      
            distill_loss = self.compute_distillation_loss(
                outputs['student_hidden'],
                outputs['teacher_hidden'],
                batch['attention_mask']
            )
            
            hidden_loss = self.compute_representation_loss(
                outputs['student_hidden'],
                outputs['teacher_hidden'],
                batch['attention_mask']
            )
            
            # Combine losses
            loss = (self.config.training.qa_loss_weight * qa_loss +
                   self.config.training.distill_loss_weight * distill_loss +
                   self.config.training.hidden_loss_weight * hidden_loss)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Log progress
            if step % self.config.training.logging_steps == 0:
                self.logger.info(f'Step {step}: Loss = {loss.item():.4f}')
        
        return {'loss': total_loss / len(self.train_dataloader)}
        
    def evaluate(self):
        self.model.eval()
        total_qa_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc='Evaluating'):
              
                model_inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                }
                
              
                if 'token_type_ids' in batch:
                    model_inputs['token_type_ids'] = batch['token_type_ids'].to(self.device)
                
                outputs = self.model(**model_inputs)
                
            
                if 'start_positions' in batch and 'end_positions' in batch:
                    start_positions = batch['start_positions'].to(self.device)
                    end_positions = batch['end_positions'].to(self.device)
                    
                    qa_loss = (self.qa_loss_fct(outputs['start_logits'], start_positions) +
                              self.qa_loss_fct(outputs['end_logits'], end_positions)) / 2
                    
                    total_qa_loss += qa_loss.item()
                    
               
                    start_preds = torch.argmax(outputs['start_logits'], dim=1)
                    end_preds = torch.argmax(outputs['end_logits'], dim=1)
                    
                    all_predictions.extend(list(zip(start_preds.cpu().numpy(), 
                                                 end_preds.cpu().numpy())))
                    all_labels.extend(list(zip(start_positions.cpu().numpy(),
                                             end_positions.cpu().numpy())))
                else:
                 
                    start_preds = torch.argmax(outputs['start_logits'], dim=1)
                    end_preds = torch.argmax(outputs['end_logits'], dim=1)
                    all_predictions.extend(list(zip(start_preds.cpu().numpy(), 
                                                 end_preds.cpu().numpy())))
        
       
        metrics = {}
        if all_labels:
            metrics = self.calculate_metrics(all_predictions, all_labels)
            metrics['qa_loss'] = total_qa_loss / len(self.eval_dataloader)
        else:
            self.logger.warning("No labels found in evaluation data. Skipping metric calculation.")
            metrics['qa_loss'] = float('nan')  
        
        return metrics
    
    def calculate_metrics(self, predictions, labels):
        """Calculate Exact Match and F1 scores"""
        exact_match = sum(1 for pred, label in zip(predictions, labels)
                         if pred == label) / len(predictions)
        return {
            'exact_match': exact_match * 100,
            'f1': 0  # F1 score calculation will be implemented in metrics.py
        }