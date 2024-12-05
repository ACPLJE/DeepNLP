# src/trainers/base_trainer.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup 
from tqdm import tqdm
import os

class BaseTrainer:
    def __init__(self, model, config, train_dataloader, eval_dataloader, logger):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup optimizer and scheduler
        self.setup_optimization()
        
    def setup_optimization(self):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        num_training_steps = len(self.train_dataloader) * self.config['training']['num_epochs']
        num_warmup_steps = self.config['training']['warmup_steps']
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
    def save_checkpoint(self, epoch, metrics, checkpoint_dir):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        self.logger.info(f'Saved checkpoint: {path}')
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train_epoch(self, epoch):
        raise NotImplementedError
        
    def evaluate(self):
        raise NotImplementedError
        
    def train(self):
        self.model.to(self.device)
        best_metric = float('-inf')
        checkpoint_dir = self.config['training'].get('checkpoint_dir', 'checkpoints')
        
        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss = self._training_step(batch)
                total_loss += loss
                
                if batch_idx % self.config['training'].get('log_interval', 100) == 0:
                    current = batch_idx * len(batch)
                    print(f'Epoch: {epoch} [{current}/{len(self.train_dataloader.dataset)} '
                          f'({100. * batch_idx / len(self.train_dataloader):.0f}%)]\tLoss: {loss:.6f}')
    
           
            avg_loss = total_loss / len(self.train_dataloader)
            print(f'Epoch {epoch} Average Loss: {avg_loss:.6f}')
            
      
            eval_metrics = self.evaluate()
            
        
            self.save_checkpoint(
                epoch=epoch,
                metrics={
                    'loss': avg_loss,
                    'eval_metrics': eval_metrics
                },
                checkpoint_dir=checkpoint_dir
            )
            
           
            if eval_metrics.get('accuracy', 0) > best_metric: 
                best_metric = eval_metrics.get('accuracy', 0)
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': eval_metrics
                }, best_checkpoint_path)
                self.logger.info(f'Saved best model: {best_checkpoint_path}')