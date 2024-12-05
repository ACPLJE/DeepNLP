import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup 
from tqdm import tqdm
import os
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(self, model, optimizer, train_dataloader, eval_dataloader, config):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.setup_optimization()
        
    def setup_optimization(self):
        if self.optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        
        num_training_steps = len(self.train_dataloader) * self.config['training']['epochs']
        num_warmup_steps = self.config['training']['warmup_steps']
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def _training_step(self, batch):
        """단일 학습 스텝을 처리합니다."""
        # 배치를 디바이스로 이동
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward 패스
        self.optimizer.zero_grad()  # 그래디언트 초기화를 여기서 수행
        outputs = self.model(**batch)
        
        # 모델 출력이 손실을 포함하는 튜플인 경우를 처리
        if isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            
        return loss
    
    def _eval_step(self, batch):
        """단일 평가 스텝을 처리합니다."""
        with torch.no_grad():  # 평가시에는 그래디언트 계산 불필요
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            outputs = self.model(**batch)
            
            if isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
        return loss
    
    def train(self):
        logger.info("Starting training...")
        self.model.train()
        
        for epoch in range(self.config['training']['epochs']):
            epoch_iterator = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch}", 
                disable=not self.config['training']['show_progress_bar']
            )
            
            for step, batch in enumerate(epoch_iterator):
                try:
                    # 학습 스텝 수행
                    loss = self._training_step(batch)
                    
            
                    
                    # 그래디언트 클리핑
                    if self.config['training']['max_grad_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['training']['max_grad_norm']
                        )
                    
                    # 옵티마이저 스텝
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    # 그래디언트 초기화
                    self.optimizer.zero_grad()
                    
                    # 진행상황 업데이트
                    epoch_iterator.set_postfix(loss=loss.item())
                    
                    # 로깅
                    if self.global_step % self.config['training']['logging_steps'] == 0:
                        logger.info(f"Step: {self.global_step}, Loss: {loss.item():.4f}")
                    
                    # 평가
                    if self.global_step % self.config['training']['eval_steps'] == 0:
                        eval_loss = self.evaluate()
                        if isinstance(eval_loss, dict):
                            eval_loss_value = eval_loss['total_loss'] if 'total_loss' in eval_loss else sum(eval_loss.values())
                        else:
                            eval_loss_value = eval_loss
                        
                        logger.info(f"Step: {self.global_step}, Eval Loss: {float(eval_loss_value):.4f}")
                        
                        if eval_loss_value < self.best_eval_loss:
                            self.best_eval_loss = eval_loss_value
                            if self.config['training']['save_steps'] > 0:
                                self.save_model(f"checkpoint-{self.global_step}")
                    
                    self.global_step += 1
                    
                except Exception as e:
                    logger.error(f"Error during training step: {str(e)}")
                    raise e
    
    def evaluate(self):
        logger.info("Starting evaluation...")
        self.model.eval()
        total_eval_loss = 0
        eval_steps = 0
        
        with torch.no_grad():  # 평가 전체를 no_grad로 감싸기
            for batch in tqdm(
                self.eval_dataloader,
                desc="Evaluating",
                disable=not self.config['training']['show_progress_bar']
            ):
                # eval_step에서 반환된 loss가 tensor인지 확인
                eval_loss = self._eval_step(batch)
                
                # tensor에서 숫자 값으로 변환
                if isinstance(eval_loss, dict):
                    # 딕셔너리인 경우 total loss 값을 사용
                    eval_loss = eval_loss['total_loss'] if 'total_loss' in eval_loss else sum(eval_loss.values())
                
                total_eval_loss += eval_loss.item() if torch.is_tensor(eval_loss) else eval_loss
                eval_steps += 1
                
        avg_eval_loss = total_eval_loss / eval_steps
        
        self.model.train()
        return avg_eval_loss  
    
    def save_model(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))
        logger.info(f"Model saved to {output_dir}")
    
    def save_checkpoint(self, epoch, metrics, checkpoint_dir):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics
        }
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        logger.info(f'Saved checkpoint: {path}')
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']