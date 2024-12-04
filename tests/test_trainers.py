import unittest
import torch
from src.trainers.base_trainer import BaseTrainer
from src.trainers.distillation_trainer import DistillationTrainer
from src.models.model_factory import ModelFactory
from src.data.data_loader import DataLoaderFactory
from src.utils.logger import setup_logger

class TestTrainers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = setup_logger('test_trainers')
        cls.model_config = {
            'bert': {
                'model_name': 'bert-base-uncased',
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_hidden_layers': 12
            }
        }
        cls.training_config = {
            'num_epochs': 3,
            'learning_rate': 3e-5,
            'warmup_steps': 0,
            'max_grad_norm': 1.0,
            'logging_steps': 100,
            'save_steps': 1000
        }

    def setUp(self):
        self.model_factory = ModelFactory()
        self.data_loader_factory = DataLoaderFactory()
        
    def test_base_trainer_initialization(self):
        model = self.model_factory.create_model('bert', self.model_config['bert'])
        trainer = BaseTrainer(
            model=model,
            training_config=self.training_config,
            device='cpu'
        )
        self.assertIsNotNone(trainer)
        
    def test_distillation_trainer_initialization(self):
        teacher_model = self.model_factory.create_model('bert', self.model_config['bert'])
        student_model = self.model_factory.create_model('bert', self.model_config['bert'])
        
        trainer = DistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            training_config=self.training_config,
            device='cpu',
            temperature=2.0
        )
        self.assertIsNotNone(trainer)
        
    def test_training_step(self):
        model = self.model_factory.create_model('bert', self.model_config['bert'])
        trainer = BaseTrainer(
            model=model,
            training_config=self.training_config,
            device='cpu'
        )
        
        batch = {
            'input_ids': torch.randint(0, 30522, (4, 128)),
            'attention_mask': torch.ones(4, 128),
            'token_type_ids': torch.zeros(4, 128),
            'start_positions': torch.randint(0, 127, (4,)),
            'end_positions': torch.randint(0, 127, (4,))
        }
        
        loss = trainer.training_step(batch)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.requires_grad)
        
    def test_validation_step(self):
        model = self.model_factory.create_model('bert', self.model_config['bert'])
        trainer = BaseTrainer(
            model=model,
            training_config=self.training_config,
            device='cpu'
        )
        
        batch = {
            'input_ids': torch.randint(0, 30522, (4, 128)),
            'attention_mask': torch.ones(4, 128),
            'token_type_ids': torch.zeros(4, 128),
            'start_positions': torch.randint(0, 127, (4,)),
            'end_positions': torch.randint(0, 127, (4,))
        }
        
        with torch.no_grad():
            outputs = trainer.validation_step(batch)
        
        self.assertIn('loss', outputs)
        self.assertIn('start_logits', outputs)
        self.assertIn('end_logits', outputs)

if __name__ == '__main__':
    unittest.main()