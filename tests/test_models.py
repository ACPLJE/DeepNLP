import unittest
import torch
from src.models.model_factory import ModelFactory
from src.models.base_model import BaseModel
from src.models.context_aware_distillation import ContextAwareDistillation
from src.utils.logger import setup_logger

class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = setup_logger('test_models')
        cls.model_config = {
            'bert': {
                'model_name': 'bert-base-uncased',
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_hidden_layers': 12
            },
            'distilbert': {
                'model_name': 'distilbert-base-uncased',
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_hidden_layers': 6
            }
        }

    def setUp(self):
        self.model_factory = ModelFactory()
        
    def test_create_bert_model(self):
        model = self.model_factory.create_model('bert', self.model_config['bert'])
        self.assertIsInstance(model, BaseModel)
        
    def test_create_distilbert_model(self):
        model = self.model_factory.create_model('distilbert', self.model_config['distilbert'])
        self.assertIsInstance(model, BaseModel)
        
    def test_model_forward_pass(self):
        model = self.model_factory.create_model('bert', self.model_config['bert'])
        batch_size = 4
        seq_length = 128
        
        dummy_input = {
            'input_ids': torch.randint(0, 30522, (batch_size, seq_length)),
            'attention_mask': torch.ones(batch_size, seq_length),
            'token_type_ids': torch.zeros(batch_size, seq_length)
        }
        
        outputs = model(**dummy_input)
        self.assertIn('start_logits', outputs)
        self.assertIn('end_logits', outputs)
        
    def test_context_aware_distillation(self):
        teacher_model = self.model_factory.create_model('bert', self.model_config['bert'])
        student_model = self.model_factory.create_model('distilbert', self.model_config['distilbert'])
        
        distillation_model = ContextAwareDistillation(
            teacher_model=teacher_model,
            student_model=student_model,
            temperature=2.0
        )
        
        self.assertIsNotNone(distillation_model)
        
    def test_invalid_model_type(self):
        with self.assertRaises(ValueError):
            self.model_factory.create_model('invalid_model', {})

if __name__ == '__main__':
    unittest.main()