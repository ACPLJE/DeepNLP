# src/utils/evaluation.py
import torch
from tqdm import tqdm
from .metrics import compute_exact_match, compute_f1

class Evaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def evaluate(self, dataloader):
        self.model.eval()
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Model forward pass
                outputs = self.model(**batch)
                
                # Get predicted answer spans
                start_logits = outputs['start_logits']
                end_logits = outputs['end_logits']
                
                # Get predictions for each example in batch
                for i in range(start_logits.size(0)):
                    # Get most likely start and end positions
                    start_idx = torch.argmax(start_logits[i]).item()
                    end_idx = torch.argmax(end_logits[i]).item()
                    
                    # Ensure valid span (start <= end)
                    if start_idx > end_idx:
                        start_idx, end_idx = end_idx, start_idx
                        
                    # Convert tokens to text
                    predicted_tokens = batch['input_ids'][i][start_idx:end_idx + 1]
                    predicted_answer = self.tokenizer.decode(predicted_tokens)
                    
                    # Get reference answer
                    reference_answer = batch['answers'][i]['text'][0]
                    
                    all_predictions.append(predicted_answer)
                    all_references.append(reference_answer)
        
        # Calculate metrics
        results = self.compute_metrics(all_predictions, all_references)
        return results
    
    def compute_metrics(self, predictions, references):
        exact_match = sum(compute_exact_match(pred, ref) 
                         for pred, ref in zip(predictions, references)) / len(predictions)
        f1 = sum(compute_f1(pred, ref) 
                for pred, ref in zip(predictions, references)) / len(predictions)
        
        return {
            'exact_match': exact_match * 100,
            'f1': f1 * 100
        }



