import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

class SQuADDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=384):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        question = example['question']
        context = example['context']
        
        # 정답이 있는 경우(학습/검증용)와 없는 경우(테스트용) 구분
        if 'answers' in example and len(example['answers']['text']) > 0:
            answer_text = example['answers']['text'][0]
            answer_start = example['answers']['answer_start'][0]
        else:
            answer_text = ""
            answer_start = 0

        # 토큰화
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True,
            stride=128
        )

        # offset_mapping을 이용해 시작/끝 위치 찾기
        offset_mapping = encoding.pop('offset_mapping').squeeze()
        
        # 정답이 있는 경우에만 시작/끝 위치 계산
        if answer_text:
            start_positions = end_positions = 0
            for idx, (start, end) in enumerate(offset_mapping):
                if start <= answer_start < end:
                    start_positions = idx
                if start < answer_start + len(answer_text) <= end:
                    end_positions = idx
                    break
        else:
            start_positions = end_positions = 0

        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding['start_positions'] = torch.tensor(start_positions)
        encoding['end_positions'] = torch.tensor(end_positions)
        
        return encoding

class ParallelRepresentationQA(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.token_query = nn.Linear(hidden_size, hidden_size)
        self.seq_key = nn.Linear(hidden_size, hidden_size)
        self.seq_value = nn.Linear(hidden_size, hidden_size)
        self.qa_outputs = nn.Linear(hidden_size * 2, 2)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        token_queries = self.token_query(sequence_output)
        seq_keys = self.seq_key(sequence_output)
        seq_values = self.seq_value(sequence_output)
        
        attention_scores = torch.matmul(token_queries, seq_keys.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.bert.config.hidden_size, dtype=torch.float32))
        
        attention_scores = attention_scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        context_aware_repr = torch.matmul(attention_probs, seq_values)
        final_repr = torch.cat([sequence_output, context_aware_repr], dim=-1)
        
        logits = self.qa_outputs(final_repr)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

def compute_metrics(predictions, references, tokenizer):
    f1_scores = []
    exact_matches = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        
        if pred_tokens == ref_tokens:
            exact_matches += 1
            
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(0)
            continue
            
        common_tokens = pred_tokens.intersection(ref_tokens)
        precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return {
        'f1': np.mean(f1_scores),
        'exact_match': exact_matches / len(predictions) * 100
    }

def train_model(model, train_dataloader, val_dataloader, tokenizer, device, num_epochs=10):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.to(device)
    
    best_val_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            optimizer.zero_grad()
            start_logits, end_logits = model(input_ids, attention_mask)
            
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"\nEpoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        val_results = evaluate_model(model, val_dataloader, tokenizer, device)
        print(f"Validation F1: {val_results['f1']:.4f}")
        print(f"Validation EM: {val_results['exact_match']:.4f}")
        
        if val_results['f1'] > best_val_f1:
            best_val_f1 = val_results['f1']
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved new best model!")

def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            start_logits, end_logits = model(input_ids, attention_mask)
            
            # Get predictions
            start_idx = torch.argmax(start_logits, dim=1)
            end_idx = torch.argmax(end_logits, dim=1)
            
            # Ensure end_idx >= start_idx
            end_idx = torch.max(end_idx, start_idx)
            
            for i in range(input_ids.shape[0]):
                pred_tokens = input_ids[i][start_idx[i]:end_idx[i]+1]
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                all_predictions.append(pred_text)
                
                ref_tokens = input_ids[i][batch['start_positions'][i]:batch['end_positions'][i]+1]
                ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                all_references.append(ref_text)
    
    return compute_metrics(all_predictions, all_references, tokenizer)

class BaselineQA(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

if __name__ == "__main__":
    print("Initializing...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    squad_dataset = load_dataset('squad')
    
    print("Creating datasets...")
    train_dataset = SQuADDataset(squad_dataset['train'], tokenizer)
    val_dataset = SQuADDataset(squad_dataset['validation'], tokenizer)
    
    print("Creating dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    print("\nTraining Parallel Representation Model...")
    parallel_model = ParallelRepresentationQA()
    train_model(parallel_model, train_dataloader, val_dataloader, tokenizer, device)
    
    print("\nTraining Baseline Model...")
    baseline_model = BaselineQA()
    train_model(baseline_model, train_dataloader, val_dataloader, tokenizer, device)
    
    print("\nFinal Evaluation Results:")
    parallel_results = evaluate_model(parallel_model, val_dataloader, tokenizer, device)
    baseline_results = evaluate_model(baseline_model, val_dataloader, tokenizer, device)
    
    print("\nParallel Representation Model:")
    print(f"F1 Score: {parallel_results['f1']:.4f}")
    print(f"Exact Match: {parallel_results['exact_match']:.4f}")
    
    print("\nBaseline DistilBERT Model:")
    print(f"F1 Score: {baseline_results['f1']:.4f}")
    print(f"Exact Match: {baseline_results['exact_match']:.4f}")