import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class VerifierDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_claim_length=128, max_evidence_length=512):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}\n"
                f"Run 'python preprocess.py' first to generate processed data."
            )
            
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_claim_length = max_claim_length
        self.max_evidence_length = max_evidence_length
        
        required = {'claim', 'evidence', 'label', 'source'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}. Found: {list(self.df.columns)}")
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        claim = str(row['claim'])
        evidence = str(row['evidence']) if pd.notna(row['evidence']) else "No evidence provided."
        label = int(row['label'])
        source = str(row['source'])
        
        claim_enc = self.tokenizer(
            claim,
            max_length=self.max_claim_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        ev_enc = self.tokenizer(
            evidence,
            max_length=self.max_evidence_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'claim_input_ids': claim_enc['input_ids'].squeeze(0),
            'claim_attention_mask': claim_enc['attention_mask'].squeeze(0),
            'evidence_input_ids': ev_enc['input_ids'].squeeze(0),
            'evidence_attention_mask': ev_enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'source': source
        }