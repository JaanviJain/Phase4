import os
import json
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
)

from src.model import MultimodalVerifier
from src.pipeline_loader import load_phase3_for_ablation


class AblationDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_claim_length=128, max_evidence_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_claim_length = max_claim_length
        self.max_evidence_length = max_evidence_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        claim = str(row['claim'])
        evidence = str(row['evidence']) if pd.notna(row['evidence']) else "No evidence provided."
        label = int(row['label'])
        
        claim_enc = self.tokenizer(
            claim, max_length=self.max_claim_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        ev_enc = self.tokenizer(
            evidence, max_length=self.max_evidence_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        return {
            'claim_input_ids': claim_enc['input_ids'].squeeze(0),
            'claim_attention_mask': claim_enc['attention_mask'].squeeze(0),
            'evidence_input_ids': ev_enc['input_ids'].squeeze(0),
            'evidence_attention_mask': ev_enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'claim_id': row.get('claim_id', str(idx))
        }


def evaluate_condition(model, dataloader, device, condition_name):
    model.eval()
    all_preds = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            claim_input_ids = batch['claim_input_ids'].to(device)
            claim_attention_mask = batch['claim_attention_mask'].to(device)
            evidence_input_ids = batch['evidence_input_ids'].to(device)
            evidence_attention_mask = batch['evidence_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(
                claim_input_ids=claim_input_ids,
                claim_attention_mask=claim_attention_mask,
                evidence_input_ids=evidence_input_ids,
                evidence_attention_mask=evidence_attention_mask
            )
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(batch['claim_id'])
    
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision, recall, f1_per, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    
    return {
        'condition': condition_name,
        'accuracy': float(acc),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_per_class': f1_per.tolist(),
        'support': support.tolist(),
        'confusion_matrix': cm.tolist(),
        'predictions': [int(p) for p in all_preds],
        'labels': [int(l) for l in all_labels],
        'claim_ids': all_ids
    }


def main():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['biobert'],
        use_fast=False
    )
    
    checkpoint_path = os.path.join(config['output']['checkpoint_dir'], "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Train model first! Checkpoint not found: {checkpoint_path}")
        print("Run: python train.py")
        return
    
    model = MultimodalVerifier(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n[OK] Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"     Best validation F1 during training: {checkpoint.get('best_f1', 0):.4f}")
    
    p3 = config.get('phase3', {})
    weighted_path = p3.get('weighted_evidence_path', 'data/pipeline/evidence_weighted.json')
    unweighted_path = p3.get('unweighted_evidence_path', 'data/pipeline/evidence_unweighted.json')
    top_k = p3.get('top_k_evidence', 3)
    
    if not os.path.exists(weighted_path):
        print(f"\n[ERROR] Phase 3 weighted evidence not found: {weighted_path}")
        return
    if not os.path.exists(unweighted_path):
        print(f"\n[ERROR] Phase 3 unweighted evidence not found: {unweighted_path}")
        return
    
    print("\n" + "="*70)
    print("ABLATION STUDY: Weighted vs. Unweighted Evidence")
    print("="*70)
    
    results = {}
    
    for condition, path in [('Weighted', weighted_path), ('Unweighted', unweighted_path)]:
        print(f"\n{'='*70}")
        print(f"CONDITION: {condition.upper()}")
        print(f"{'='*70}")
        
        df = load_phase3_for_ablation(path, condition=condition.lower(), top_k=top_k)
        
        if len(df) == 0:
            print("[ERROR] No data loaded!")
            continue
        
        print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")
        
        dataset = AblationDataset(
            df, tokenizer,
            max_claim_length=config['data']['max_claim_length'],
            max_evidence_length=config['data']['max_evidence_length']
        )
        loader = DataLoader(dataset, batch_size=config['training']['batch_size'], 
                           shuffle=False, num_workers=0)
        
        metrics = evaluate_condition(model, loader, device, condition)
        results[condition] = metrics
        
        print(f"\nResults:")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
        print(f"  Per-class F1 (Refuted/NEI/Supported): {metrics['f1_per_class']}")
        print(f"  Confusion Matrix:")
        print(f"                 Pred:0   Pred:1   Pred:2")
        for i, row in enumerate(metrics['confusion_matrix']):
            name = ['Refuted ', 'NEI     ', 'Supported'][i]
            print(f"  True {name}  {row[0]:6d}   {row[1]:6d}   {row[2]:6d}")
    
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    w = results.get('Weighted', {})
    u = results.get('Unweighted', {})
    
    if w and u:
        print(f"\n{'Metric':<25} {'Weighted':<12} {'Unweighted':<12} {'Delta':<12}")
        print("-" * 61)
        print(f"{'Accuracy':<25} {w['accuracy']:.4f}       {u['accuracy']:.4f}       {w['accuracy']-u['accuracy']:+.4f}")
        print(f"{'Macro F1':<25} {w['macro_f1']:.4f}       {u['macro_f1']:.4f}       {w['macro_f1']-u['macro_f1']:+.4f}")
        print(f"{'Weighted F1':<25} {w['weighted_f1']:.4f}       {u['weighted_f1']:.4f}       {w['weighted_f1']-u['weighted_f1']:+.4f}")
        
        delta_f1 = w['macro_f1'] - u['macro_f1']
        print(f"\n{'='*70}")
        if delta_f1 > 0.02:
            print(f"✅ CONCLUSION: Weighted evidence IMPROVES F1 by {delta_f1:+.4f} ({delta_f1*100:+.1f}%)")
        elif delta_f1 < -0.02:
            print(f"⚠️ CONCLUSION: Unweighted evidence is better by {abs(delta_f1):.4f}")
            print("   The neural retriever already ranks high-quality evidence well.")
        else:
            print(f"🟡 CONCLUSION: No significant difference (ΔF1 = {delta_f1:+.4f})")
            print("   Hierarchy weighting does not change retrieval ranking meaningfully.")
        print(f"{'='*70}")
    
    report = {
        'weighted': {k: v for k, v in w.items() if k not in ['predictions', 'labels', 'claim_ids']},
        'unweighted': {k: v for k, v in u.items() if k not in ['predictions', 'labels', 'claim_ids']},
        'delta_macro_f1': w.get('macro_f1', 0) - u.get('macro_f1', 0)
    }
    
    report_path = os.path.join(config['output']['log_dir'], "ablation_report.json")
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVE] Ablation report: {report_path}")


if __name__ == "__main__":
    main()