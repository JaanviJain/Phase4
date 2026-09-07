"""
multimodal_demo.py
Demonstrates end-to-end multimodal inference on Phase 1 datasets.
This proves the architecture can process images, even though benchmark training was text-only.
"""

import os
import json
import yaml
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ViTImageProcessor

from src.model import MultimodalVerifier


class MultimodalDemoDataset(torch.utils.data.Dataset):
    """
    Loads FakeHealth/ReCOVery/Honest OOC samples with real images.
    """
    def __init__(self, csv_path, image_dir, tokenizer, image_processor, 
                 max_claim_length=128, max_evidence_length=512):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_claim_length = max_claim_length
        self.max_evidence_length = max_evidence_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Claim text
        claim = str(row.get('claim', row.get('title', row.get('caption', 'No claim'))))
        
        # Evidence text
        evidence = str(row.get('evidence', row.get('explanation', row.get('body', 'No evidence'))))
        if len(evidence) < 10:
            evidence = "No detailed evidence available."
        
        # Image
        img_path = row.get('image_path', row.get('image_url', None))
        if img_path and os.path.exists(os.path.join(self.image_dir, img_path)):
            image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
            image = self.image_processor(image, return_tensors='pt')['pixel_values'].squeeze(0)
        else:
            image = None
        
        # Label mapping (binary → 3-class)
        raw_label = str(row.get('label', row.get('verdict', row.get('rating', '1')))).lower().strip()
        if raw_label in ['true', 'real', 'supported', '1', 'yes']:
            label = 2
        elif raw_label in ['false', 'fake', 'refuted', '0', 'no']:
            label = 0
        else:
            label = 1
        
        # Tokenize
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
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'claim_text': claim
        }


def collate_fn(batch):
    """Handle images that might be None."""
    images = [item['image'] for item in batch]
    
    valid_images = [img for img in images if img is not None]
    if valid_images:
        pixel_values = torch.stack(valid_images)
    else:
        pixel_values = None
    
    return {
        'claim_input_ids': torch.stack([item['claim_input_ids'] for item in batch]),
        'claim_attention_mask': torch.stack([item['claim_attention_mask'] for item in batch]),
        'evidence_input_ids': torch.stack([item['evidence_input_ids'] for item in batch]),
        'evidence_attention_mask': torch.stack([item['evidence_attention_mask'] for item in batch]),
        'image': pixel_values,
        'label': torch.stack([item['label'] for item in batch]),
        'claim_text': [item['claim_text'] for item in batch]
    }


def evaluate_multimodal(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_texts = []
    
    with torch.no_grad():
        for batch in dataloader:
            claim_input_ids = batch['claim_input_ids'].to(device)
            claim_attention_mask = batch['claim_attention_mask'].to(device)
            evidence_input_ids = batch['evidence_input_ids'].to(device)
            evidence_attention_mask = batch['evidence_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            if batch['image'] is not None:
                image = batch['image'].to(device)
            else:
                image = None
            
            logits = model(
                claim_input_ids=claim_input_ids,
                claim_attention_mask=claim_attention_mask,
                evidence_input_ids=evidence_input_ids,
                evidence_attention_mask=evidence_attention_mask,
                image=image
            )
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(batch['claim_text'])
    
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    
    return {
        'accuracy': float(acc),
        'macro_f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'predictions': [int(p) for p in all_preds],
        'labels': [int(l) for l in all_labels],
        'claims': all_texts
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
    image_processor = ViTImageProcessor.from_pretrained(config['model']['vit'])
    
    checkpoint_path = os.path.join(config['output']['checkpoint_dir'], "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Train model first! {checkpoint_path}")
        return
    
    model = MultimodalVerifier(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model (best val F1: {checkpoint.get('best_f1', 0):.4f})")
    
    # ============================================================
    # EVALUATE ON FAKEHEALTH (or ReCOVery, or Honest OOC)
    # ============================================================
    
    demo_csv = "data/pipeline/fakehealth_for_demo.csv"
    image_dir = "data/raw/fakehealth/images"
    
    if not os.path.exists(demo_csv):
        print(f"\n❌ Demo CSV not found: {demo_csv}")
        print("You need to create a CSV from your Phase 1 data with columns:")
        print("  claim, evidence, label, image_path")
        print("\nExample row:")
        print('  "COVID vaccine causes infertility", "Study shows no link...", 0, "img_001.jpg"')
        print("\nRun: python create_demo_data.py")
        return
    
    print(f"\n{'='*60}")
    print("MULTIMODAL INFERENCE DEMO")
    print(f"{'='*60}")
    print(f"Dataset: {demo_csv}")
    
    dataset = MultimodalDemoDataset(
        demo_csv, image_dir, tokenizer, image_processor,
        max_claim_length=config['data']['max_claim_length'],
        max_evidence_length=config['data']['max_evidence_length']
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    print(f"Samples: {len(dataset)}")
    
    metrics = evaluate_multimodal(model, loader, device)
    
    print(f"\n{'='*60}")
    print("MULTIMODAL DEMO RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"Confusion Matrix:")
    print(f"            Pred:0   Pred:1   Pred:2")
    for i, row in enumerate(metrics['confusion_matrix']):
        name = ['Refuted ', 'NEI     ', 'Supported'][i]
        print(f"True {name}  {row[0]:6d}   {row[1]:6d}   {row[2]:6d}")
    
    out_path = os.path.join(config['output']['log_dir'], "multimodal_demo_results.json")
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to: {out_path}")
    
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*60}")
    for i in range(min(5, len(metrics['claims']))):
        label_names = {0: "REFUTED", 1: "NEI", 2: "SUPPORTED"}
        print(f"\nClaim: {metrics['claims'][i][:100]}...")
        print(f"True: {label_names[metrics['labels'][i]]} | Pred: {label_names[metrics['predictions'][i]]}")


if __name__ == "__main__":
    main()