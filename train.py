import os
import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model import MultimodalVerifier
from src.dataset import VerifierDataset
from src.trainer import VerifierTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    counts = df['label'].value_counts().sort_index()
    full_counts = [counts.get(i, 1) for i in range(3)]
    total = sum(full_counts)
    weights = [total / (3.0 * c) for c in full_counts]
    weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Class counts: {full_counts}")
    print(f"Class weights: {weights.numpy()}")
    return weights


def main():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['training']['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        if config['data']['max_evidence_length'] > 512:
            print("WARNING: Evidence length > 512. If OOM, reduce batch_size to 4.")
    
    print(f"\nLoading tokenizer: {config['model']['biobert']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['biobert'],
        use_fast=False
    )
    
    train_csv = os.path.join(config['data']['processed_dir'], "combined_train.csv")
    val_csv = os.path.join(config['data']['processed_dir'], "combined_val.csv")
    
    if not os.path.exists(train_csv):
        print(f"\nERROR: Preprocessed data not found: {train_csv}")
        print("Run this first: python preprocess.py")
        return
    
    print("\nLoading datasets...")
    train_dataset = VerifierDataset(
        train_csv, tokenizer,
        max_claim_length=config['data']['max_claim_length'],
        max_evidence_length=config['data']['max_evidence_length']
    )
    val_dataset = VerifierDataset(
        val_csv, tokenizer,
        max_claim_length=config['data']['max_claim_length'],
        max_evidence_length=config['data']['max_evidence_length']
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    class_weights = compute_class_weights(train_csv)
    
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    model = MultimodalVerifier(config)
    
    trainer = VerifierTrainer(model, config, train_loader, val_loader, device, class_weights)
    
    print("\n" + "="*60)
    print("STARTING PHASE 4 TRAINING")
    print("="*60)
    history = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation F1: {trainer.best_f1:.4f}")
    print(f"Best model saved to: {config['output']['checkpoint_dir']}\\best_model.pt")


if __name__ == "__main__":
    main()