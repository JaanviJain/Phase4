import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model import MultimodalVerifier
from src.dataset import VerifierDataset
from src.trainer import VerifierTrainer
import json


def main():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['biobert'],
        use_fast=False
    )
    
    test_csv = os.path.join(config['data']['processed_dir'], "combined_test.csv")
    if not os.path.exists(test_csv):
        print(f"ERROR: {test_csv} not found. Run preprocess.py first.")
        return
    
    test_dataset = VerifierDataset(
        test_csv, tokenizer,
        max_claim_length=config['data']['max_claim_length'],
        max_evidence_length=config['data']['max_evidence_length']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=0
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    checkpoint_path = os.path.join(config['output']['checkpoint_dir'], "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Run train.py first.")
        return
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MultimodalVerifier(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')} (Best Val F1: {checkpoint.get('best_f1', 0):.4f})")
    
    trainer = VerifierTrainer(model, config, None, None, device)
    metrics = trainer.evaluate(test_loader, "TEST")
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Macro F1:     {metrics['f1']:.4f}")
    print(f"\nPer-class breakdown (0=Refuted, 1=NEI, 2=Supported):")
    print(f"  Precision:  {metrics['precision_per_class']}")
    print(f"  Recall:     {metrics['recall_per_class']}")
    print(f"  F1:         {metrics['f1_per_class']}")
    print(f"  Support:    {metrics['support_per_class']}")
    
    print(f"\nPer-source accuracy:")
    for src, vals in metrics['per_source'].items():
        print(f"  {src:12s} | Acc: {vals['accuracy']:.4f} | F1: {vals['f1']:.4f} | N: {vals['support']}")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              0(Ref)  1(NEI)  2(Sup)")
    for i, row in enumerate(metrics['confusion_matrix']):
        name = ['Refuted ', 'NEI     ', 'Support'][i]
        print(f"True {name}  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")
    
    out_path = os.path.join(config['output']['log_dir'], "test_results.json")
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()