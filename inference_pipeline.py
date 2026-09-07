"""
End-to-end inference: Claim -> Retrieve Evidence (mock) -> Weight -> Verify
"""

import torch
import yaml
from transformers import AutoTokenizer
from src.model import MultimodalVerifier


def predict_claim(model, tokenizer, claim, evidence, device):
    """Run single claim through trained verifier."""
    model.eval()
    
    claim_enc = tokenizer(claim, max_length=128, padding='max_length', 
                         truncation=True, return_tensors='pt')
    ev_enc = tokenizer(evidence, max_length=512, padding='max_length', 
                      truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        logits = model(
            claim_input_ids=claim_enc['input_ids'].to(device),
            claim_attention_mask=claim_enc['attention_mask'].to(device),
            evidence_input_ids=ev_enc['input_ids'].to(device),
            evidence_attention_mask=ev_enc['attention_mask'].to(device)
        )
    
    pred = torch.argmax(logits, dim=-1).item()
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    
    labels = {0: "REFUTED", 1: "NOT ENOUGH INFO", 2: "SUPPORTED"}
    return labels[pred], probs


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(config['model']['biobert'])
    
    model = MultimodalVerifier(config).to(device)
    ckpt = torch.load(config['output']['checkpoint_dir'] + "/best_model.pt", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # Example
    claim = "COVID-19 vaccines cause infertility."
    evidence = "[TIER_B] Observational study of 45,000 participants found no association..."
    
    label, probs = predict_claim(model, tokenizer, claim, evidence, device)
    print(f"Claim: {claim}")
    print(f"Prediction: {label}")
    print(f"Probabilities: Refuted={probs[0]:.3f}, NEI={probs[1]:.3f}, Supported={probs[2]:.3f}")


if __name__ == "__main__":
    main()