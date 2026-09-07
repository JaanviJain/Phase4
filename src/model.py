import os
import torch
import torch.nn as nn
from transformers import AutoModel


class MultimodalVerifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        print("Loading BioBERT...")
        self.biobert = AutoModel.from_pretrained(config['model']['biobert'])
        print("Loading ViT...")
        self.vit = AutoModel.from_pretrained(config['model']['vit'])
        
        biobert_dim = self.biobert.config.hidden_size
        vit_dim = self.vit.config.hidden_size
        
        fusion_type = config['model'].get('fusion_type', 'concat')
        if fusion_type == 'concat':
            fusion_input_dim = biobert_dim + vit_dim
        else:
            fusion_input_dim = biobert_dim
            
        self.fusion = nn.Linear(fusion_input_dim, biobert_dim)
        self._load_phase1_fusion(config['model']['phase1_fusion_path'])
        
        self.no_image_token = nn.Parameter(torch.randn(biobert_dim))
        self.evidence_encoder = nn.Linear(biobert_dim, biobert_dim)
        
        # DEEPER CLASSIFIER WITH DROPOUT
        classifier_dim = biobert_dim * 2
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(classifier_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3)
        )
        
        self.dropout = nn.Dropout(0.1)
        
        self._freeze_backbones(config['model'])
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        
    def _load_phase1_fusion(self, checkpoint_path):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"WARNING: Phase 1 fusion checkpoint NOT FOUND at: {checkpoint_path}")
            return
            
        print(f"Loading Phase 1 fusion from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(state, dict):
            if 'state_dict' in state:
                state = state['state_dict']
            elif 'model_state_dict' in state:
                state = state['model_state_dict']
            
            if any(k.startswith('fusion.') for k in state.keys()):
                state = {k.replace('fusion.', ''): v for k, v in state.items() if k.startswith('fusion.')}
        
        try:
            self.fusion.load_state_dict(state, strict=False)
            print("Fusion layer loaded successfully from Phase 1.")
        except Exception as e:
            print(f"ERROR loading fusion weights: {e}")
            
    def _freeze_backbones(self, model_config):
        # PARTIAL BIOBERT UNFREEZING
        if model_config.get('freeze_biobert', True):
            for param in self.biobert.parameters():
                param.requires_grad = False
            print("BioBERT: FULLY FROZEN")
        else:
            n_unfreeze = model_config.get('biobert_unfreeze_top_n', 4)
            n_layers = len(self.biobert.encoder.layer)
            
            for param in self.biobert.embeddings.parameters():
                param.requires_grad = False
            
            for i, layer in enumerate(self.biobert.encoder.layer):
                if i < (n_layers - n_unfreeze):
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
            
            print(f"BioBERT: Bottom {n_layers - n_unfreeze} layers FROZEN, Top {n_unfreeze} layers UNFROZEN")
            
        if model_config.get('freeze_vit', True):
            for param in self.vit.parameters():
                param.requires_grad = False
            print("ViT: FROZEN")
        else:
            print("ViT: UNFROZEN")
            
        if model_config.get('freeze_fusion', True):
            for param in self.fusion.parameters():
                param.requires_grad = False
            print("Fusion: FROZEN")
        else:
            print("Fusion: UNFROZEN")
            
    def forward(self, claim_input_ids, claim_attention_mask,
                evidence_input_ids, evidence_attention_mask,
                image=None):
        batch_size = claim_input_ids.size(0)
        
        text_outputs = self.biobert(
            input_ids=claim_input_ids,
            attention_mask=claim_attention_mask
        )
        text_emb = text_outputs.last_hidden_state[:, 0, :]
        
        if image is not None:
            vit_outputs = self.vit(pixel_values=image)
            img_emb = vit_outputs.last_hidden_state[:, 0, :]
        else:
            img_emb = self.no_image_token.unsqueeze(0).expand(batch_size, -1)
        
        if self.config['model'].get('fusion_type', 'concat') == 'concat':
            fused_input = torch.cat([text_emb, img_emb], dim=-1)
        else:
            fused_input = text_emb + img_emb
            
        fused = self.fusion(fused_input)
        fused = torch.relu(fused)
        fused = self.dropout(fused)
        
        ev_outputs = self.biobert(
            input_ids=evidence_input_ids,
            attention_mask=evidence_attention_mask
        )
        ev_emb = ev_outputs.last_hidden_state[:, 0, :]
        ev_encoded = self.evidence_encoder(ev_emb)
        ev_encoded = torch.relu(ev_encoded)
        ev_encoded = self.dropout(ev_encoded)
        
        combined = torch.cat([fused, ev_encoded], dim=-1)
        logits = self.classifier(combined)
        
        return logits