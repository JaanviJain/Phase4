import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)


class VerifierTrainer:

    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader,
        device,
        class_weights=None
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.checkpoint_dir = config['output']['checkpoint_dir']
        self.log_dir = config['output']['log_dir']

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # ============================================================
        # Build optimizer with different learning rates per component
        # ============================================================

        param_groups = []

        # New parameters: classifier, evidence_encoder, no_image_token
        # NOTE: model.classifier is now nn.Sequential, but .parameters() works the same
        new_params = (
            list(model.classifier.parameters())
            + list(model.evidence_encoder.parameters())
        )
        new_params.append(model.no_image_token)

        param_groups.append({
            'params': new_params,
            'lr': config['training']['lr_classifier'],
            'name': 'new_layers'
        })

        # Fusion (if unfrozen for fine-tuning)
        if not config['model'].get('freeze_fusion', True):
            param_groups.append({
                'params': model.fusion.parameters(),
                'lr': config['training'].get('lr_fusion', 1e-5),
                'name': 'fusion'
            })

        # BioBERT (PARTIAL UNFREEZING — only top N layers are trainable)
        # model._freeze_backbones() already set requires_grad on each layer.
        # We only add the trainable parameters to the optimizer.
        if not config['model'].get('freeze_biobert', True):
            biobert_trainable = [p for p in model.biobert.parameters() if p.requires_grad]
            
            if biobert_trainable:
                param_groups.append({
                    'params': biobert_trainable,
                    'lr': config['training'].get('lr_biobert_top', 5e-6),
                    'name': 'biobert_top'
                })
                print(f"Added {len(biobert_trainable)} BioBERT trainable parameters "
                      f"(lr={config['training'].get('lr_biobert_top', 5e-6)})")
            else:
                print("WARNING: freeze_biobert=false but no trainable BioBERT params found.")

        # ============================================================
        # Optimizer
        # ============================================================

        self.optimizer = AdamW(
            param_groups,
            weight_decay=config['training']['weight_decay']
        )

        # Validate learning rates are numeric
        for group in self.optimizer.param_groups:
            if isinstance(group['lr'], (list, tuple)):
                print(
                    f"WARNING: Param group '{group.get('name', 'unknown')}' "
                    f"has lr as list: {group['lr']}"
                )
                group['lr'] = float(group['lr'][0])
                print(f"  Fixed to: {group['lr']}")
            elif not isinstance(group['lr'], (int, float)):
                print(
                    f"WARNING: Param group '{group.get('name', 'unknown')}' "
                    f"has invalid lr type: {type(group['lr'])}"
                )
                group['lr'] = float(group['lr'])

        # ============================================================
        # Learning rate scheduler
        # ============================================================

        total_steps = len(train_loader) * config['training']['epochs']
        warmup_steps = int(total_steps * config['training']['warmup_ratio'])

        for i, group in enumerate(self.optimizer.param_groups):
            print(
                f"  Param group {i} ({group.get('name', 'unknown')}): "
                f"lr={group['lr']}"
            )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # ============================================================
        # Loss function
        # ============================================================

        if class_weights is not None:
            print(f"Using class weights: {class_weights.cpu().numpy()}")
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        # ============================================================
        # AMP / Mixed Precision
        # ============================================================

        self.use_amp = (
            config['training'].get('use_amp', True)
            and device.type == 'cuda'
        )

        self.scaler = None

        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler("cuda")
                print("Using AMP with torch.amp.GradScaler")
            except (AttributeError, TypeError):
                try:
                    from torch.cuda.amp import GradScaler
                    self.scaler = GradScaler()
                    print("Using AMP with torch.cuda.amp.GradScaler")
                except Exception as e:
                    print(f"WARNING: Could not initialize GradScaler: {e}")
                    self.use_amp = False
                    self.scaler = None

        if not self.use_amp:
            print("AMP disabled. Training in full precision.")

        # ============================================================
        # Training state
        # ============================================================

        self.best_f1 = 0.0
        self.patience_counter = 0

    # ================================================================
    # Autocast context
    # ================================================================

    def _autocast_context(self):
        """Return the correct autocast context manager."""

        if not self.use_amp:
            from contextlib import nullcontext
            return nullcontext()

        try:
            return torch.autocast(device_type="cuda")
        except (AttributeError, TypeError):
            from torch.cuda.amp import autocast
            return autocast()

    # ================================================================
    # Train one epoch
    # ================================================================

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc="Training")

        for batch in pbar:

            claim_input_ids = batch['claim_input_ids'].to(self.device)
            claim_attention_mask = batch['claim_attention_mask'].to(self.device)
            evidence_input_ids = batch['evidence_input_ids'].to(self.device)
            evidence_attention_mask = batch['evidence_attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP
            with self._autocast_context():
                logits = self.model(
                    claim_input_ids=claim_input_ids,
                    claim_attention_mask=claim_attention_mask,
                    evidence_input_ids=evidence_input_ids,
                    evidence_attention_mask=evidence_attention_mask
                )
                loss = self.criterion(logits, labels)

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                self.optimizer.step()

            self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)

        return {
            'loss': avg_loss,
            'f1': f1,
            'accuracy': acc
        }

    # ================================================================
    # Evaluation
    # ================================================================

    def evaluate(self, loader, split_name="Val"):

        self.model.eval()

        all_preds = []
        all_labels = []
        all_sources = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {split_name}"):

                claim_input_ids = batch['claim_input_ids'].to(self.device)
                claim_attention_mask = batch['claim_attention_mask'].to(self.device)
                evidence_input_ids = batch['evidence_input_ids'].to(self.device)
                evidence_attention_mask = batch['evidence_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                with self._autocast_context():
                    logits = self.model(
                        claim_input_ids=claim_input_ids,
                        claim_attention_mask=claim_attention_mask,
                        evidence_input_ids=evidence_input_ids,
                        evidence_attention_mask=evidence_attention_mask
                    )
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_sources.extend(batch['source'])

        avg_loss = total_loss / len(loader)
        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)

        # Per-class breakdown
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            all_labels,
            all_preds,
            labels=[0, 1, 2],
            zero_division=0
        )

        # Per-source breakdown
        sources = sorted(set(all_sources))
        per_source = {}

        for src in sources:
            mask = np.array(all_sources) == src
            src_labels = np.array(all_labels)[mask]
            src_preds = np.array(all_preds)[mask]

            per_source[src] = {
                'accuracy': float(accuracy_score(src_labels, src_preds)),
                'f1': float(f1_score(src_labels, src_preds, average='macro', zero_division=0)),
                'support': int(mask.sum())
            }

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

        return {
            'loss': avg_loss,
            'f1': f1,
            'accuracy': acc,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support_per_class': support.tolist(),
            'per_source': per_source,
            'confusion_matrix': cm.tolist()
        }

    # ================================================================
    # Full training loop
    # ================================================================

    def train(self):

        patience = self.config['training']['early_stopping_patience']
        history = []

        for epoch in range(1, self.config['training']['epochs'] + 1):

            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{self.config['training']['epochs']}")
            print(f"{'=' * 60}")

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.evaluate(self.val_loader, "Val")

            # Print metrics
            print(
                f"\n[Train] "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"F1: {train_metrics['f1']:.4f} | "
                f"Acc: {train_metrics['accuracy']:.4f}"
            )

            print(
                f"[Val]   "
                f"Loss: {val_metrics['loss']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f}"
            )

            print(
                "Per-class F1 (Refuted/NEI/Supported): "
                f"{val_metrics['f1_per_class']}"
            )

            print("Per-source accuracy:")
            for src, vals in val_metrics['per_source'].items():
                print(
                    f"  {src:12s} | "
                    f"Acc: {vals['accuracy']:.4f} | "
                    f"F1: {vals['f1']:.4f} | "
                    f"N: {vals['support']}"
                )

            # Save history
            history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            })

            # Save best model
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.patience_counter = 0

                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    "best_model.pt"
                )

                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_f1': self.best_f1,
                        'config': self.config
                    },
                    checkpoint_path
                )

                print(
                    "*** NEW BEST MODEL SAVED "
                    f"(Val F1: {self.best_f1:.4f}) ***"
                )

            else:
                self.patience_counter += 1
                print(
                    f"Early stopping patience: "
                    f"{self.patience_counter}/{patience}"
                )

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        # Save training history
        history_path = os.path.join(
            self.log_dir,
            "training_history.json"
        )

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\nTraining history saved to: {history_path}")

        return history