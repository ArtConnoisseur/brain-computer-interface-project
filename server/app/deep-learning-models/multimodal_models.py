# =============================================

# Copyright (c) 2026 Rudraksh Srivastava
# SPDX-License-Identifier: AGPL 3.0

# This file is part of Rudraksh Srivastava's
# Final Year Project at the University
# of Edinburgh.

# This file contains the architectural 
# definitions for the 

# =============================================


import timm 
import torch 

from torch.utils.data import DataLoader
from torch import nn 
from typing import Tuple

from .common import (
    TrainingInstance,
    ModelConfig,
    EarlyStopping,
)
from .models import get_eegnet_backbone, ClassifierHead

class ProjectionHead(nn.Module):
    def __init__(self,
        input_size, 
        d_model=128
    ):
        super().__init__()
        self.projection_head = nn.Linear(input_size, d_model)

    def forward(self, x):
        return self.projection_head(x)

class TimeBranch(nn.Module):
    def __init__(self, 
        n_chans=64, 
        n_times=577,
        hidden_size=128, 
        num_layers=2, 
        drop_prob=0.5, 
        kernel_length=64
    ):
        super().__init__()
        self.backbone, feat_size = get_eegnet_backbone(n_chans, n_times, drop_prob, kernel_length=kernel_length)
        
        self.lstm = nn.LSTM(
            input_size=feat_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_prob if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(drop_prob)

        self.proj = ProjectionHead(input_size=hidden_size)

    def forward(self, x):
        # x: (batch, n_chans, n_times)
        features = self.backbone(x).flatten(1).unsqueeze(1)  # (batch, 1, feat_size)
        out, _ = self.lstm(features)
        out = self.dropout(out[:, -1, :])
        return self.proj(out) 
    
class FreqBranch(nn.Module):
    def __init__(self, 
        n_chans=64, 
        n_times=129, 
        drop_prob=0.5, 
        kernel_length=64
    ):
        super().__init__()
        self.backbone, feat_size = get_eegnet_backbone(n_chans, n_times, drop_prob, kernel_length=kernel_length)
        self.dropout = nn.Dropout(drop_prob)

        self.proj = ProjectionHead(input_size=feat_size)

    def forward(self, x):
        # x: (batch, n_chans, n_times)
        features = self.backbone(x).flatten(1)
        out = self.dropout(features)
        return self.proj(out) 
    
class ImageBranch(nn.Module):
    def __init__(self, 
        n_chans=64,
        freq_bins=20, 
        n_times=577,
        drop_prob=0.5,
        d_model=128 
    ):
        super().__init__()

        self.shallow_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32), 
            nn.ELU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(drop_prob),
            nn.AvgPool2d(kernel_size=(2, 4)) # -> (64, 10, 144)
        )

        # ViT-Tiny expects 3-channel input, so we use in_chans=64 from CNN output
        self.vit = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,       # too different from ImageNet to benefit
            img_size=(10, 144),     # match CNN output spatial dims
            in_chans=64,            # match CNN out_channels
            num_classes=0           # removes classification head, returns embedding
        )

        self.proj = ProjectionHead(input_size=self.vit.embed_dim)

    def forward(self, x):
        x = self.shallow_cnn(x)
        x = self.vit(x)
        return self.proj(x)


class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_branch = TimeBranch()
        self.freq_branch = FreqBranch()
        self.image_branch = ImageBranch()

        d_model = 128 # constant 
        self.classification_head = ClassifierHead(d_model*3)

    def forward(self, X_time, X_freq, X_image):
        batch, epochs = X_time.shape[:2]
        X_time = X_time.view(batch * epochs, *X_time.shape[2:])
        X_freq = X_freq.view(batch * epochs, *X_freq.shape[2:])
        X_image = X_image.view(batch * epochs, 1, *X_image.shape[2:])  # unsqueeze channel dim for CNN
        
        time_feat = self.time_branch(X_time)
        freq_feat = self.freq_branch(X_freq)
        image_feat = self.image_branch(X_image)

        concat = torch.cat([time_feat, freq_feat, image_feat], dim=-1)

        return self.classification_head(concat)


def train_epoch_multimodal(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X_time, X_image, X_freq, y in loader:
        batch, epochs = X_time.shape[:2]
        X_time  = X_time.view(batch * epochs, *X_time.shape[2:]).to(device)
        X_freq  = X_freq.view(batch * epochs, *X_freq.shape[2:]).to(device)
        X_image = X_image.view(batch * epochs, 1, *X_image.shape[2:]).to(device)
        y       = y.view(batch * epochs).to(device)

        optimizer.zero_grad()
        logits = model(X_time, X_freq, X_image)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


def eval_epoch_multimodal(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for X_time, X_image, X_freq, y in loader:
            batch, epochs = X_time.shape[:2]
            X_time  = X_time.view(batch * epochs, *X_time.shape[2:]).to(device)
            X_freq  = X_freq.view(batch * epochs, *X_freq.shape[2:]).to(device)
            X_image = X_image.view(batch * epochs, 1, *X_image.shape[2:]).to(device)
            y       = y.view(batch * epochs).to(device)

            logits = model(X_time, X_freq, X_image)
            loss   = criterion(logits, y)

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total

def train_model(
        name: str,  
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_channels: int = 64, 
        n_times: int = 577, 
        n_classes: int = 5,
        n_epochs: int = 80
) -> Tuple[TrainingInstance, ModelConfig]:
    # Warp model in ModelConfig 
    cfg = ModelConfig(
        name,
        model, 
        n_chans=n_channels,
        n_outputs=n_classes,          # placeholder, we strip this
        n_times=n_times,
        # kernel_length=64,     # 0.5s at 128Hz
        drop_prob=0.5,
        # final_conv_length='auto',
    )   # add kwargs if model takes any: ModelConfig(CNNLSTM, n_classes=5)


    weights = torch.tensor([1/14] + [1/7.5]*4, dtype=torch.float32)
    weights /= weights.sum()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(cfg.model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Optional: early stopping — stops if val_loss doesn't improve for 15 epochs.
    es = EarlyStopping(patience=15, min_delta=1e-4, mode="min")

    # Create the training instance.
    runner = TrainingInstance(
        model_config   = cfg,
        train_fn       = train_epoch_multimodal,
        eval_fn        = eval_epoch_multimodal,
        train_loader   = train_loader,
        val_loader     = val_loader,
        optimizer      = optimizer,
        criterion      = criterion,
        scheduler      = scheduler,
        # early_stopping = es,
        device         = device,
    )

    # Train.
    runner.run(n_epochs=n_epochs, checkpoint_path="best_cnnlstm.pt")

    # Resume later (optimizer state + epoch count are fully restored).
    # runner.run(n_epochs=20, checkpoint_path="best_cnnlstm.pt",
    #            resume_from="best_cnnlstm.pt")

    # Save / reload history for plotting without keeping the model in memory.
    runner.save_history("cnnlstm_history.pkl")

    # Load model for inference only (weights file, no optimizer state).
    cfg.save_weights("cnnlstm_weights_only.pt")

    return runner, cfg 