# =============================================

# Copyright (c) 2026 Rudraksh Srivastava
# SPDX-License-Identifier: AGPL 3.0

# This file is part of Rudraksh Srivastava's
# Final Year Project at the University
# of Edinburgh.

# This file contains all the primitives that
# that define actual model architectures. Note
# that multimodal models are in the alternate 
# file.

# =============================================

import torch 

from torch.utils.data import DataLoader
from torch import nn 
from braindecode.models import (
    EEGNet, EEGConformer, ATCNet, EEGTCNet, FBCNet,
    ShallowFBCSPNet, Deep4Net, EEGNeX, CTNet
)
from typing import Tuple

from .common import (
    TrainingInstance,
    train_epoch,
    eval_epoch,
    ModelConfig,
    EarlyStopping,
)


class ClassifierHead(nn.Module):
    def __init__(self, input_size, n_classes, drop_prob=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, 64),
            nn.ELU(),
            nn.Dropout(drop_prob),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.block(x)

class CNNBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.spatial = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=(n_channels, 1), padding=(0, 32), bias=False),
            nn.BatchNorm2d(n_channels),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.6)
        )

        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=32, padding=16, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.6),

            nn.Conv1d(128, 128, kernel_size=32, padding=16, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.6)
        )

    def forward(self, x):
        x = x.unsqueeze(1)       # (batch, 1, n_channels, timesteps)
        x = self.spatial(x)      # (batch, n_channels, 1, timesteps')
        x = x.squeeze(2)         # (batch, n_channels, timesteps')
        x = self.temporal(x)     # (batch, 128, timesteps'')
        return x

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.65):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,    
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,         
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False       
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)      
        out = out[:, -1, :]    
        return self.dropout(out)
    
class CNNLSTM(nn.Module):
    def __init__(self, n_chans=64, n_times=577, n_outputs=5, kernel_length=64, drop_prob=0.5):
        super().__init__() 
        self.cnn_block = CNNBlock(n_channels=n_chans)
        self.lstm_block = LSTMBlock(input_size=n_chans)

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(64, n_outputs)
        )

    def forward(self, x):
        # x = self.cnn_block(x)
        # x = x.permute(0, 2, 1)
        # x = self.lstm_block(x)
        # logits = self.classifier(x)

        cnn_out = self.cnn_block(x)
        cnn_out = cnn_out.mean(dim=-1)

        lstm_in = x.permute(0, 2, 1)
        lstm_out = self.lstm_block(lstm_in)

        combined = torch.cat([cnn_out, lstm_out], dim=-1)  # (batch, 256)
        logits = self.classifier(combined)

        return logits

# Just checking how the combination models 
# perform against my custom CNN Combination 
# models.


def get_eegnet_backbone(n_chans=64, n_times=577, drop_prob=0.5, kernel_length=64):
    """EEGNet with classifier stripped — used as CNN frontend."""
    model = EEGNet(
        n_chans=n_chans,
        n_outputs=2,          # placeholder, we strip this
        n_times=n_times,
        kernel_length=kernel_length,     # 0.5s at 128Hz
        drop_prob=drop_prob,
        final_conv_length='auto'
    )
    # Strip final linear layer — EEGNet is nn.Sequential
    backbone = nn.Sequential(*list(model.children())[:-1])
    
    # Infer output feature size
    with torch.no_grad():
        dummy = torch.zeros(1, n_chans, n_times)
        out = backbone(dummy)
        feature_size = out.flatten(1).shape[1]
    
    return backbone, feature_size

class EEGNetLSTM(nn.Module):
    def __init__(self, n_chans=64, n_times=577, n_outputs=5,
                 hidden_size=128, num_layers=2, drop_prob=0.5, kernel_length=64):
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
        self.classifier = ClassifierHead(input_size=hidden_size, n_classes=n_outputs, dropout=0.5)

    def forward(self, x):
        # x: (batch, n_chans, n_times)
        features = self.backbone(x).flatten(1).unsqueeze(1)  # (batch, 1, feat_size)
        out, _ = self.lstm(features)
        out = self.dropout(out[:, -1, :])
        return self.classifier(out)


class EEGNetGRU(nn.Module):
    def __init__(self, n_chans=64, n_times=577, n_outputs=5,
                 hidden_size=128, num_layers=2, drop_prob=0.5, kernel_length=64):
        super().__init__()
        self.backbone, feat_size = get_eegnet_backbone(n_chans, n_times, drop_prob, kernel_length=kernel_length)
        
        self.gru = nn.GRU(
            input_size=feat_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_prob if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(drop_prob)
        self.classifier = ClassifierHead(input_size=hidden_size, n_classes=n_outputs, dropout=0.5)

    def forward(self, x):
        features = self.backbone(x).flatten(1).unsqueeze(1)  # (batch, 1, feat_size)
        out, _ = self.gru(features)
        out = self.dropout(out[:, -1, :])
        return self.classifier(out)

class EEGNetLSTMRes(nn.Module):
    def __init__(self, n_chans=64, n_times=577, n_outputs=5,
                 hidden_size=128, num_layers=2, drop_prob=0.5, kernel_length=64):
        super().__init__()
        self.backbone, feat_size = get_eegnet_backbone(n_chans, n_times, drop_prob, kernel_length=kernel_length)

        self.lstm_1 = nn.LSTM(
            input_size=feat_size,
            hidden_size=64,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_prob if num_layers > 1 else 0,
            bidirectional=False
        )

        self.lstm_2 = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_prob if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout  = nn.Dropout(drop_prob)
        self.classifier = ClassifierHead(input_size=hidden_size, n_classes=n_outputs, dropout=drop_prob)

    def forward(self, x):
        # x: (batch, n_chans, n_times)
        features = self.backbone(x).flatten(1).unsqueeze(1)           # (batch, feat_size, timesteps)

        out, _ = self.lstm_1(features)        # (batch, timesteps, 64)
        residual = out                        # save for residual

        out, _ = self.lstm_2(out)             # (batch, timesteps, hidden_size)
        # out = out + residual[:, :, :out.shape[2]] if out.shape == residual.shape else out  # residual if dims match

        out = self.dropout(out[:, -1, :])     # (batch, hidden_size)
        return self.classifier(out)

class EEGNetTransformer(nn.Module):
    def __init__(self, n_chans=64, n_times=577, n_outputs=5,
                 nhead=4, num_layers=2, dim_feedforward=256, drop_prob=0.5, kernel_length=64):
        super().__init__()
        self.backbone, feat_size = get_eegnet_backbone(n_chans, n_times, drop_prob, kernel_length=kernel_length)
        
        # feat_size must be divisible by nhead
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(drop_prob)
        self.classifier = ClassifierHead(input_size=feat_size, n_classes=n_outputs, dropout=0.5)

    def forward(self, x):
        features = self.backbone(x).flatten(1).unsqueeze(1)  # (batch, 1, feat_size)
        out = self.transformer(features)
        out = self.dropout(out[:, -1, :])
        return self.classifier(out)

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
        train_fn       = train_epoch,
        eval_fn        = eval_epoch,
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

def train_config(
        cfg: ModelConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 80
) -> Tuple[TrainingInstance]:
    # Training hyperparams
    weights = torch.tensor([1/14] + [1/7.5]*4, dtype=torch.float32)
    weights /= weights.sum()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(cfg.model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Create the training instance.
    runner = TrainingInstance(
        model_config   = cfg,
        train_fn       = train_epoch,
        eval_fn        = eval_epoch,
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

# Model training configs 

N_CLASSES = 5
N_CHANS = 64
N_TIMES = 577
SFREQ = 128 

model_configs = [
    {
        "name": "Conformer",
        "class": EEGConformer,
        "kwargs": {
            "n_outputs": N_CLASSES,
            "n_chans": N_CHANS,
            "n_times": N_TIMES,
            "drop_prob": 0.5,
        }
    },
    {
        "name": "ATCNet",
        "class": ATCNet,
        "kwargs": {
            "n_outputs": N_CLASSES,
            "n_chans": N_CHANS,
            "sfreq": SFREQ,
            "input_window_seconds": N_TIMES / SFREQ,
        }
    },
    {
        "name": "EEGTCNet",
        "class": EEGTCNet,
        "kwargs": {
            "n_outputs": N_CLASSES,
            "n_chans": N_CHANS,
            "n_times": N_TIMES,
            "drop_prob": 0.5,
        }
    },
    {
        "name": "FBCNet",
        "class": FBCNet,
        "kwargs": {
            "n_outputs": N_CLASSES,
            "n_chans": N_CHANS,
            "n_times": N_TIMES,
            "sfreq": SFREQ,
        }
    },
    {
        "name": "ShallowFBC",
        "class": ShallowFBCSPNet,
        "kwargs": {
            "n_outputs": N_CLASSES,
            "n_chans": N_CHANS,
            "n_times": N_TIMES,
            "final_conv_length": "auto",
        }
    },
    {
        "name": "Deep4Net",
        "class": Deep4Net,
        "kwargs": {
            "n_outputs": N_CLASSES,
            "n_chans": N_CHANS,
            "n_times": N_TIMES,
            "final_conv_length": "auto",
        }
    },
    {
        "name": "EEGNeX",
        "class": EEGNeX,
        "kwargs": {
            "n_outputs": N_CLASSES,
            "n_chans": N_CHANS,
            "n_times": N_TIMES,
        }
    },
    {
        "name": "CTNet",
        "class": CTNet,
        "kwargs": {
            "n_outputs": N_CLASSES,
            "n_chans": N_CHANS,
            "n_times": N_TIMES,
        }
    },
]