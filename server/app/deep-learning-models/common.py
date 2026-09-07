# =============================================

# Copyright (c) 2026 Rudraksh Srivastava
# SPDX-License-Identifier: AGPL 3.0

# This file is part of Rudraksh Srivastava's
# Final Year Project at the University
# of Edinburgh.

# This file contains all the primitives that
# are used by all the models for trainin purposes

# =============================================

import numpy as np
import os 
import pickle
import time
import torch 

from dataclasses import dataclass, field
from torch import nn 
from typing import Callable, List, Dict, Optional, Type, Any, Tuple


class EarlyStopping:
    """
    Monitors a metric and signals when training should stop.

    Parameters
    ----------
    patience : int
        How many epochs without improvement to tolerate before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : {"min", "max"}
        Whether a lower ("min") or higher ("max") metric is better.

    Usage
    -----
    >>> es = EarlyStopping(patience=10)
    >>> if es(val_loss):
    ...     break          # inside epoch loop
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._counter = 0
        self._best: Optional[float] = None

    def __call__(self, metric: float) -> bool:
        """Return True if training should stop."""
        if self._best is None:
            self._best = metric
            return False

        improved = (
            metric < self._best - self.min_delta
            if self.mode == "min"
            else metric > self._best + self.min_delta
        )

        if improved:
            self._best = metric
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self.patience

    @property
    def counter(self) -> int:
        return self._counter

    def reset(self) -> None:
        self._counter = 0
        self._best = None


class ModelConfig:
    """
    Owns model construction and all checkpoint I/O.

    Keeps model hyper-params alongside the weights so a checkpoint is
    fully self-contained — you can reconstruct the model from disk without
    keeping the original constructor call anywhere in your notebook.

    Checkpoint format (torch.save dict)
    ------------------------------------
    {
        "model_class":      str,
        "model_config":     dict,          # kwargs forwarded to __init__
        "model_state":      OrderedDict,   # model.state_dict()
        "optimizer_state":  dict | None,   # optimizer.state_dict()
        "epoch":            int,
        "best_val_loss":    float,
        "best_val_acc":     float,
    }

    Saving the optimizer state means training is fully resumable, not just
    usable for inference — per PyTorch's official checkpoint tutorial.
    """

    def __init__(self, name: str,  model_class: Type[nn.Module], **model_kwargs):
        self.name = name 
        self.root_dir = rf"/kaggle/working/{self.name}_train"
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.model: nn.Module = model_class(**model_kwargs)

    # ------------------------------------------------------------------
    # Checkpoint — full training state (model + optimizer)
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        best_val_loss: float = float("inf"),
        best_val_acc: float = 0.0,
    ) -> None:
        """
        Save a resumable checkpoint.  Includes optimizer state so that
        momentum buffers and adaptive LR state (e.g. Adam's exp. moving
        averages) are preserved across interruptions.
        """
        torch.save(
            {
                "model_class":     self.model_class.__name__,
                "model_config":    self.model_kwargs,
                "model_state":     self.model.state_dict(),
                "optimizer_state": optimizer.state_dict() if optimizer else None,
                "epoch":           epoch,
                "best_val_loss":   best_val_loss,
                "best_val_acc":    best_val_acc,
            },
            path,
        )
        print(f"[ModelConfig] Checkpoint saved → {path}  (epoch {epoch})")

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load a checkpoint into *this* model (and optionally its optimizer).

        Returns the metadata dict (epoch, best_val_loss, best_val_acc) so
        TrainingInstance can restore its own bookkeeping state.
        """
        ckpt = torch.load(path, map_location=map_location)
        self.model.load_state_dict(ckpt["model_state"])
        if optimizer and ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"[ModelConfig] Checkpoint loaded ← {path}  (epoch {ckpt['epoch']})")
        return {
            "epoch":         ckpt["epoch"],
            "best_val_loss": ckpt["best_val_loss"],
            "best_val_acc":  ckpt["best_val_acc"],
        }

    # ------------------------------------------------------------------
    # Weights-only save — for inference / sharing without optimizer state
    # ------------------------------------------------------------------

    def save_weights(self, path: str) -> None:
        """Lightweight export: model config + weights only (no optimizer)."""
        path =  rf"{self.root_dir}/{path if path else f"{self.name}_weights.pt"}"
        torch.save(
            {
                "model_class":  self.model_class.__name__,
                "model_config": self.model_kwargs,
                "model_state":  self.model.state_dict(),
            },
            path,
        )
        print(f"[ModelConfig] Weights saved → {path}")

    @classmethod
    def from_weights(
        cls,
        model_class: Type[nn.Module],
        path: str,
        map_location: str = "cpu",
    ) -> "ModelConfig":
        """Reconstruct a ModelConfig + model purely from a weights file."""
        ckpt = torch.load(path, map_location=map_location)
        instance = cls(model_class, **ckpt["model_config"])
        instance.model.load_state_dict(ckpt["model_state"])
        print(f"[ModelConfig] Weights loaded ← {path}")
        return instance


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    duration_seconds: float


class TrainingInstance:
    """
    Orchestrates training epochs; delegates all persistence 
    to ModelConfig.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        train_fn: Callable,
        eval_fn: Callable,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler=None,
        early_stopping: Optional[EarlyStopping] = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.cfg = model_config
        self.name = self.cfg.name 
        self.root_dir = self.cfg.root_dir 
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.device = device

        # Bookkeeping
        self.history: List[EpochRecord] = []
        self.best_val_loss: float = float("inf")
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0
        self.confusion_matrices: List[np.ndarray] = []

        # Track total epochs run (useful when resuming)
        self._epochs_run: int = 0
        os.mkdir(self.root_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        n_epochs: int,
        checkpoint_path: str = "best_model.pt",
        resume_from: Optional[str] = None,
    ) -> None:
        """
        Train for n_epochs, saving a checkpoint whenever val_loss improves.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train for *from the current state*.
        checkpoint_path : str
            Where to write the best-model checkpoint.
        resume_from : str | None
            Path to an existing checkpoint to resume from before training.
            Restores model weights, optimizer state, and bookkeeping counters.
        """
        if resume_from:
            meta = self.cfg.load_checkpoint(
                resume_from, optimizer=self.optimizer, map_location=str(self.device)
            )
            self._epochs_run = meta["epoch"]
            self.best_val_loss = meta["best_val_loss"]
            self.best_val_acc = meta["best_val_acc"]

        self.cfg.model.to(self.device)

        for epoch in range(self._epochs_run + 1, self._epochs_run + n_epochs + 1):
            start = time.time()

            train_loss, train_acc = self.train_fn(
                self.cfg.model, self.train_loader,
                self.optimizer, self.criterion, self.device,
            )
            val_loss, val_acc = self.eval_fn(
                self.cfg.model, self.val_loader,
                self.criterion, self.device,
            )

            duration = time.time() - start
            lr = self.optimizer.param_groups[0]["lr"]

            if self.scheduler:
                self.scheduler.step(val_loss)

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=lr,
                duration_seconds=duration,
            )
            self.history.append(record)

            # Best-model checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.cfg.save_checkpoint(
                    f"{self.root_dir}/{checkpoint_path}",
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_val_loss=self.best_val_loss,
                    best_val_acc=self.best_val_acc,
                )

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
                f"LR: {lr:.2e} | {duration:.1f}s"
                + (f" [patience {self.early_stopping.counter}/{self.early_stopping.patience}]"
                   if self.early_stopping else "")
            )

            # Early stopping
            if self.early_stopping and self.early_stopping(val_loss):
                print(
                    f"[EarlyStopping] No improvement for {self.early_stopping.patience} "
                    f"epochs. Stopping at epoch {epoch}."
                )
                break

        self._epochs_run = epoch  # persist for potential future resume

    def save_history(self, path: str = "") -> None:
        """Pickle the training history + bookkeeping (no model / loaders)."""
        path =  rf"{self.root_dir}/{path if path else "training_history.pkl"}"
        payload = {
            "history":           self.history,
            "best_val_loss":     self.best_val_loss,
            "best_val_acc":      self.best_val_acc,
            "best_epoch":        self.best_epoch,
            "epochs_run":        self._epochs_run,
            "confusion_matrices":self.confusion_matrices,
            "model_class":       self.cfg.model_class.__name__,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[TrainingInstance] History saved → {path}")

    @classmethod
    def load_history(cls, path: str) -> "TrainingInstance":
        """
        Reconstruct a lightweight replay object from a saved history file.
        The returned instance has no model / loaders — only history data.
        Useful for plotting / auditing past runs.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls.__new__(cls)
        obj.history           = payload["history"]
        obj.best_val_loss     = payload["best_val_loss"]
        obj.best_val_acc      = payload["best_val_acc"]
        obj.best_epoch        = payload["best_epoch"]
        obj._epochs_run       = payload.get("epochs_run", len(payload["history"]))
        obj.confusion_matrices= payload["confusion_matrices"]
        # Dead references — not serialised
        obj.cfg = obj.train_fn = obj.eval_fn = None
        obj.train_loader = obj.val_loader = None
        obj.optimizer = obj.criterion = obj.scheduler = None
        obj.early_stopping = None
        return obj

    def summary(self) -> Dict[str, Any]:
        return {
            "best_epoch":          self.best_epoch,
            "best_val_loss":       self.best_val_loss,
            "best_val_acc":        self.best_val_acc,
            "total_epochs":        len(self.history),
            "total_time_seconds":  sum(r.duration_seconds for r in self.history),
        }

    def to_dict(self) -> Dict[str, List]:
        """Export history as plain dict — convenient for pandas / matplotlib."""
        return {
            "epoch":      [r.epoch      for r in self.history],
            "train_loss": [r.train_loss for r in self.history],
            "train_acc":  [r.train_acc  for r in self.history],
            "val_loss":   [r.val_loss   for r in self.history],
            "val_acc":    [r.val_acc    for r in self.history],
            "lr":         [r.lr         for r in self.history],
            "duration":   [r.duration_seconds for r in self.history],
        }

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X, y in loader:
        X = X.squeeze(0).to(device)   # (n_trials, 64, timesteps)
        y = y.squeeze(0).to(device)   # (n_trials,)

        optimizer.zero_grad()
        logits = model(X)              # (n_trials, n_classes)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X = X.squeeze(0).to(device)
            y = y.squeeze(0).to(device)

            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total

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