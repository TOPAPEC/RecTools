"""PyTorch Lightning wrapper for FlatSASRec."""

import typing as tp

import pytorch_lightning as pl
import torch
from torch import nn

from .net import FlatSASRec


class FlatSASRecLightning(pl.LightningModule):
    """Lightning module wrapping FlatSASRec with softmax / BCE losses."""

    SUPPORTED_LOSSES = ("softmax", "BCE")

    def __init__(
        self,
        net: FlatSASRec,
        lr: float = 1e-3,
        loss: str = "softmax",
        n_negatives: int = 1,
    ) -> None:
        super().__init__()
        self.net = net
        self.lr = lr
        self.loss_name = loss
        self.n_negatives = n_negatives

        if loss == "softmax":
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        elif loss == "BCE":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss: {loss}. Use one of {self.SUPPORTED_LOSSES}")

    def on_train_start(self) -> None:
        for p in self.net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self.net(batch)
        y = batch["y"]  # (B, L)
        mask = y != FlatSASRec.PADDING_IDX  # ignore padding positions

        if self.loss_name == "softmax":
            # logits: (B, L, n_items) — full catalog
            # targets need to be 0-indexed item ids (subtract 1 since item ids start from 1)
            targets = (
                y - 1
            )  # shift to 0-based for CrossEntropyLoss; padding (0) becomes -1 -> ignore_index=0 won't work
            # Actually, we set ignore_index=0 but padding maps to -1.
            # Let's use a different approach: set padding targets to 0 and use ignore_index=0
            targets = y.clone()
            targets[~mask] = 0
            # For CE loss: targets should index into logits dim=-1 which is [0..n_items-1]
            # Our item ids in y are 1..n_items, so subtract 1
            targets = targets - 1
            targets[~mask] = -100  # PyTorch ignore index
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            # BCE: logits shape (B, L, 1+N)
            B, L, C = logits.shape
            labels = torch.zeros(B, L, C, device=logits.device)
            labels[:, :, 0] = 1.0  # first column is positive
            loss_per_elem = self.loss_fn(logits, labels)  # (B, L, C)
            # Mask out padding positions
            loss_per_elem = loss_per_elem * mask.unsqueeze(-1).float()
            loss = loss_per_elem.sum() / mask.sum().clamp(min=1) / C

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98))
