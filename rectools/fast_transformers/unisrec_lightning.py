"""Lightning wrapper for UniSRec with configurable loss, optimizer, scheduler."""

import math
import typing as tp

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .unisrec_net import UniSRec

SUPPORTED_LOSSES = ("softmax", "BCE", "gBCE", "sampled_softmax")
SUPPORTED_OPTIMIZERS = ("adam", "adamw")
SUPPORTED_SCHEDULERS = (None, "cosine_warmup")


class UniSRecLightning(pl.LightningModule):
    """
    Thin Lightning wrapper reused across all training phases.

    Each phase creates a fresh ``UniSRecLightning`` with appropriate
    ``param_groups`` and ``use_id`` flag, sharing the same ``net`` instance.
    """

    def __init__(
        self,
        net: UniSRec,
        param_groups: tp.List[tp.Dict[str, tp.Any]],
        use_id: bool = False,
        loss: str = "softmax",
        n_negatives: tp.Optional[int] = None,
        gbce_t: float = 0.2,
        optimizer: str = "adamw",
        scheduler: tp.Optional[str] = None,
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.1,
        total_steps: tp.Optional[int] = None,
    ) -> None:
        super().__init__()
        self.net = net
        self._param_groups = param_groups
        self.use_id = use_id
        self.loss_name = loss
        self.n_negatives = n_negatives
        self.gbce_t = gbce_t
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.total_steps = total_steps

    # ── helpers ──

    def _get_item_embs(self, item_ids: torch.Tensor) -> torch.Tensor:
        if self.use_id:
            return self.net.item_emb(item_ids)
        return self.net._adapt_score(self.net._sample_frozen(item_ids))

    def _get_all_embs(self) -> torch.Tensor:
        if self.use_id:
            return self.net.item_emb.weight
        return self.net.project_all()

    def _get_pos_neg_logits(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute (B, L, 1+N) logits where index 0 = positive."""
        emb_pos = self._get_item_embs(labels)
        logits_pos = (hidden * emb_pos).sum(dim=-1)

        emb_neg = self._get_item_embs(negatives)
        logits_neg = torch.matmul(
            hidden.unsqueeze(2),
            emb_neg.transpose(2, 3),
        ).squeeze(2)

        return torch.cat([logits_pos.unsqueeze(-1), logits_neg], dim=-1)

    # ── losses ──

    def _calc_loss(
        self,
        hidden: torch.Tensor,
        batch: tp.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        labels = batch["y"]
        has_neg = "negatives" in batch

        if self.loss_name == "softmax" and not has_neg:
            return self._full_softmax_loss(hidden, labels)

        if self.loss_name == "softmax" and has_neg:
            # full softmax even if negatives are available
            return self._full_softmax_loss(hidden, labels)

        if not has_neg:
            raise ValueError(f"Loss '{self.loss_name}' requires negatives but batch has none")

        logits = self._get_pos_neg_logits(hidden, labels, batch["negatives"])
        mask = labels != 0

        if self.loss_name == "sampled_softmax":
            return self._sampled_softmax_loss(logits, mask)
        if self.loss_name == "BCE":
            return self._bce_loss(logits, mask)
        if self.loss_name == "gBCE":
            return self._gbce_loss(logits, mask)

        raise ValueError(f"Unknown loss: {self.loss_name}")

    def _full_softmax_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        all_emb = self._get_all_embs()
        logits = hidden @ all_emb.T
        logits[:, :, 0] = float("-inf")

        targets = labels.clone()
        targets[targets == 0] = -100
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

    def _sampled_softmax_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sampled softmax: positive at index 0, swap to index 1 so index 0 can be ignored."""
        logits = logits.clone()
        logits[:, :, [0, 1]] = logits[:, :, [1, 0]]
        targets = mask.long()  # 1 where non-padding, 0 where padding
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=0,
        )

    def _bce_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = torch.zeros_like(logits)
        target[:, :, 0] = 1.0
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = loss.mean(-1) * mask
        return loss.sum() / mask.sum().clamp(min=1)

    def _gbce_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n_items = self.net.n_items
        n_neg = self.n_negatives or logits.size(-1) - 1
        alpha = n_neg / max(n_items - 1, 1)
        beta = alpha * (self.gbce_t * (1 - 1 / alpha) + 1 / alpha)

        dtype = torch.float64
        pos_logits = logits[:, :, 0:1].to(dtype)
        neg_logits = logits[:, :, 1:]

        eps = 1e-10
        pos_probs = torch.clamp(torch.sigmoid(pos_logits), eps, 1 - eps)
        pos_adjusted = torch.clamp(pos_probs.pow(-beta), 1 + eps, torch.finfo(dtype).max)
        pos_adjusted = torch.clamp(1.0 / (pos_adjusted - 1), eps, torch.finfo(dtype).max)
        pos_transformed = torch.log(pos_adjusted).to(logits.dtype)

        adjusted_logits = torch.cat([pos_transformed, neg_logits], dim=-1)
        return self._bce_loss(adjusted_logits, mask)

    # ── training / validation ──

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        hidden = self.net(batch["x"], use_id=self.use_id)
        loss = self._calc_loss(hidden, batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        hidden = self.net(batch["x"], use_id=self.use_id)
        # Validation batch has y of shape (B, 1) -- take last hidden position only
        hidden = hidden[:, -1:, :]
        loss = self._calc_loss(hidden, batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # ── optimizer / scheduler ──

    def configure_optimizers(self) -> tp.Any:
        if self.optimizer_name == "adamw":
            opt = torch.optim.AdamW(self._param_groups)
        elif self.optimizer_name == "adam":
            opt = torch.optim.Adam(self._param_groups)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.scheduler_name is None:
            return opt

        if self.scheduler_name == "cosine_warmup":
            total = self.total_steps or 1
            warmup = int(total * self.warmup_ratio)
            scheduler = _cosine_warmup_scheduler(opt, warmup, total, self.min_lr_ratio)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        raise ValueError(f"Unknown scheduler: {self.scheduler_name}")


def _cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
