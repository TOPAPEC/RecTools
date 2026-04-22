"""Lightning wrapper for UniSRec: supports full-softmax and sampled CE loss."""

import typing as tp

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .unisrec_net import UniSRec


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
    ) -> None:
        super().__init__()
        self.net = net
        self._param_groups = param_groups
        self.use_id = use_id

    # ── helpers ──

    def _get_item_embs(self, item_ids: torch.Tensor) -> torch.Tensor:
        if self.use_id:
            return self.net.item_emb(item_ids)
        return self.net._adapt_score(self.net._sample_frozen(item_ids))

    # ── training step ──

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch["x"]
        labels = batch["y"]
        hidden = self.net(input_ids, use_id=self.use_id)  # (B, L, D)

        if "negatives" in batch:
            loss = self._sampled_ce_loss(hidden, labels, batch["negatives"])
        else:
            loss = self._full_softmax_loss(hidden, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _full_softmax_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.use_id:
            all_emb = self.net.item_emb.weight  # (n_items+1, D)
        else:
            all_emb = self.net.project_all()     # (n_items+1, D)

        logits = hidden @ all_emb.T              # (B, L, n_items+1)
        logits[:, :, 0] = float("-inf")          # never predict padding

        targets = labels.clone()
        targets[targets == 0] = -100             # padding → ignore
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

    def _sampled_ce_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        emb_pos = self._get_item_embs(labels)                  # (B, L, D)
        logits_pos = (hidden * emb_pos).sum(dim=-1)             # (B, L)

        emb_neg = self._get_item_embs(negatives)               # (B, L, N, D)
        logits_neg = torch.matmul(                              # (B, L, N)
            hidden.unsqueeze(2), emb_neg.transpose(2, 3),
        ).squeeze(2)

        logits = torch.cat([logits_pos.unsqueeze(-1), logits_neg], dim=-1)  # (B, L, 1+N)

        targets = torch.zeros_like(labels)       # positive class = index 0
        targets[labels == 0] = -100              # padding → ignore
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

    # ── optimizer ──

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self._param_groups)
