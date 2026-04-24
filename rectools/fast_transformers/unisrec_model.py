"""UniSRecModel: standalone model with configurable three-phase training."""

import typing as tp
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

from .gpu_data import align_embeddings, build_sequences, hash_item_ids, make_dataloader
from .unisrec_lightning import SUPPORTED_LOSSES, SUPPORTED_OPTIMIZERS, SUPPORTED_SCHEDULERS, UniSRecLightning
from .unisrec_net import UniSRec


class _ProjectAllWrapper(torch.nn.Module):
    def __init__(self, net: UniSRec) -> None:
        super().__init__()
        self.net = net

    def forward(self) -> torch.Tensor:
        return self.net.project_all()


class UniSRecModel:
    """
    UniSRec sequential recommender with pretrained text embeddings.

    Three training phases
    ---------------------
    1. **Phase 1** - SASRec on ID embeddings (``item_emb`` + transformer).
    2. **Phase 2** - Adaptor only (transformer frozen, pretrained embeddings).
    3. **Phase 3** - Full fine-tune (adaptor + transformer, pretrained embeddings).

    Parameters
    ----------
    pretrained_item_embeddings : Tensor
        Shape ``(max_external_item_id + 1, D_text)`` or
        ``(max_external_item_id + 1, n_variants, D_text)``.
        Index *i* holds the text embedding for the item whose **external** ID
        equals *i*.  Index 0 is padding (zeros).
    """

    def __init__(
        self,
        pretrained_item_embeddings: torch.Tensor,
        # architecture
        n_factors: int = 256,
        projection_hidden: int = 512,
        n_blocks: int = 2,
        n_heads: int = 1,
        session_max_len: int = 200,
        dropout: float = 0.1,
        adaptor_dropout: float = 0.2,
        adaptor_type: str = "pca",
        use_adaptor_ffn: bool = True,
        ffn_type: str = "conv1d",
        ffn_expansion: int = 1,
        # training phases
        phase1_epochs: int = 10,
        phase2_epochs: int = 10,
        phase3_epochs: int = 10,
        phase1_lr: float = 1e-3,
        phase2_lr: float = 3e-4,
        phase3_lr: float = 1e-4,
        lr_head: float = 0.3,
        lr_wp: float = 0.1,
        lr_transformer: float = 3.0,
        # optimizer / scheduler
        optimizer: str = "adamw",
        scheduler: tp.Optional[str] = None,
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.1,
        grad_clip: float = 1.0,
        weight_decay: float = 0.01,
        # loss
        loss: str = "softmax",
        gbce_t: float = 0.2,
        n_negatives: tp.Optional[int] = None,
        # early stopping
        patience: tp.Optional[int] = None,
        # data
        batch_size: int = 128,
        dataloader_num_workers: int = 0,
        train_min_user_interactions: int = 2,
        id_mapping: str = "dense",
        verbose: int = 0,
    ) -> None:
        if loss not in SUPPORTED_LOSSES:
            raise ValueError(f"Unsupported loss '{loss}'. Choose from {SUPPORTED_LOSSES}")
        if loss in ("BCE", "gBCE", "sampled_softmax") and n_negatives is None:
            raise ValueError(f"Loss '{loss}' requires n_negatives to be set")
        if optimizer not in SUPPORTED_OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer '{optimizer}'. Choose from {SUPPORTED_OPTIMIZERS}")
        if scheduler not in SUPPORTED_SCHEDULERS:
            raise ValueError(f"Unsupported scheduler '{scheduler}'. Choose from {SUPPORTED_SCHEDULERS}")

        self.pretrained_item_embeddings = pretrained_item_embeddings
        self.n_factors = n_factors
        self.projection_hidden = projection_hidden
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.session_max_len = session_max_len
        self.dropout = dropout
        self.adaptor_dropout = adaptor_dropout
        self.adaptor_type = adaptor_type
        self.use_adaptor_ffn = use_adaptor_ffn
        self.ffn_type = ffn_type
        self.ffn_expansion = ffn_expansion
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.phase3_epochs = phase3_epochs
        self.phase1_lr = phase1_lr
        self.phase2_lr = phase2_lr
        self.phase3_lr = phase3_lr
        self.lr_head = lr_head
        self.lr_wp = lr_wp
        self.lr_transformer = lr_transformer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.loss = loss
        self.gbce_t = gbce_t
        self.n_negatives = n_negatives
        self.patience = patience
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.train_min_user_interactions = train_min_user_interactions
        self.id_mapping = id_mapping
        self.verbose = verbose

        self._net: tp.Optional[UniSRec] = None
        self._unique_items: tp.Optional[torch.Tensor] = None
        self._unique_users: tp.Optional[torch.Tensor] = None
        self.is_fitted: bool = False

    # ── helpers ──

    def _make_trainer(self, max_epochs: int, val_dl: tp.Any = None) -> pl.Trainer:
        callbacks = []
        if self.patience is not None and val_dl is not None:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=self.patience, mode="min"))

        return pl.Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=self.grad_clip,
            callbacks=callbacks or None,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=self.verbose > 0,
            enable_progress_bar=self.verbose > 0,
        )

    def _make_lightning(
        self,
        net: UniSRec,
        param_groups: tp.List[tp.Dict],
        use_id: bool,
        max_epochs: int,
        train_dl: tp.Any,
    ) -> UniSRecLightning:
        total_steps = len(train_dl) * max_epochs if self.scheduler else None
        return UniSRecLightning(
            net=net,
            param_groups=param_groups,
            use_id=use_id,
            loss=self.loss,
            n_negatives=self.n_negatives,
            gbce_t=self.gbce_t,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            warmup_ratio=self.warmup_ratio,
            min_lr_ratio=self.min_lr_ratio,
            total_steps=total_steps,
        )

    # ── Phase param-groups ──

    def _phase1_params(self, net: UniSRec) -> tp.List[tp.Dict[str, tp.Any]]:
        return [{"params": list(net.item_emb.parameters()) + net.transformer_params, "lr": self.phase1_lr}]

    def _phase2_params(self, net: UniSRec) -> tp.List[tp.Dict[str, tp.Any]]:
        if self.adaptor_type == "pca":
            groups: tp.List[tp.Dict[str, tp.Any]] = [
                {"params": [net.whitening_proj], "lr": self.phase2_lr * self.lr_wp, "weight_decay": 0.0},
                {"params": [net.whitening_bias], "lr": self.phase2_lr * 10.0, "weight_decay": 0.0},
            ]
            if net.head is not None:
                groups.append(
                    {
                        "params": list(net.head.parameters()),
                        "lr": self.phase2_lr * self.lr_head,
                        "weight_decay": self.weight_decay,
                    }
                )
        else:
            groups = [
                {"params": list(net.bn_input.parameters()), "lr": self.phase2_lr, "weight_decay": 0.0},
                {"params": list(net.bn_score.parameters()), "lr": self.phase2_lr, "weight_decay": 0.0},
                {
                    "params": list(net.head.parameters()),
                    "lr": self.phase2_lr * self.lr_head,
                    "weight_decay": self.weight_decay,
                },
            ]
        return groups

    def _phase3_params(self, net: UniSRec) -> tp.List[tp.Dict[str, tp.Any]]:
        if self.adaptor_type == "pca":
            adaptor: tp.List[tp.Dict[str, tp.Any]] = [
                {"params": [net.whitening_proj], "lr": self.phase3_lr * self.lr_wp, "weight_decay": 0.0},
                {"params": [net.whitening_bias], "lr": self.phase3_lr * 10.0, "weight_decay": 0.0},
            ]
        else:
            adaptor = [
                {"params": list(net.bn_input.parameters()), "lr": self.phase3_lr, "weight_decay": 0.0},
                {"params": list(net.bn_score.parameters()), "lr": self.phase3_lr, "weight_decay": 0.0},
            ]
        head: tp.List[tp.Dict[str, tp.Any]] = []
        if net.head is not None:
            head = [
                {
                    "params": list(net.head.parameters()),
                    "lr": self.phase3_lr * self.lr_head,
                    "weight_decay": self.weight_decay,
                }
            ]
        transformer = [
            {"params": list(net.pos_emb.parameters()), "lr": self.phase3_lr * self.lr_transformer, "weight_decay": 0.0},
            {
                "params": (
                    [p for layer in net.attention_layers for p in layer.parameters()]
                    + [p for layer in net.forward_layers for p in layer.parameters()]
                ),
                "lr": self.phase3_lr * self.lr_transformer,
                "weight_decay": self.weight_decay,
            },
            {
                "params": (
                    [p for layer in net.attention_layernorms for p in layer.parameters()]
                    + [p for layer in net.forward_layernorms for p in layer.parameters()]
                    + list(net.last_layernorm.parameters())
                ),
                "lr": self.phase3_lr,
                "weight_decay": 0.0,
            },
        ]
        return adaptor + head + transformer

    # ── fit ──

    def fit(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> "UniSRecModel":
        """
        Train the model on interaction data.

        Parameters
        ----------
        user_ids : LongTensor (N,)
            External user IDs for each interaction.
        item_ids : LongTensor (N,)
            External item IDs for each interaction.
        timestamps : LongTensor (N,)
            Timestamps (any monotonic int64 values).

        Returns
        -------
        self
        """
        x, y, unique_items, unique_users = build_sequences(
            user_ids,
            item_ids,
            timestamps,
            max_len=self.session_max_len,
            min_interactions=self.train_min_user_interactions,
            id_mapping=self.id_mapping,
        )
        self._unique_items = unique_items.cpu()
        self._unique_users = unique_users.cpu()
        n_items = len(unique_items)

        aligned_emb = align_embeddings(self.pretrained_item_embeddings, unique_items, n_items, self.id_mapping)

        net = UniSRec(
            n_items=n_items,
            pretrained_embeddings=aligned_emb,
            n_factors=self.n_factors,
            projection_hidden=self.projection_hidden,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            session_max_len=self.session_max_len,
            dropout=self.dropout,
            adaptor_dropout=self.adaptor_dropout,
            adaptor_type=self.adaptor_type,
            use_adaptor_ffn=self.use_adaptor_ffn,
            ffn_type=self.ffn_type,
            ffn_expansion=self.ffn_expansion,
        )

        train_dl = make_dataloader(x, y, batch_size=self.batch_size, shuffle=True)

        val_dl = None
        if self.patience is not None:
            val_y_last = y[:, -1:]
            val_dl = make_dataloader(x, val_y_last, batch_size=self.batch_size, shuffle=False)

        def _run_phase(param_groups: tp.List[tp.Dict], use_id: bool, max_epochs: int) -> None:
            lm = self._make_lightning(net, param_groups, use_id, max_epochs, train_dl)
            trainer = self._make_trainer(max_epochs, val_dl)
            trainer.fit(lm, train_dl, val_dl)

        if self.phase1_epochs > 0:
            _run_phase(self._phase1_params(net), use_id=True, max_epochs=self.phase1_epochs)

        if self.phase2_epochs > 0 and self.use_adaptor_ffn:
            net.freeze_transformer()
            _run_phase(self._phase2_params(net), use_id=False, max_epochs=self.phase2_epochs)

        if self.phase3_epochs > 0:
            net.unfreeze_transformer()
            _run_phase(self._phase3_params(net), use_id=False, max_epochs=self.phase3_epochs)

        self._net = net
        self.is_fitted = True
        return self

    # ── save / load ──

    def save_checkpoint(self, path: tp.Union[str, Path]) -> None:
        assert self._net is not None
        torch.save(
            {
                "net": self._net.state_dict(),
                "unique_items": self._unique_items,
                "unique_users": self._unique_users,
                "n_items": len(self._unique_items),
                "id_mapping": self.id_mapping,
            },
            path,
        )

    def load_checkpoint(self, path: tp.Union[str, Path], device: str = "cuda") -> None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self._unique_items = ckpt["unique_items"].cpu()
        self._unique_users = ckpt["unique_users"].cpu()
        n_items = ckpt["n_items"]
        self.id_mapping = ckpt.get("id_mapping", "dense")

        aligned_emb = align_embeddings(self.pretrained_item_embeddings, self._unique_items, n_items, self.id_mapping)

        self._net = UniSRec(
            n_items=n_items,
            pretrained_embeddings=aligned_emb,
            n_factors=self.n_factors,
            projection_hidden=self.projection_hidden,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            session_max_len=self.session_max_len,
            dropout=self.dropout,
            adaptor_dropout=self.adaptor_dropout,
            adaptor_type=self.adaptor_type,
            use_adaptor_ffn=self.use_adaptor_ffn,
            ffn_type=self.ffn_type,
            ffn_expansion=self.ffn_expansion,
        )
        self._net.load_state_dict(ckpt["net"])
        self._net.to(device).eval()
        self.is_fitted = True

    # ── ONNX export ──

    def export_to_onnx(
        self,
        encoder_path: tp.Union[str, Path],
        items_path: tp.Optional[tp.Union[str, Path]] = None,
        opset_version: int = 18,
    ) -> None:
        """Export the model to ONNX.

        Parameters
        ----------
        encoder_path
            Path for the encoder graph (input_ids -> hidden states).
        items_path
            If given, also exports project_all (-> item embeddings).
        opset_version
            ONNX opset version (default 18).
        """
        assert self._net is not None, "Model not fitted or loaded"
        net = self._net
        was_training = net.training
        net.eval()

        device = next(net.parameters()).device
        dummy = torch.zeros(1, 5, dtype=torch.long, device=device)

        torch.onnx.export(
            net,
            (dummy, False),
            str(encoder_path),
            input_names=["input_ids"],
            output_names=["hidden"],
            opset_version=opset_version,
        )

        if items_path is not None:
            wrapper = _ProjectAllWrapper(net)
            wrapper.eval()
            torch.onnx.export(
                wrapper,
                (),
                str(items_path),
                input_names=[],
                output_names=["item_embs"],
                opset_version=opset_version,
            )

        if was_training:
            net.train()

    def map_item_ids(self, external_ids: torch.Tensor) -> torch.Tensor:
        """Map external item IDs to internal IDs used by the model.

        Parameters
        ----------
        external_ids : LongTensor
            External item IDs.

        Returns
        -------
        LongTensor
            Internal IDs in ``[0, n_items]``.  0 means unknown item.
        """
        assert self._unique_items is not None, "Model not fitted or loaded"
        if self.id_mapping == "hash":
            n_items = len(self._unique_items)
            known = torch.isin(external_ids, self._unique_items)
            result = torch.zeros_like(external_ids)
            result[known] = hash_item_ids(external_ids[known], n_items)
            return result

        lookup = {int(v): i + 1 for i, v in enumerate(self._unique_items.tolist())}
        return torch.tensor([lookup.get(int(x), 0) for x in external_ids.tolist()], dtype=torch.long)

    @property
    def net(self) -> UniSRec:
        assert self._net is not None, "Model not fitted or loaded"
        return self._net

    @property
    def item_id_mapping(self) -> torch.Tensor:
        return self._unique_items
