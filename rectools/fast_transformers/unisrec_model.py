"""UniSRecModel: ModelBase wrapper with three-phase training."""

import typing as tp

import numpy as np
import torch
import pytorch_lightning as pl
from scipy import sparse

from rectools.dataset import Dataset
from rectools.models.base import InternalRecoTriplet, ModelBase, ModelConfig
from rectools.models.nn.transformers.sasrec import SASRecDataPreparator
from rectools.models.nn.transformers.negative_sampler import CatalogUniformSampler
from rectools.types import InternalIdsArray
from rectools.utils.config import BaseConfig

from .unisrec_net import UniSRec
from .unisrec_lightning import UniSRecLightning
from .ranking import rank_topk


class UniSRecConfig(BaseConfig):
    """Hyperparameters for UniSRecModel (without pretrained embeddings)."""

    n_factors: int = 256
    projection_hidden: int = 512
    n_blocks: int = 2
    n_heads: int = 1
    session_max_len: int = 200
    dropout: float = 0.1
    adaptor_dropout: float = 0.2
    adaptor_type: str = "pca"
    use_adaptor_ffn: bool = True

    phase1_epochs: int = 10
    phase2_epochs: int = 10
    phase3_epochs: int = 10
    phase1_lr: float = 1e-3
    phase2_lr: float = 3e-4
    phase3_lr: float = 1e-4
    lr_head: float = 0.3
    lr_wp: float = 0.1
    lr_transformer: float = 3.0

    grad_clip: float = 1.0
    weight_decay: float = 0.01
    batch_size: int = 128
    recommend_batch_size: int = 256
    dataloader_num_workers: int = 0
    train_min_user_interactions: int = 2
    n_negatives: tp.Optional[int] = None


class UniSRecModelConfig(ModelConfig):
    """Full model config (cls + verbose + hyper-params)."""

    model: UniSRecConfig = UniSRecConfig()


class UniSRecModel(ModelBase[UniSRecModelConfig]):
    """
    UniSRec integrated into RecTools via ``ModelBase``.

    Three training phases
    ---------------------
    1. **Phase 1** — SASRec on ID embeddings (``item_emb`` + transformer).
    2. **Phase 2** — Adaptor only (transformer frozen, pretrained embeddings).
    3. **Phase 3** — Full fine-tune (adaptor + transformer, pretrained embeddings).

    Parameters
    ----------
    pretrained_item_embeddings : Tensor
        Shape ``(max_external_item_id + 1, D_text)`` or
        ``(max_external_item_id + 1, n_variants, D_text)``.
        Index *i* holds the text embedding for the item whose **external** ID
        equals *i*.  Index 0 is padding (zeros).
        During ``fit`` the tensor is reindexed to match the internal ID map
        produced by ``SASRecDataPreparator``.
    """

    config_class = UniSRecModelConfig
    recommends_for_warm = False
    recommends_for_cold = False

    def __init__(
        self,
        pretrained_item_embeddings: torch.Tensor,
        n_factors: int = 256,
        projection_hidden: int = 512,
        n_blocks: int = 2,
        n_heads: int = 1,
        session_max_len: int = 200,
        dropout: float = 0.1,
        adaptor_dropout: float = 0.2,
        adaptor_type: str = "pca",
        use_adaptor_ffn: bool = True,
        phase1_epochs: int = 10,
        phase2_epochs: int = 10,
        phase3_epochs: int = 10,
        phase1_lr: float = 1e-3,
        phase2_lr: float = 3e-4,
        phase3_lr: float = 1e-4,
        lr_head: float = 0.3,
        lr_wp: float = 0.1,
        lr_transformer: float = 3.0,
        grad_clip: float = 1.0,
        weight_decay: float = 0.01,
        batch_size: int = 128,
        recommend_batch_size: int = 256,
        dataloader_num_workers: int = 0,
        train_min_user_interactions: int = 2,
        n_negatives: tp.Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
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
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.phase3_epochs = phase3_epochs
        self.phase1_lr = phase1_lr
        self.phase2_lr = phase2_lr
        self.phase3_lr = phase3_lr
        self.lr_head = lr_head
        self.lr_wp = lr_wp
        self.lr_transformer = lr_transformer
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.recommend_batch_size = recommend_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.train_min_user_interactions = train_min_user_interactions
        self.n_negatives = n_negatives

        self._net: tp.Optional[UniSRec] = None
        self._data_preparator: tp.Optional[SASRecDataPreparator] = None

    # ── config boilerplate (embeddings are not serialised) ──

    def _get_config(self) -> UniSRecModelConfig:
        return UniSRecModelConfig(
            cls=self.__class__,
            verbose=self.verbose,
            model=UniSRecConfig(
                n_factors=self.n_factors,
                projection_hidden=self.projection_hidden,
                n_blocks=self.n_blocks,
                n_heads=self.n_heads,
                session_max_len=self.session_max_len,
                dropout=self.dropout,
                adaptor_dropout=self.adaptor_dropout,
                adaptor_type=self.adaptor_type,
                use_adaptor_ffn=self.use_adaptor_ffn,
                phase1_epochs=self.phase1_epochs,
                phase2_epochs=self.phase2_epochs,
                phase3_epochs=self.phase3_epochs,
                phase1_lr=self.phase1_lr,
                phase2_lr=self.phase2_lr,
                phase3_lr=self.phase3_lr,
                lr_head=self.lr_head,
                lr_wp=self.lr_wp,
                lr_transformer=self.lr_transformer,
                grad_clip=self.grad_clip,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                recommend_batch_size=self.recommend_batch_size,
                dataloader_num_workers=self.dataloader_num_workers,
                train_min_user_interactions=self.train_min_user_interactions,
                n_negatives=self.n_negatives,
            ),
        )

    @classmethod
    def _from_config(cls, config: UniSRecModelConfig) -> "UniSRecModel":
        raise NotImplementedError(
            "UniSRecModel cannot be restored from config alone — "
            "pretrained_item_embeddings must be supplied at construction time."
        )

    # ── helpers ──

    def _align_embeddings(self, dp: SASRecDataPreparator) -> torch.Tensor:
        """Reindex ``pretrained_item_embeddings`` to the preparator's internal IDs."""
        ext_ids = dp.item_id_map.to_external.values  # array[internal_id] → external_id
        n_internal = dp.item_id_map.size
        n_extra = dp.n_item_extra_tokens

        emb = self.pretrained_item_embeddings
        if emb.ndim == 2:
            aligned = torch.zeros(n_internal, emb.shape[1])
        else:
            aligned = torch.zeros(n_internal, emb.shape[1], emb.shape[2])

        for int_id in range(n_extra, n_internal):
            ext_id = int(ext_ids[int_id])
            if 0 <= ext_id < emb.shape[0]:
                aligned[int_id] = emb[ext_id]

        return aligned

    def _make_trainer(self, max_epochs: int) -> pl.Trainer:
        return pl.Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=self.grad_clip,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=self.verbose > 0,
            enable_progress_bar=self.verbose > 0,
        )

    # ── Phase param-groups ──

    def _phase2_params(self, net: UniSRec) -> tp.List[tp.Dict[str, tp.Any]]:
        if self.adaptor_type == "pca":
            groups: tp.List[tp.Dict[str, tp.Any]] = [
                {"params": [net.whitening_proj], "lr": self.phase2_lr * self.lr_wp, "weight_decay": 0.0},
                {"params": [net.whitening_bias], "lr": self.phase2_lr * 10.0, "weight_decay": 0.0},
            ]
            if net.head is not None:
                groups.append({
                    "params": list(net.head.parameters()),
                    "lr": self.phase2_lr * self.lr_head,
                    "weight_decay": self.weight_decay,
                })
        else:
            groups = [
                {"params": list(net.bn_input.parameters()), "lr": self.phase2_lr, "weight_decay": 0.0},
                {"params": list(net.bn_score.parameters()), "lr": self.phase2_lr, "weight_decay": 0.0},
                {"params": list(net.head.parameters()), "lr": self.phase2_lr * self.lr_head, "weight_decay": self.weight_decay},
            ]
        return groups

    def _phase3_params(self, net: UniSRec) -> tp.List[tp.Dict[str, tp.Any]]:
        # adaptor
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
        # head
        head: tp.List[tp.Dict[str, tp.Any]] = []
        if net.head is not None:
            head = [{"params": list(net.head.parameters()), "lr": self.phase3_lr * self.lr_head, "weight_decay": self.weight_decay}]
        # transformer
        transformer = [
            {"params": list(net.pos_emb.parameters()), "lr": self.phase3_lr * self.lr_transformer, "weight_decay": 0.0},
            {
                "params": (
                    [p for l in net.attention_layers for p in l.parameters()]
                    + [p for l in net.forward_layers for p in l.parameters()]
                ),
                "lr": self.phase3_lr * self.lr_transformer,
                "weight_decay": self.weight_decay,
            },
            {
                "params": (
                    [p for l in net.attention_layernorms for p in l.parameters()]
                    + [p for l in net.forward_layernorms for p in l.parameters()]
                    + list(net.last_layernorm.parameters())
                ),
                "lr": self.phase3_lr,
                "weight_decay": 0.0,
            },
        ]
        return adaptor + head + transformer

    # ── fit ──

    def _fit(self, dataset: Dataset, *args: tp.Any, **kwargs: tp.Any) -> None:
        # Data preparation
        negative_sampler = None
        n_negatives_dp: tp.Optional[int] = None
        if self.n_negatives is not None:
            negative_sampler = CatalogUniformSampler(n_negatives=self.n_negatives)
            n_negatives_dp = self.n_negatives

        dp = SASRecDataPreparator(
            session_max_len=self.session_max_len,
            batch_size=self.batch_size,
            dataloader_num_workers=self.dataloader_num_workers,
            train_min_user_interactions=self.train_min_user_interactions,
            n_negatives=n_negatives_dp,
            negative_sampler=negative_sampler,
        )
        dp.process_dataset_train(dataset)
        self._data_preparator = dp

        n_real_items = dp.item_id_map.size - dp.n_item_extra_tokens
        aligned_emb = self._align_embeddings(dp)

        net = UniSRec(
            n_items=n_real_items,
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
        )

        train_dl = dp.get_dataloader_train()

        # ── Phase 1: ID embeddings ──
        if self.phase1_epochs > 0:
            p1_params = [{"params": list(net.item_emb.parameters()) + net.transformer_params, "lr": self.phase1_lr}]
            lm = UniSRecLightning(net, p1_params, use_id=True)
            self._make_trainer(self.phase1_epochs).fit(lm, train_dl)

        # ── Phase 2: adaptor only (transformer frozen) ──
        if self.phase2_epochs > 0 and self.use_adaptor_ffn:
            net.freeze_transformer()
            lm = UniSRecLightning(net, self._phase2_params(net), use_id=False)
            self._make_trainer(self.phase2_epochs).fit(lm, train_dl)

        # ── Phase 3: full fine-tune ──
        if self.phase3_epochs > 0:
            net.unfreeze_transformer()
            lm = UniSRecLightning(net, self._phase3_params(net), use_id=False)
            self._make_trainer(self.phase3_epochs).fit(lm, train_dl)

        self._net = net

    # ── dataset transforms ──

    def _custom_transform_dataset_u2i(
        self,
        dataset: Dataset,
        users: tp.Any,
        on_unsupported_targets: tp.Any,
        context: tp.Optional["pd.DataFrame"] = None,
    ) -> Dataset:
        assert self._data_preparator is not None
        return self._data_preparator.transform_dataset_u2i(dataset, users)

    def _custom_transform_dataset_i2i(
        self, dataset: Dataset, target_items: tp.Any, on_unsupported_targets: tp.Any
    ) -> Dataset:
        assert self._data_preparator is not None
        return self._data_preparator.transform_dataset_i2i(dataset)

    # ── embeddings for ranking ──

    @torch.no_grad()
    def _get_user_embeddings(self, dataset: Dataset) -> torch.Tensor:
        assert self._data_preparator is not None and self._net is not None
        self._net.eval()
        device = next(self._net.parameters()).device
        recommend_dl = self._data_preparator.get_dataloader_recommend(dataset, self.recommend_batch_size)
        all_embs = []
        for batch in recommend_dl:
            x = batch["x"].to(device)
            all_embs.append(self._net.encode_last(x, use_id=False))
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def _get_item_embeddings(self) -> torch.Tensor:
        assert self._net is not None
        self._net.eval()
        all_emb = self._net.project_all()  # (n_items+1, D)
        return all_emb[1:]                  # skip padding → (n_items, D)

    # ── recommend ──

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> InternalRecoTriplet:
        assert self._data_preparator is not None
        device = next(self._net.parameters()).device  # type: ignore[union-attr]

        user_embs = self._get_user_embeddings(dataset)
        item_embs = self._get_item_embeddings()

        # viewed-item filter
        filter_csr = None
        if filter_viewed:
            ui_mat = dataset.get_user_item_matrix(include_weights=False)
            n_users_mat = ui_mat.shape[0]
            n_items_emb = item_embs.shape[0]
            n_extra = self._data_preparator.n_item_extra_tokens

            sliced = ui_mat[:, n_extra:] if ui_mat.shape[1] > n_extra else sparse.csr_matrix((n_users_mat, 0))
            n_cols = sliced.shape[1]
            if n_cols < n_items_emb:
                filter_csr = sparse.hstack([sliced, sparse.csr_matrix((n_users_mat, n_items_emb - n_cols))], format="csr")
            elif n_cols > n_items_emb:
                filter_csr = sliced[:, :n_items_emb]
            else:
                filter_csr = sliced

        # whitelist
        whitelist = None
        if sorted_item_ids_to_recommend is not None:
            n_extra = self._data_preparator.n_item_extra_tokens
            wl = sorted_item_ids_to_recommend - n_extra
            whitelist = wl[(wl >= 0) & (wl < item_embs.shape[0])]

        u_ids, i_ids, scores = rank_topk(
            user_embs, item_embs, k,
            filter_csr=filter_csr,
            whitelist=whitelist,
            batch_size=self.recommend_batch_size,
        )

        n_extra = self._data_preparator.n_item_extra_tokens
        i_ids = i_ids + n_extra
        return u_ids, i_ids, scores

    def _recommend_i2i(
        self,
        target_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> InternalRecoTriplet:
        assert self._data_preparator is not None and self._net is not None

        item_embs = self._get_item_embeddings()
        n_extra = self._data_preparator.n_item_extra_tokens

        target_emb_idx = target_ids - n_extra
        target_embs = item_embs[target_emb_idx]

        whitelist = None
        if sorted_item_ids_to_recommend is not None:
            wl = sorted_item_ids_to_recommend - n_extra
            whitelist = wl[(wl >= 0) & (wl < item_embs.shape[0])]

        t_ids, i_ids, scores = rank_topk(
            target_embs, item_embs, k,
            whitelist=whitelist,
            batch_size=self.recommend_batch_size,
        )

        result_target_ids = target_ids[t_ids]
        result_item_ids = i_ids + n_extra
        return result_target_ids, result_item_ids, scores
