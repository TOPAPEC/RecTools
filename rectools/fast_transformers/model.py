"""FlatSASRecModel: standalone flat sequential recommender built on ModelBase."""

import typing as tp

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from scipy import sparse

from rectools import Columns
from rectools.dataset import Dataset
from rectools.dataset.identifiers import IdMap
from rectools.models.base import InternalRecoTriplet, ModelBase, ModelConfig
from rectools.models.nn.transformers.sasrec import SASRecDataPreparator
from rectools.models.nn.transformers.negative_sampler import CatalogUniformSampler
from rectools.types import InternalIdsArray
from rectools.utils.config import BaseConfig

from .lightning_wrap import FlatSASRecLightning
from .net import FlatSASRec
from .ranking import rank_topk


class FlatSASRecConfig(BaseConfig):
    """Configuration for FlatSASRecModel."""

    n_factors: int = 64
    n_blocks: int = 2
    n_heads: int = 2
    session_max_len: int = 32
    dropout: float = 0.1
    loss: str = "softmax"
    n_negatives: int = 1
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    recommend_batch_size: int = 256
    dataloader_num_workers: int = 0
    train_min_user_interactions: int = 2


class FlatSASRecModelConfig(ModelConfig):
    """Full model config including cls."""

    model: FlatSASRecConfig = FlatSASRecConfig()


class FlatSASRecModel(ModelBase[FlatSASRecModelConfig]):
    """
    Flat SASRec model: sequential recommender without the ItemNet hierarchy.

    Uses SASRecDataPreparator for data processing and a standalone FlatSASRec
    network for encoding.
    """

    config_class = FlatSASRecModelConfig
    recommends_for_warm = False
    recommends_for_cold = False

    def __init__(
        self,
        n_factors: int = 64,
        n_blocks: int = 2,
        n_heads: int = 2,
        session_max_len: int = 32,
        dropout: float = 0.1,
        loss: str = "softmax",
        n_negatives: int = 1,
        epochs: int = 5,
        batch_size: int = 128,
        lr: float = 1e-3,
        recommend_batch_size: int = 256,
        dataloader_num_workers: int = 0,
        train_min_user_interactions: int = 2,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)

        if loss not in FlatSASRecLightning.SUPPORTED_LOSSES:
            raise ValueError(f"Unsupported loss '{loss}'. Choose from {FlatSASRecLightning.SUPPORTED_LOSSES}")

        self.n_factors = n_factors
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.session_max_len = session_max_len
        self.dropout = dropout
        self.loss = loss
        self.n_negatives = n_negatives
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.recommend_batch_size = recommend_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.train_min_user_interactions = train_min_user_interactions

        self._net: tp.Optional[FlatSASRec] = None
        self._lightning: tp.Optional[FlatSASRecLightning] = None
        self._data_preparator: tp.Optional[SASRecDataPreparator] = None

    def _get_config(self) -> FlatSASRecModelConfig:
        return FlatSASRecModelConfig(
            cls=self.__class__,
            verbose=self.verbose,
            model=FlatSASRecConfig(
                n_factors=self.n_factors,
                n_blocks=self.n_blocks,
                n_heads=self.n_heads,
                session_max_len=self.session_max_len,
                dropout=self.dropout,
                loss=self.loss,
                n_negatives=self.n_negatives,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                recommend_batch_size=self.recommend_batch_size,
                dataloader_num_workers=self.dataloader_num_workers,
                train_min_user_interactions=self.train_min_user_interactions,
            ),
        )

    @classmethod
    def _from_config(cls, config: FlatSASRecModelConfig) -> "FlatSASRecModel":
        m = config.model
        return cls(
            n_factors=m.n_factors,
            n_blocks=m.n_blocks,
            n_heads=m.n_heads,
            session_max_len=m.session_max_len,
            dropout=m.dropout,
            loss=m.loss,
            n_negatives=m.n_negatives,
            epochs=m.epochs,
            batch_size=m.batch_size,
            lr=m.lr,
            recommend_batch_size=m.recommend_batch_size,
            dataloader_num_workers=m.dataloader_num_workers,
            train_min_user_interactions=m.train_min_user_interactions,
            verbose=config.verbose,
        )

    def _fit(self, dataset: Dataset, *args: tp.Any, **kwargs: tp.Any) -> None:
        negative_sampler = None
        n_negatives_dp: tp.Optional[int] = None
        if self.loss == "BCE":
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

        n_items = dp.item_id_map.size  # includes extra tokens (padding)
        # item ids in the preparator go from 0 (padding) to n_items-1
        # FlatSASRec expects n_items = max real item count (embedding table = n_items+1 with padding at 0)
        # The preparator's item_id_map.size includes the padding token, so real items = size - 1
        n_real_items = dp.item_id_map.size - dp.n_item_extra_tokens

        net = FlatSASRec(
            n_items=n_real_items,
            n_factors=self.n_factors,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            session_max_len=self.session_max_len,
            dropout=self.dropout,
        )

        lightning_model = FlatSASRecLightning(
            net=net,
            lr=self.lr,
            loss=self.loss,
            n_negatives=self.n_negatives,
        )

        train_dl = dp.get_dataloader_train()
        val_dl = dp.get_dataloader_val()

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=self.verbose > 0,
            enable_progress_bar=self.verbose > 0,
        )
        trainer.fit(lightning_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        self._net = net
        self._lightning = lightning_model

    def _custom_transform_dataset_u2i(
        self,
        dataset: Dataset,
        users: tp.Any,
        on_unsupported_targets: tp.Any,
        context: tp.Optional[pd.DataFrame] = None,
    ) -> Dataset:
        assert self._data_preparator is not None
        return self._data_preparator.transform_dataset_u2i(dataset, users)

    def _custom_transform_dataset_i2i(
        self, dataset: Dataset, target_items: tp.Any, on_unsupported_targets: tp.Any
    ) -> Dataset:
        assert self._data_preparator is not None
        return self._data_preparator.transform_dataset_i2i(dataset)

    @torch.no_grad()
    def _get_user_embeddings(self, dataset: Dataset) -> torch.Tensor:
        """Compute user embeddings from their interaction sequences."""
        assert self._data_preparator is not None and self._net is not None
        self._net.eval()

        recommend_dl = self._data_preparator.get_dataloader_recommend(dataset, self.recommend_batch_size)
        device = next(self._net.parameters()).device

        all_embs = []
        for batch in recommend_dl:
            x = batch["x"].to(device)
            embs = self._net.encode_last(x)  # (batch, D)
            all_embs.append(embs)
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def _get_item_embeddings(self) -> torch.Tensor:
        """Get all item embeddings from the network."""
        assert self._net is not None
        self._net.eval()
        return self._net.all_item_embeddings()

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> InternalRecoTriplet:
        assert self._data_preparator is not None
        device = next(self._net.parameters()).device  # type: ignore

        user_embs = self._get_user_embeddings(dataset)  # (n_users, D)
        item_embs = self._get_item_embeddings()  # (n_items, D)

        # Build filter matrix
        filter_csr = None
        if filter_viewed:
            ui_mat = dataset.get_user_item_matrix(include_weights=False)
            n_users_mat = ui_mat.shape[0]
            n_items_emb = item_embs.shape[0]
            n_extra = self._data_preparator.n_item_extra_tokens
            # item_embs[i] corresponds to preparator internal item id (i + n_extra).
            # ui_mat columns are dataset internal item ids which share the preparator's id_map.
            # Slice out the extra-token columns and pad/trim to exactly n_items_emb cols.
            if ui_mat.shape[1] > n_extra:
                sliced = ui_mat[:, n_extra:]
            else:
                sliced = sparse.csr_matrix((n_users_mat, 0))
            n_cols = sliced.shape[1]
            if n_cols < n_items_emb:
                pad = sparse.csr_matrix((n_users_mat, n_items_emb - n_cols))
                filter_csr = sparse.hstack([sliced, pad], format="csr")
            elif n_cols > n_items_emb:
                filter_csr = sliced[:, :n_items_emb]
            else:
                filter_csr = sliced

        # Map whitelist to item_embs indices (0-based, without extra tokens)
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

        # Convert item indices back to preparator's internal ids
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
        device = next(self._net.parameters()).device

        item_embs = self._get_item_embeddings()  # (n_items, D)
        n_extra = self._data_preparator.n_item_extra_tokens

        # Target embeddings: target_ids are preparator internal ids
        target_emb_idx = target_ids - n_extra
        target_embs = item_embs[target_emb_idx]  # (n_targets, D)

        whitelist = None
        if sorted_item_ids_to_recommend is not None:
            wl = sorted_item_ids_to_recommend - n_extra
            whitelist = wl[(wl >= 0) & (wl < item_embs.shape[0])]

        t_ids, i_ids, scores = rank_topk(
            target_embs, item_embs, k,
            whitelist=whitelist,
            batch_size=self.recommend_batch_size,
        )

        # Map back
        result_target_ids = target_ids[t_ids]
        result_item_ids = i_ids + n_extra

        return result_target_ids, result_item_ids, scores
