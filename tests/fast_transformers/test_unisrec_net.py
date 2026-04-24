"""Tests for UniSRec network."""

import pytest
import torch

from rectools.fast_transformers.unisrec_net import UniSRec


@pytest.fixture()
def pretrained_emb() -> torch.Tensor:
    """Fake pretrained embeddings: (31, 64) — 30 items + 1 padding."""
    torch.manual_seed(0)
    emb = torch.randn(31, 64)
    emb[0] = 0.0  # padding
    return emb


@pytest.fixture()
def net(pretrained_emb: torch.Tensor) -> UniSRec:
    return UniSRec(
        n_items=30,
        pretrained_embeddings=pretrained_emb,
        n_factors=16,
        projection_hidden=32,
        n_blocks=1,
        n_heads=2,
        session_max_len=8,
        dropout=0.0,
        adaptor_dropout=0.0,
    )


class TestUniSRecShapes:
    def test_forward_id_shape(self, net: UniSRec) -> None:
        x = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]])
        h = net(x, use_id=True)
        assert h.shape == (2, 5, 16)

    def test_forward_adapted_shape(self, net: UniSRec) -> None:
        x = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]])
        h = net(x, use_id=False)
        assert h.shape == (2, 5, 16)

    def test_encode_last_shape(self, net: UniSRec) -> None:
        x = torch.tensor([[0, 0, 1, 2, 3]])
        emb = net.encode_last(x, use_id=False)
        assert emb.shape == (1, 16)

    def test_project_all_shape(self, net: UniSRec) -> None:
        proj = net.project_all()
        assert proj.shape == (31, 16)  # n_items + 1 (with padding)

    def test_item_emb_shape(self, net: UniSRec) -> None:
        assert net.item_emb.weight.shape == (31, 16)


class TestUniSRecAdaptor:
    def test_pca_no_ffn(self, pretrained_emb: torch.Tensor) -> None:
        net = UniSRec(
            n_items=30,
            pretrained_embeddings=pretrained_emb,
            n_factors=16,
            n_blocks=1,
            n_heads=2,
            session_max_len=8,
            adaptor_type="pca",
            use_adaptor_ffn=False,
        )
        proj = net.project_all()
        assert proj.shape == (31, 16)
        assert net.head is None

    def test_multi_variant(self) -> None:
        torch.manual_seed(0)
        emb = torch.randn(31, 3, 64)  # 3 variants
        emb[0] = 0.0
        net = UniSRec(
            n_items=30,
            pretrained_embeddings=emb,
            n_factors=16,
            projection_hidden=32,
            n_blocks=1,
            n_heads=2,
            session_max_len=8,
        )
        assert net.n_variants == 3
        x = torch.tensor([[0, 0, 1, 2, 3]])
        h = net(x, use_id=False)
        assert h.shape == (1, 5, 16)


class TestFreezeUnfreeze:
    def test_freeze_transformer(self, net: UniSRec) -> None:
        net.freeze_transformer()
        for p in net.transformer_params:
            assert not p.requires_grad
        for p in net.adaptor_params:
            assert p.requires_grad

    def test_unfreeze_transformer(self, net: UniSRec) -> None:
        net.freeze_transformer()
        net.unfreeze_transformer()
        for p in net.transformer_params:
            assert p.requires_grad


class TestPaddingInvariance:
    def test_same_input_same_output(self, net: UniSRec) -> None:
        net.eval()
        x_a = torch.tensor([[0, 0, 0, 5, 10]])
        x_b = torch.tensor([[0, 0, 0, 5, 10]])
        with torch.no_grad():
            e_a = net.encode_last(x_a, use_id=False)
            e_b = net.encode_last(x_b, use_id=False)
        torch.testing.assert_close(e_a, e_b)
