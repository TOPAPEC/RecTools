"""Tests for FlatSASRecLightning wrapper."""

import torch
import pytest

from rectools.fast_transformers.net import FlatSASRec
from rectools.fast_transformers.lightning_wrap import FlatSASRecLightning


@pytest.fixture()
def net() -> FlatSASRec:
    return FlatSASRec(
        n_items=10,
        n_factors=8,
        n_blocks=1,
        n_heads=1,
        session_max_len=5,
        dropout=0.0,
    )


class TestFlatSASRecLightning:
    # ---- constructor ----

    def test_init_softmax_loss(self, net: FlatSASRec) -> None:
        module = FlatSASRecLightning(net, loss="softmax")
        assert module.loss_name == "softmax"
        assert isinstance(module.loss_fn, torch.nn.CrossEntropyLoss)

    def test_init_bce_loss(self, net: FlatSASRec) -> None:
        module = FlatSASRecLightning(net, loss="BCE")
        assert module.loss_name == "BCE"
        assert isinstance(module.loss_fn, torch.nn.BCEWithLogitsLoss)

    def test_init_invalid_loss_raises(self, net: FlatSASRec) -> None:
        with pytest.raises(ValueError, match="Unsupported loss"):
            FlatSASRecLightning(net, loss="mse")

    def test_init_stores_hyperparams(self, net: FlatSASRec) -> None:
        module = FlatSASRecLightning(net, lr=0.005, n_negatives=4)
        assert module.lr == 0.005
        assert module.n_negatives == 4

    # ---- configure_optimizers ----

    def test_configure_optimizers_type_and_lr(self, net: FlatSASRec) -> None:
        lr = 2e-4
        module = FlatSASRecLightning(net, lr=lr)
        optimizer = module.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == lr

    def test_configure_optimizers_betas(self, net: FlatSASRec) -> None:
        module = FlatSASRecLightning(net)
        optimizer = module.configure_optimizers()
        assert optimizer.defaults["betas"] == (0.9, 0.98)

    # ---- on_train_start ----

    def test_on_train_start_reinitializes_params(self, net: FlatSASRec) -> None:
        module = FlatSASRecLightning(net)

        # Snapshot parameters with dim > 1 before reinit
        snapshots_before = {
            name: p.clone() for name, p in module.net.named_parameters() if p.dim() > 1
        }
        assert len(snapshots_before) > 0, "Expected at least one param with dim > 1"

        # Force parameters to a constant value so reinit is detectable
        with torch.no_grad():
            for p in module.net.parameters():
                if p.dim() > 1:
                    p.fill_(42.0)

        module.on_train_start()

        changed = False
        for name, p in module.net.named_parameters():
            if p.dim() > 1 and not torch.all(p == 42.0):
                changed = True
                break
        assert changed, "on_train_start should reinitialize parameters via xavier_uniform_"

    # ---- training_step with softmax ----

    def test_training_step_softmax_returns_scalar(self, net: FlatSASRec) -> None:
        module = FlatSASRecLightning(net, loss="softmax")
        batch = {
            "x": torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
            "y": torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]]),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

    def test_training_step_softmax_positive_loss(self, net: FlatSASRec) -> None:
        module = FlatSASRecLightning(net, loss="softmax")
        batch = {
            "x": torch.tensor([[1, 2, 3, 4, 5]]),
            "y": torch.tensor([[2, 3, 4, 5, 6]]),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.item() > 0, "Cross-entropy loss should be positive"

    def test_training_step_softmax_all_padding_returns_nan(self, net: FlatSASRec) -> None:
        """When all targets are padding (y=0), cross_entropy with ignore_index=-100 returns NaN."""
        module = FlatSASRecLightning(net, loss="softmax")
        batch = {
            "x": torch.tensor([[0, 0, 0, 0, 0]]),
            "y": torch.tensor([[0, 0, 0, 0, 0]]),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0
        # PyTorch cross_entropy returns NaN when all targets are ignored
        assert torch.isnan(loss)

    # ---- training_step with BCE ----

    def test_training_step_bce_returns_scalar(self, net: FlatSASRec) -> None:
        n_negatives = 3
        module = FlatSASRecLightning(net, loss="BCE", n_negatives=n_negatives)
        batch = {
            "x": torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
            "y": torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]]),
            "negatives": torch.randint(1, 10, (2, 5, n_negatives)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

    def test_training_step_bce_positive_loss(self, net: FlatSASRec) -> None:
        n_negatives = 2
        module = FlatSASRecLightning(net, loss="BCE", n_negatives=n_negatives)
        batch = {
            "x": torch.tensor([[1, 2, 3, 4, 5]]),
            "y": torch.tensor([[2, 3, 4, 5, 6]]),
            "negatives": torch.randint(1, 10, (1, 5, n_negatives)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.item() > 0, "BCE loss should be positive"

    def test_training_step_bce_mask_reduces_loss(self, net: FlatSASRec) -> None:
        """Padding positions should not contribute to BCE loss."""
        n_negatives = 2
        module = FlatSASRecLightning(net, loss="BCE", n_negatives=n_negatives)
        module.eval()

        torch.manual_seed(0)
        negs = torch.randint(1, 10, (1, 5, n_negatives))

        # Batch with no padding
        batch_full = {
            "x": torch.tensor([[1, 2, 3, 4, 5]]),
            "y": torch.tensor([[2, 3, 4, 5, 6]]),
            "negatives": negs.clone(),
        }
        # Batch with partial padding
        batch_padded = {
            "x": torch.tensor([[0, 0, 3, 4, 5]]),
            "y": torch.tensor([[0, 0, 4, 5, 6]]),
            "negatives": negs.clone(),
        }

        with torch.no_grad():
            loss_full = module.training_step(batch_full, batch_idx=0)
            loss_padded = module.training_step(batch_padded, batch_idx=0)

        # Losses should differ because the padded batch masks out some positions
        assert loss_full.item() != pytest.approx(loss_padded.item(), abs=1e-6)

    # ---- supported losses constant ----

    def test_supported_losses_tuple(self) -> None:
        assert FlatSASRecLightning.SUPPORTED_LOSSES == ("softmax", "BCE")
