"""Tests for training pipeline components.

Verifies:
1. EMA centering tracks and updates correctly
2. NCE loss computes valid gradients
3. Task loss combines energy and BCE
4. Sampling strategies produce correct shapes
5. Full trainer can run a few steps without errors
"""

import pytest
import torch
import numpy as np

from crel.training.ema import EMACenter
from crel.training.sampling import sample_bernoulli, sample_gaussian_noise
from crel.losses.nce import NCELoss, compute_log_prob
from crel.losses.task_loss import TaskLoss
from crel.models.task_net import TaskNet
from crel.models.loss_net import LossNet
from crel.diagnostics.metrics import compute_f1_metrics


class TestEMACenter:
    """Test EMA centering module."""

    def test_per_sample_init(self):
        ema = EMACenter(num_labels=10, dataset_size=100, beta=0.99)
        assert ema.marginals.shape == (100, 10)
        assert (ema.marginals == 0.5).all()

    def test_global_init(self):
        ema = EMACenter(num_labels=10, dataset_size=100, beta=0.99, per_sample=False)
        assert ema.marginals.shape == (10,)

    def test_update_changes_marginals(self):
        ema = EMACenter(num_labels=10, dataset_size=50, beta=0.9)
        predictions = torch.ones(5, 10) * 0.8
        indices = torch.arange(5)

        ema.update(predictions, indices)
        # After first update, should be initialized directly to predictions
        retrieved = ema.get_marginals(indices)
        assert torch.allclose(retrieved, predictions, atol=1e-6)

        # Second update with different values should apply EMA
        predictions2 = torch.ones(5, 10) * 0.2
        ema.update(predictions2, indices)
        retrieved2 = ema.get_marginals(indices)
        # Should be between 0.2 and 0.8
        assert (retrieved2 > 0.15).all() and (retrieved2 < 0.85).all()

    def test_warmup_beta_annealing(self):
        ema = EMACenter(
            num_labels=5, dataset_size=10,
            beta=0.99, warmup_beta=0.5, warmup_steps=100,
        )
        assert abs(ema.current_beta() - 0.5) < 1e-6  # at step 0

        ema.step_count.fill_(50)
        beta_mid = ema.current_beta()
        assert 0.5 < beta_mid < 0.99  # midway

        ema.step_count.fill_(100)
        assert abs(ema.current_beta() - 0.99) < 1e-6  # after warmup

    def test_initialize_from_labels(self):
        ema = EMACenter(num_labels=5, dataset_size=10, per_sample=False)
        labels = torch.tensor([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ]).float()

        ema.initialize_from_labels(labels)
        expected = labels.mean(dim=0)
        assert torch.allclose(ema.get_marginals(), expected)


class TestSampling:
    """Test sampling strategies."""

    def test_bernoulli_shape(self):
        preds = torch.rand(4, 20)
        samples = sample_bernoulli(preds, num_samples=16)
        assert samples.shape == (4, 16, 20)

    def test_bernoulli_binary(self):
        preds = torch.rand(4, 20)
        samples = sample_bernoulli(preds, num_samples=8)
        # All values should be 0 or 1
        assert ((samples == 0) | (samples == 1)).all()

    def test_gaussian_noise_shape(self):
        preds = torch.rand(4, 20)
        samples = sample_gaussian_noise(preds, num_samples=16, sigma=0.3)
        assert samples.shape == (4, 16, 20)

    def test_gaussian_noise_bounded(self):
        preds = torch.rand(4, 20)
        samples = sample_gaussian_noise(preds, num_samples=16, sigma=0.3)
        assert (samples >= 0).all() and (samples <= 1).all()


class TestNCELoss:
    """Test NCE ranking loss."""

    def test_forward_shape(self):
        nce = NCELoss(num_samples=8)
        energy_gt = torch.randn(4)
        energy_neg = torch.randn(4, 8)
        log_prob_gt = torch.randn(4)
        log_prob_neg = torch.randn(4, 8)

        loss = nce(energy_gt, energy_neg, log_prob_gt, log_prob_neg)
        assert loss.shape == ()  # scalar

    def test_perfect_ranking_low_loss(self):
        """When ground truth has much higher score, loss should be low."""
        nce = NCELoss(num_samples=4)
        # GT has high score (energy - log_prob), negatives have low
        energy_gt = torch.tensor([10.0, 10.0])
        energy_neg = torch.tensor([[-5.0, -5.0, -5.0, -5.0],
                                    [-5.0, -5.0, -5.0, -5.0]])
        log_prob_gt = torch.zeros(2)
        log_prob_neg = torch.zeros(2, 4)

        loss = nce(energy_gt, energy_neg, log_prob_gt, log_prob_neg)
        assert loss.item() < 0.1  # should be very low

    def test_compute_log_prob(self):
        labels = torch.tensor([[1.0, 0.0, 1.0]])
        preds = torch.tensor([[0.9, 0.1, 0.8]])

        log_p = compute_log_prob(labels, preds)
        expected = (torch.log(torch.tensor(0.9)) +
                    torch.log(torch.tensor(0.9)) +
                    torch.log(torch.tensor(0.8)))
        assert torch.allclose(log_p, expected.unsqueeze(0), atol=1e-5)


class TestTaskLoss:
    """Test combined task-net loss."""

    def test_warmup_phase_bce_only(self):
        loss_fn = TaskLoss(lambda_energy=1.0, lambda_bce=1.0)
        energy = torch.randn(4)
        y_pred = torch.rand(4, 10)
        y_true = (torch.rand(4, 10) > 0.5).float()

        result = loss_fn(energy, y_pred, y_true, phase="warmup")
        assert result["energy"].item() == 0.0
        assert result["bce"].item() > 0.0

    def test_dynamic_phase_both(self):
        loss_fn = TaskLoss(lambda_energy=1.0, lambda_bce=1.0)
        energy = torch.randn(4, requires_grad=True)
        y_pred = torch.rand(4, 10, requires_grad=True)
        y_true = (torch.rand(4, 10) > 0.5).float()

        result = loss_fn(energy, y_pred, y_true, phase="dynamic")
        assert result["energy"].item() != 0.0
        assert result["bce"].item() > 0.0
        assert result["total"].requires_grad


class TestF1Metrics:
    """Test F1 computation."""

    def test_perfect_predictions(self):
        y_true = torch.tensor([[1, 0, 1], [0, 1, 0]]).float()
        y_pred = torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]]).float()

        metrics = compute_f1_metrics(y_pred, y_true)
        assert metrics["micro_f1"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["macro_f1"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["sample_f1"] == pytest.approx(1.0, abs=1e-6)

    def test_all_wrong(self):
        y_true = torch.tensor([[1, 0, 1], [0, 1, 0]]).float()
        y_pred = torch.tensor([[0.1, 0.9, 0.1], [0.9, 0.1, 0.9]]).float()

        metrics = compute_f1_metrics(y_pred, y_true)
        assert metrics["micro_f1"] == pytest.approx(0.0, abs=1e-6)


class TestEndToEnd:
    """Test that the full pipeline can execute without errors."""

    def test_forward_pass_crel(self):
        """Full forward pass: input → task-net → loss-net → energy."""
        input_dim, num_labels = 32, 15
        batch = 4

        task_net = TaskNet(input_dim, num_labels, hidden_dims=[64])
        loss_net = LossNet(input_dim, num_labels, energy_type="crel",
                          feature_hidden_dims=[64], rank=8,
                          label_embed_dim=8, proj_hidden_dim=16)

        x = torch.randn(batch, input_dim)
        y_pred = task_net(x)
        mu = torch.rand(batch, num_labels)

        energy = loss_net(x, y_pred, mu)
        assert energy.shape == (batch,)

        # Verify gradients flow to task-net
        energy.sum().backward()
        for p in task_net.parameters():
            assert p.grad is not None

    def test_forward_pass_seal(self):
        """Full forward pass with SEAL baseline."""
        input_dim, num_labels = 32, 15
        batch = 4

        task_net = TaskNet(input_dim, num_labels, hidden_dims=[64])
        loss_net = LossNet(input_dim, num_labels, energy_type="seal",
                          feature_hidden_dims=[64], global_hidden_dim=32)

        x = torch.randn(batch, input_dim)
        y_pred = task_net(x)
        mu = torch.rand(batch, num_labels)

        energy = loss_net(x, y_pred, mu)
        assert energy.shape == (batch,)

    def test_nce_training_step(self):
        """Simulate one NCE training step for the loss-net."""
        input_dim, num_labels = 32, 15
        batch = 4
        K = 8

        task_net = TaskNet(input_dim, num_labels, hidden_dims=[64])
        loss_net = LossNet(input_dim, num_labels, energy_type="crel",
                          feature_hidden_dims=[64], rank=8,
                          label_embed_dim=8, proj_hidden_dim=16)
        nce = NCELoss(num_samples=K)

        x = torch.randn(batch, input_dim)
        y_true = (torch.rand(batch, num_labels) > 0.5).float()
        mu = torch.rand(batch, num_labels)

        with torch.no_grad():
            y_pred = task_net(x)

        # Ground truth energy
        energy_gt = loss_net(x, y_true, mu)

        # Negative samples
        neg_samples = sample_bernoulli(y_pred, K)
        x_exp = x.unsqueeze(1).expand(batch, K, -1).reshape(batch * K, -1)
        neg_flat = neg_samples.reshape(batch * K, -1)
        mu_exp = mu.unsqueeze(1).expand(batch, K, -1).reshape(batch * K, -1)

        energy_neg = loss_net(x_exp, neg_flat, mu_exp).reshape(batch, K)

        # Log probabilities
        log_p_gt = compute_log_prob(y_true, y_pred)
        log_p_neg = compute_log_prob(neg_samples, y_pred.unsqueeze(1).expand_as(neg_samples))

        loss = nce(energy_gt, energy_neg, log_p_gt, log_p_neg)
        loss.backward()

        # Check loss-net has gradients
        grad_count = sum(1 for p in loss_net.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0, "No gradients flowed to loss-net"
