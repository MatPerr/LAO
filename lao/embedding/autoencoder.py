import math
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lao.embedding.graph_dataloader import get_dataloaders
from lao.graph.search_space import SearchSpace


def get_device() -> Any:
    """
    Get device.

    Args:
        None

    Returns:
        Any: Function output.
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


class NonNegativeLinear(nn.Linear):
    def __init__(self, in_features: Any, out_features: Any, bias: Any = True) -> None:
        """
        Init.

        Args:
            in_features (Any): Input parameter.
            out_features (Any): Input parameter.
            bias (Any): Input parameter.
        """
        super().__init__(in_features, out_features, bias)
        self.rectifier = nn.Softplus()
        self.in_features = in_features
        self.reset_parameters()

    def reset_parameters(self) -> Any:
        """
        Reset parameters.

        Args:
            None

        Returns:
            Any: Function output.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Any) -> Any:
        """Forward."""
        non_neg_weight = self.rectifier(self.weight)
        gain = 1.0 / self.in_features
        return gain * F.linear(input, non_neg_weight, self.bias)


class ICNN(nn.Module):
    def __init__(self, input_size: Any, hs: Any, convex_flag: Any = True) -> None:
        """
        Init.

        Args:
            input_size (Any): Input parameter.
            hs (Any): Input parameter.
            convex_flag (Any): Input parameter.
        """
        super(ICNN, self).__init__()
        self.input_size = input_size
        self.hs = hs
        self.activation = nn.ReLU()
        num_hidden = len(hs)
        self.w_zs = nn.ModuleList()
        for i in range(1, num_hidden):
            if convex_flag:
                self.w_zs.append(NonNegativeLinear(hs[i - 1], hs[i]))
            else:
                self.w_zs.append(nn.Linear(hs[i - 1], hs[i]))
        if convex_flag:
            self.w_zs.append(NonNegativeLinear(hs[-1], 1))
        else:
            self.w_zs.append(nn.Linear(hs[-1], 1))
        self.w_xs = nn.ModuleList()
        self.w_xs.append(nn.Linear(input_size, hs[0]))
        for i in range(1, num_hidden):
            self.w_xs.append(nn.Linear(input_size, hs[i]))
        self.w_xs.append(nn.Linear(input_size, 1))

    def forward(self, x: Any) -> Any:
        """
        Forward.

        Args:
            x (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        z = self.activation(self.w_xs[0](x))
        for i, (Wz, Wx) in enumerate(zip(self.w_zs[:-1], self.w_xs[1:-1])):
            z = Wz(z) + Wx(x)
            z = self.activation(z)
        y = self.activation(self.w_zs[-1](z) + self.w_xs[-1](x))
        y = y.squeeze(-1)
        if single:
            y = y.squeeze(0)
        return y


def MMD_RBF_gaussian_prior(
    z: Any, scale: Any = 1 / 8, adaptive: Any = True, unbiased: Any = True, standardized: Any = True
) -> Any:
    """
    Mmd rbf gaussian prior.

    Args:
        z (Any): Input parameter.
        scale (Any): Input parameter.
        adaptive (Any): Input parameter.
        unbiased (Any): Input parameter.
        standardized (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    batch_size = z.size(0)
    z_dim = z.size(1)
    norms2 = torch.sum(z**2, dim=1, keepdim=True)
    dotprods = torch.matmul(z, z.t())
    dists2 = norms2 + norms2.t() - 2 * dotprods
    dists2 = dists2.clamp(min=0.0)
    if adaptive:
        mean_norms2 = torch.mean(norms2)
        gamma2 = (scale * mean_norms2).detach()
    else:
        gamma2 = scale * z_dim
    Eknn = (gamma2 / (2.0 + gamma2)) ** (z_dim / 2.0)
    Ekzn = (gamma2 / (1.0 + gamma2)) ** (z_dim / 2.0) * torch.mean(torch.exp(-norms2 / (2.0 * (1.0 + gamma2))))
    if unbiased:
        Ekzz = (torch.sum(torch.exp(-dists2 / (2.0 * gamma2))) - batch_size) / (batch_size * batch_size - batch_size)
    else:
        Ekzz = torch.sum(torch.exp(-dists2 / (2.0 * gamma2))) / (batch_size * batch_size)
    MMD = Eknn - 2.0 * Ekzn + Ekzz
    if standardized:
        variance = (
            (gamma2 / (2.0 + gamma2)) ** z_dim
            + (gamma2 / (4.0 + gamma2)) ** (z_dim / 2.0)
            - 2.0 * (gamma2**2.0 / ((1.0 + gamma2) * (3.0 + gamma2))) ** (z_dim / 2.0)
        )
        variance = 2.0 * variance / (batch_size * (batch_size - 1.0))
        variance_normalization = variance ** (-1.0 / 2.0)
        return variance_normalization * MMD
    else:
        return MMD


class CodeNorm1d(nn.Module):
    """
    A custom normalization layer for 1D input that performs only the normalization
    part of BatchNorm1d without the learnable affine parameters.
    """

    def __init__(self, eps: Any = 1e-05, momentum: Any = 0.1, track_running_stats: Any = True) -> None:
        """
        Init.

        Args:
            eps (Any): Input parameter.
            momentum (Any): Input parameter.
            track_running_stats (Any): Input parameter.
        """
        super(CodeNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.tensor(0.0))
            self.register_buffer("running_var", torch.tensor(1.0))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

    def forward(self, input: Any) -> Any:
        """
        Forward.

        Args:
            input (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        if input.dim() != 2:
            raise ValueError(f"Expected 2D input (N, L), got {input.dim()}D input")
        if self.training or not self.track_running_stats:
            batch_mean = input.mean()
            batch_var = input.var(unbiased=False)
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                    self.num_batches_tracked += 1
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        input_normalized = (input - mean) / torch.sqrt(var + self.eps)
        return input_normalized

    def extra_repr(self) -> Any:
        """Extra repr."""
        return f"eps={self.eps}, momentum={self.momentum}, track_running_stats={self.track_running_stats}"


class CosineAnnealingAlphaLR(_LRScheduler):
    def __init__(self, optimizer: Any, T_max: Any, alpha: Any, last_epoch: Any = -1) -> None:
        """Init."""
        self.T_max = T_max
        self.alpha = alpha
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> Any:
        """Get lr."""
        return [
            base_lr * (self.alpha + 0.5 * (1 - self.alpha) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)))
            for base_lr in self.base_lrs
        ]


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: Any) -> None:
        """Init."""
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x: Any) -> Any:
        """
        Forward.

        Args:
            x (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


def batch_quantized_decode(quantized_values_batch: Any, possible_values: Any) -> Any:
    """
    Batch quantized decode.

    Args:
        quantized_values_batch (Any): Input parameter.
        possible_values (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    import torch

    if not isinstance(possible_values, torch.Tensor):
        possible_values = torch.tensor(possible_values, device=quantized_values_batch.device)
    n_bins = len(possible_values)
    if n_bins == 1:
        return torch.full_like(quantized_values_batch, possible_values[0])
    bin_width = 1.0 / n_bins
    clipped_values = torch.clamp(quantized_values_batch, 0, 1)
    bin_indices = torch.min(
        torch.floor(clipped_values / bin_width).long(), torch.tensor(n_bins - 1, device=quantized_values_batch.device)
    )
    decoded_values = possible_values[bin_indices]
    return decoded_values


def ordered_logistic_loss(predictions: Any, targets: Any, possible_values: Any, device: Any = None) -> Any:
    """
    Ordered logistic loss.

    Args:
        predictions (Any): Input parameter.
        targets (Any): Input parameter.
        possible_values (Any): Input parameter.
        device (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    if device is None:
        device = predictions.device
    batch_size, n_nodes, n_features = predictions.shape
    total_loss = 0.0
    for j in range(n_features):
        feature_values = possible_values[j]
        n_bins = len(feature_values)
        if n_bins <= 1:
            continue
        bin_width = 1.0 / n_bins
        bin_edges = torch.tensor([i * bin_width for i in range(n_bins + 1)], device=device)
        feature_preds = predictions[:, :, j].reshape(-1)
        feature_targets = targets[:, :, j].reshape(-1)
        target_indices = torch.zeros_like(feature_targets, dtype=torch.long)
        for i, val in enumerate(feature_values):
            target_indices[feature_targets == val] = i
        cumulative_probs = []
        for threshold in bin_edges[1:-1]:
            prob = torch.sigmoid(feature_preds - threshold)
            cumulative_probs.append(prob)
        if not cumulative_probs:
            threshold = 0.5
            probs = torch.sigmoid(feature_preds - threshold)
            loss = F.binary_cross_entropy(probs, target_indices.float())
            total_loss += loss
            continue
        probs = torch.stack(cumulative_probs, dim=1)
        bin_probs = torch.zeros((feature_preds.size(0), n_bins), device=device)
        bin_probs[:, 0] = 1.0 - probs[:, 0]
        for k in range(1, n_bins - 1):
            bin_probs[:, k] = probs[:, k - 1] - probs[:, k]
        bin_probs[:, -1] = probs[:, -1] if probs.size(1) > 0 else 1.0 - bin_probs[:, 0]
        bin_probs = torch.clamp(bin_probs, min=1e-08, max=1.0 - 1e-08)
        loss = F.nll_loss(torch.log(bin_probs), target_indices)
        total_loss += loss
    return total_loss / n_features


class ArcAE(nn.Module):
    def __init__(
        self,
        search_space: Any,
        ae_type: Any = "WAE",
        encoder_hs: Any = [1024] * 4,
        decoder_hs: Any = [1024] * 4,
        z_dim: Any = 64,
    ) -> None:
        """
        Init.

        Args:
            search_space (Any): Input parameter.
            ae_type (Any): Input parameter.
            encoder_hs (Any): Input parameter.
            decoder_hs (Any): Input parameter.
            z_dim (Any): Input parameter.
        """
        super().__init__()
        self.search_space = search_space
        self.ae_type = ae_type
        self.z_dim = z_dim
        self.bounds = torch.zeros(2, z_dim)
        self.bounds[0, :] = float("inf")
        self.bounds[1, :] = float("-inf")
        self.node_encoding_type = search_space.graph_features.node_encoding_type
        self.n_nodes = search_space.graph_features.n_nodes[0]
        self.nA = self.n_nodes * (self.n_nodes - 1) // 2
        if self.node_encoding_type == "categorical":
            self.nX = 0
            self.feature_widths = []
            for feature_values in search_space.node_features.__dict__.values():
                w = len(feature_values)
                self.nX += w * self.n_nodes
                self.feature_widths.append(w)
        elif self.node_encoding_type == "quantized":
            self.nX = len(search_space.node_features.__dict__) * self.n_nodes
            self.feature_widths = None
        self.input_size = self.nA + self.nX
        encoder_layers = []
        prev_size = self.input_size
        encoder_layers.append(nn.Linear(prev_size, encoder_hs[0]))
        encoder_layers.append(nn.BatchNorm1d(encoder_hs[0]))
        encoder_layers.append(nn.ReLU())
        for i in range(len(encoder_hs) - 1):
            if encoder_hs[i] == encoder_hs[i + 1]:
                encoder_layers.append(ResidualBlock(encoder_hs[i]))
            else:
                encoder_layers.append(nn.Linear(encoder_hs[i], encoder_hs[i + 1]))
                encoder_layers.append(nn.BatchNorm1d(encoder_hs[i + 1]))
                encoder_layers.append(nn.ReLU())
        self.encoder_base = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(encoder_hs[-1], z_dim)
        if ae_type == "VAE":
            self.fc_logvar = nn.Linear(encoder_hs[-1], z_dim)
        if ae_type == "WAE":
            self.cn = CodeNorm1d(z_dim)
        decoder_layers = []
        prev_size = z_dim
        decoder_layers.append(nn.Linear(prev_size, decoder_hs[0]))
        decoder_layers.append(nn.BatchNorm1d(decoder_hs[0]))
        decoder_layers.append(nn.ReLU())
        for i in range(len(decoder_hs) - 1):
            if decoder_hs[i] == decoder_hs[i + 1]:
                decoder_layers.append(ResidualBlock(decoder_hs[i]))
            else:
                decoder_layers.append(nn.Linear(decoder_hs[i], decoder_hs[i + 1]))
                decoder_layers.append(nn.BatchNorm1d(decoder_hs[i + 1]))
                decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(decoder_hs[-1], self.input_size))
        self.decoder_base = nn.Sequential(*decoder_layers)
        self.params_predictor = ICNN(z_dim, [512] * 2, convex_flag=False)
        self.FLOPs_predictor = ICNN(z_dim, [512] * 2, convex_flag=False)
        self.BBGP_predictor = ICNN(z_dim, [512] * 2, convex_flag=False)

    def encode(self, v: Any) -> Any:
        """
        Encode.

        Args:
            v (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        v = self.encoder_base(v)
        mu = self.fc_mu(v)
        logvar = None
        if self.ae_type == "VAE":
            logvar = self.fc_logvar(v)
        if self.ae_type == "WAE":
            mu = self.cn(mu)
        return (mu, logvar)

    def reparameterize(self, mu: Any, logvar: Any) -> Any:
        """
        Reparameterize.

        Args:
            mu (Any): Input parameter.
            logvar (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        if self.ae_type == "VAE":
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            else:
                return mu
        else:
            return mu

    def decode(self, z: Any) -> Any:
        """
        Decode.

        Args:
            z (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        batch_size = z.shape[0]
        h = self.decoder_base(z)
        v_X, v_A = torch.split(h, [self.nX, self.nA], dim=1)
        v_A = (nn.Sigmoid()(v_A) > 0.5) * 1.0
        if self.node_encoding_type == "categorical":
            v_X_parts = torch.split(v_X.view(batch_size, self.n_nodes, -1), self.feature_widths, dim=2)
            v_X_processed = []
            for part in v_X_parts:
                part_softmax = nn.functional.softmax(part, dim=2)
                _, indices = torch.max(part_softmax, dim=2, keepdim=True)
                part_one_hot = torch.zeros_like(part_softmax)
                part_one_hot.scatter_(2, indices, 1.0)
                v_X_processed.append(part_one_hot)
            v_X = torch.cat(v_X_processed, dim=2)
        elif self.node_encoding_type == "quantized":
            v_X = v_X.view(batch_size, self.n_nodes, -1)
        return torch.cat([v_X.view(batch_size, -1), v_A], dim=1)

    def forward(self, v: Any) -> Any:
        """
        Forward.

        Args:
            v (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        mu, logvar = self.encode(v)
        z = self.reparameterize(mu, logvar)
        v_hat = self.decode(z)
        return v_hat

    def _update_bounds(self, z: Any) -> Any:
        """
        Update bounds.

        Args:
            z (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        with torch.no_grad():
            if self.bounds.device != z.device:
                self.bounds = self.bounds.to(z.device)
            batch_min = torch.min(z, dim=0)[0]
            self.bounds[0, :] = torch.minimum(self.bounds[0, :], batch_min)
            batch_max = torch.max(z, dim=0)[0]
            self.bounds[1, :] = torch.maximum(self.bounds[1, :], batch_max)

    def loss(self, v: Any, y: Any) -> Any:
        """
        Loss.

        Args:
            v (Any): Input parameter.
            y (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        batch_size = v.shape[0]
        mu, logvar = self.encode(v)
        z = self.reparameterize(mu, logvar)
        self._update_bounds(z)
        h = self.decoder_base(z)
        v_X_hat, v_A_hat = torch.split(h, [self.nX, self.nA], dim=1)
        v_X, v_A = torch.split(v, [self.nX, self.nA], dim=1)
        v_A_hat = nn.Sigmoid()(v_A_hat)
        edge_loss = nn.BCELoss()(v_A_hat, v_A)
        v_A_hat_binary = (v_A_hat > 0.5).float()
        adj_correct = torch.all(v_A_hat_binary == v_A, dim=1)
        edge_acc = adj_correct.float().mean()
        if self.node_encoding_type == "categorical":
            v_X_hat_parts = torch.split(v_X_hat.view(batch_size, self.n_nodes, -1), self.feature_widths, dim=2)
            v_X_parts = torch.split(v_X.view(batch_size, self.n_nodes, -1), self.feature_widths, dim=2)
            node_loss = 0
            all_features_correct = torch.ones(batch_size, dtype=torch.bool, device=v.device)
            for part, target_part in zip(v_X_hat_parts, v_X_parts):
                target_indices = torch.argmax(target_part, dim=2)
                node_loss += nn.CrossEntropyLoss()(part.reshape(-1, part.size(2)), target_indices.reshape(-1))
                predicted_indices = torch.argmax(part, dim=2)
                nodes_correct_for_feature = torch.all(predicted_indices == target_indices, dim=1)
                all_features_correct &= nodes_correct_for_feature
        elif self.node_encoding_type == "quantized":
            node_loss = torch.sqrt(self.n_nodes * nn.MSELoss()(v_X_hat, v_X))
            feature_count = len(self.search_space.node_features.__dict__)
            v_X_hat_reshaped = v_X_hat.reshape(batch_size, self.n_nodes, feature_count)
            v_X_reshaped = v_X.reshape(batch_size, self.n_nodes, feature_count)
            all_match = torch.ones(batch_size, dtype=torch.bool, device=v.device)
            for j, feature_name in enumerate(self.search_space.node_features.__dict__):
                feature_values = getattr(self.search_space.node_features, feature_name)
                pred_feature = v_X_hat_reshaped[:, :, j].reshape(-1, 1)
                true_feature = v_X_reshaped[:, :, j].reshape(-1, 1)
                pred_decoded = batch_quantized_decode(pred_feature, feature_values).reshape(batch_size, self.n_nodes)
                true_decoded = batch_quantized_decode(true_feature, feature_values).reshape(batch_size, self.n_nodes)
                feature_match = torch.all(pred_decoded == true_decoded, dim=1)
                all_match = all_match & feature_match
            all_features_correct = all_match
        node_acc = all_features_correct.float().mean()
        recon_loss = node_loss + edge_loss
        perfect_reconstructions = adj_correct & all_features_correct
        recon_acc = perfect_reconstructions.float().mean()
        pred_params = self.params_predictor(z)
        pred_FLOPs = self.FLOPs_predictor(z)
        pred_BBGP = self.BBGP_predictor(z)
        params_loss = nn.MSELoss()(pred_params, torch.log(y[:, 0]))
        FLOPs_loss = nn.MSELoss()(pred_FLOPs, torch.log(y[:, 1]))
        BBGP_loss = nn.MSELoss()(pred_BBGP, torch.log2(y[:, 2]))
        pred_loss = params_loss + FLOPs_loss + BBGP_loss
        if self.ae_type == "VAE":
            reg = (0.5 * (mu**2 + logvar.exp() - 1 - logvar)).sum(1).mean()
            reg_name = "KL"
        elif self.ae_type == "WAE":
            reg = MMD_RBF_gaussian_prior(z, adaptive=False)
            reg_name = "MMD"
        elif self.ae_type == "AE":
            reg = torch.maximum(torch.abs(z) - 3.0, torch.zeros_like(z)).pow(2).sum(1).mean()
            reg += z.mean(0).pow(2).sum()
            reg_name = "L2"
        total_loss = 0
        total_loss += recon_loss
        if self.beta > 0:
            total_loss += self.beta * reg
        if self.gamma > 0:
            total_loss += self.gamma * pred_loss
        return {
            "loss": total_loss,
            "recon": recon_loss,
            reg_name: reg,
            "pred": pred_loss,
            "recon_acc": recon_acc,
            "node_loss": node_loss,
            "edge_loss": edge_loss,
            "node_acc": node_acc,
            "edge_acc": edge_acc,
            "params_loss": params_loss,
            "FLOPs_loss": FLOPs_loss,
            "BBGP_loss": BBGP_loss,
        }

    def log_latent_space(
        self, train_dl: Any, writer: Any, global_step: Any, device: Any, num_samples: Any = 10000
    ) -> Any:
        """
        Log latent space.

        Args:
            train_dl (Any): Input parameter.
            writer (Any): Input parameter.
            global_step (Any): Input parameter.
            device (Any): Input parameter.
            num_samples (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        self.eval()
        all_z = []
        sample_count = 0
        with torch.no_grad():
            for v, _ in tqdm(train_dl, desc="Encoding inputs for latent histograms", leave=False):
                if sample_count >= num_samples:
                    break
                v = v.to(device)
                mu, logvar = self.encode(v)
                z = self.reparameterize(mu, logvar)
                all_z.append(z.cpu())
                sample_count += v.size(0)
        all_z = torch.cat(all_z, dim=0)[:num_samples]
        z_dim = all_z.size(1)
        for i in range(z_dim):
            writer.add_histogram(f"latent/z_dim_{i}", all_z[:, i], global_step)
        self.train()
        return all_z

    def viz_decoded_prior(
        self, writer: Any, global_step: Any, device: Any, train_dl: Any, num_samples: Any = 10000
    ) -> Any:
        """
        Viz decoded prior.

        Args:
            writer (Any): Input parameter.
            global_step (Any): Input parameter.
            device (Any): Input parameter.
            train_dl (Any): Input parameter.
            num_samples (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        import io

        import matplotlib.pyplot as plt
        import torch
        from PIL import Image
        from torchvision.transforms import ToTensor
        from tqdm import tqdm

        self.eval()
        n_nodes = self.search_space.graph_features.n_nodes[0]
        feature_names = list(self.search_space.node_features.__dict__.keys())
        feature_values = list(self.search_space.node_features.__dict__.values())
        with torch.no_grad():
            z_samples = torch.randn(num_samples, self.z_dim, device=device)
            decoded_vectors = self.decode(z_samples)
            batch_size = decoded_vectors.shape[0]
            decoded_X, decoded_A = torch.split(decoded_vectors, [self.nX, self.nA], dim=1)
            decoded_A = decoded_A.cpu().numpy()
            if self.node_encoding_type == "categorical":
                decoded_X_parts = torch.split(decoded_X.view(batch_size, n_nodes, -1), self.feature_widths, dim=2)
                decoded_X_parts = [part.cpu().numpy() for part in decoded_X_parts]
                decoded_features = np.empty((batch_size, n_nodes, len(feature_names)), dtype="object")
                for j, (feature_part, possible_values) in enumerate(zip(decoded_X_parts, feature_values)):
                    predicted_indices = np.argmax(feature_part, axis=2)
                    for b in range(batch_size):
                        for i in range(n_nodes):
                            idx = predicted_indices[b, i]
                            decoded_features[b, i, j] = possible_values[idx]
            elif self.node_encoding_type == "quantized":
                decoded_X = decoded_X.view(batch_size, n_nodes, -1)
                decoded_features = np.empty((batch_size, n_nodes, len(feature_names)), dtype="object")
                for j, possible_values in enumerate(feature_values):
                    feature_flat = decoded_X[:, :, j].reshape(-1, 1)
                    decoded_flat = batch_quantized_decode(feature_flat, possible_values)
                    decoded_flat_np = decoded_flat.cpu().numpy().flatten()
                    decoded_features[:, :, j] = decoded_flat_np.reshape(batch_size, n_nodes)
        train_features = np.empty((0, n_nodes, len(feature_names)), dtype="object")
        train_adj = np.empty((0, self.nA), dtype=np.float32)
        max_batches = min(30, len(train_dl))
        samples_collected = 0
        with torch.no_grad():
            for batch_idx, (v, _) in enumerate(tqdm(train_dl, desc="Processing training data", leave=False)):
                if batch_idx >= max_batches or samples_collected >= num_samples:
                    break
                v = v.to(device)
                batch_size = v.shape[0]
                samples_collected += batch_size
                train_X, train_A = torch.split(v, [self.nX, self.nA], dim=1)
                train_adj = np.vstack([train_adj, train_A.cpu().numpy()])
                if self.node_encoding_type == "categorical":
                    train_X_parts = torch.split(train_X.view(batch_size, n_nodes, -1), self.feature_widths, dim=2)
                    batch_features = np.empty((batch_size, n_nodes, len(feature_names)), dtype="object")
                    for j, (feature_part, possible_values) in enumerate(zip(train_X_parts, feature_values)):
                        one_hot_indices = torch.argmax(feature_part, dim=2).cpu().numpy()
                        for b in range(batch_size):
                            for i in range(n_nodes):
                                idx = one_hot_indices[b, i]
                                batch_features[b, i, j] = possible_values[idx]
                elif self.node_encoding_type == "quantized":
                    train_X = train_X.view(batch_size, n_nodes, -1)
                    batch_features = np.empty((batch_size, n_nodes, len(feature_names)), dtype="object")
                    for j, possible_values in enumerate(feature_values):
                        feature_flat = train_X[:, :, j].reshape(-1, 1)
                        decoded_flat = batch_quantized_decode(feature_flat, possible_values)
                        decoded_flat_np = decoded_flat.cpu().numpy().flatten()
                        batch_features[:, :, j] = decoded_flat_np.reshape(batch_size, n_nodes)
                train_features = np.vstack([train_features, batch_features])
                if samples_collected >= num_samples:
                    break
        train_features = train_features[:num_samples]
        train_adj = train_adj[:num_samples]
        fig_width = 5 * len(feature_names)
        fig_height = 4 * n_nodes
        fig, axes = plt.subplots(n_nodes, len(feature_names), figsize=(fig_width, fig_height))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        if n_nodes == 1:
            axes = np.array([axes])
        if len(feature_names) == 1:
            axes = axes.reshape(-1, 1)
        prior_color = "royalblue"
        train_color = "mediumseagreen"
        for i in range(n_nodes):
            for j, (feature_name, feature_vals) in enumerate(zip(feature_names, feature_values)):
                ax = axes[i, j]
                prior_values = decoded_features[:, i, j]
                train_values = train_features[:, i, j]
                prior_unique, prior_counts = np.unique(prior_values, return_counts=True)
                prior_probs = np.zeros(len(feature_vals))
                for val_idx, val in enumerate(feature_vals):
                    matching_idx = np.where(prior_unique == val)[0]
                    if len(matching_idx) > 0:
                        prior_probs[val_idx] = prior_counts[matching_idx[0]] / len(prior_values)
                train_unique, train_counts = np.unique(train_values, return_counts=True)
                train_probs = np.zeros(len(feature_vals))
                for val_idx, val in enumerate(feature_vals):
                    matching_idx = np.where(train_unique == val)[0]
                    if len(matching_idx) > 0:
                        train_probs[val_idx] = train_counts[matching_idx[0]] / len(train_values)
                x = np.arange(len(feature_vals))
                width = 0.35
                ax.bar(x - width / 2, prior_probs, width, label="Generated", color=prior_color, alpha=0.8)
                ax.bar(x + width / 2, train_probs, width, label="Training", color=train_color, alpha=0.8)
                ax.set_xticks(x)
                if isinstance(feature_vals[0], int) or isinstance(feature_vals[0], float):
                    ax.set_xticklabels(feature_vals)
                else:
                    ax.set_xticklabels([str(val) for val in feature_vals])
                alias = self.search_space.aliases.get(feature_name, feature_name)
                ax.set_title(f"Node {i}: {alias}")
                ax.set_ylabel("Probability")
                if i == 0 and j == 0:
                    ax.legend()
                if len(feature_vals) > 4:
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.suptitle(f"Distribution of Node Features: Generated vs Training (Step {global_step})", fontsize=16)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image = ToTensor()(Image.open(buf))
        writer.add_image("features_distribution_comparison", image, global_step)
        plt.close(fig)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        prior_avg_adj = np.mean(decoded_A, axis=0)
        prior_adj_matrix = np.zeros((n_nodes, n_nodes))
        adj_idx = 0
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                prior_adj_matrix[i, j] = prior_avg_adj[adj_idx]
                adj_idx += 1
        train_avg_adj = np.mean(train_adj, axis=0)
        train_adj_matrix = np.zeros((n_nodes, n_nodes))
        adj_idx = 0
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                train_adj_matrix[i, j] = train_avg_adj[adj_idx]
                adj_idx += 1
        im1 = ax1.imshow(prior_adj_matrix, cmap="Blues", vmin=0, vmax=1)
        ax1.set_title("Generated Adjacency Matrix")
        ax1.set_xlabel("Node Index")
        ax1.set_ylabel("Node Index")
        fig.colorbar(im1, ax=ax1, label="Connection Probability")
        for i in range(n_nodes):
            for j in range(n_nodes):
                if prior_adj_matrix[i, j] > 0:
                    ax1.text(
                        j,
                        i,
                        f"{prior_adj_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black" if prior_adj_matrix[i, j] < 0.5 else "white",
                    )
        im2 = ax2.imshow(train_adj_matrix, cmap="Greens", vmin=0, vmax=1)
        ax2.set_title("Training Data Adjacency Matrix")
        ax2.set_xlabel("Node Index")
        ax2.set_ylabel("Node Index")
        fig.colorbar(im2, ax=ax2, label="Connection Probability")
        for i in range(n_nodes):
            for j in range(n_nodes):
                if train_adj_matrix[i, j] > 0:
                    ax2.text(
                        j,
                        i,
                        f"{train_adj_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black" if train_adj_matrix[i, j] < 0.5 else "white",
                    )
        fig.suptitle(f"Adjacency Matrix Comparison (Step {global_step})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image = ToTensor()(Image.open(buf))
        writer.add_image("adjacency_comparison", image, global_step)
        plt.close()
        if n_nodes <= 20:
            plt.figure(figsize=(12, 8))
            edge_labels = []
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    edge_labels.append(f"{i}->{j}")
            indices = np.arange(len(edge_labels))
            width = 0.35
            plt.bar(indices - width / 2, prior_avg_adj, width, label="Generated", color=prior_color, alpha=0.8)
            plt.bar(indices + width / 2, train_avg_adj, width, label="Training", color=train_color, alpha=0.8)
            plt.xlabel("Edge")
            plt.ylabel("Connection Probability")
            plt.title("Edge Connection Probabilities: Generated vs Training")
            plt.xticks(indices, edge_labels, rotation=90)
            plt.legend()
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            image = ToTensor()(Image.open(buf))
            writer.add_image("edge_probabilities_comparison", image, global_step)
            plt.close()
        self.train()
        return

    def train_loop(
        self,
        train_dl: Any,
        val_dl: Any,
        iterations: Any,
        lr: Any = 0.0005,
        log_dir: Any = "runs/arc_ae",
        beta: Any = 0.01,
        gamma: Any = 0.001,
        save_dir: Any = "checkpoints",
        save_every: Any = 50000,
        val_every: Any = 15000,
        log_latent_every: Any = 15000,
    ) -> Any:
        """
        Train loop.

        Args:
            train_dl (Any): Input parameter.
            val_dl (Any): Input parameter.
            iterations (Any): Input parameter.
            lr (Any): Input parameter.
            log_dir (Any): Input parameter.
            beta (Any): Input parameter.
            gamma (Any): Input parameter.
            save_dir (Any): Input parameter.
            save_every (Any): Input parameter.
            val_every (Any): Input parameter.
            log_latent_every (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        self.beta = beta
        self.gamma = gamma
        device = get_device()
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = CosineAnnealingAlphaLR(optimizer, T_max=0.9 * iterations, alpha=0.0001)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_dir, f"arcae_{timestamp}")
        writer = SummaryWriter(os.path.join(log_dir, f"{timestamp}"))
        os.makedirs(save_dir, exist_ok=True)
        self.train()
        global_step = 0
        pbar = tqdm(range(iterations), desc="Training")
        best_val_loss = float("inf")
        while global_step < iterations:
            for v, y in train_dl:
                v = v.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                loss_dict = self.loss(v, y)
                loss = loss_dict["loss"]
                loss.backward()
                optimizer.step()
                scheduler.step()
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        writer.add_scalar(f"train/{key}", value.item(), global_step)
                desc_parts = [""]
                for key in list(loss_dict.keys())[0:5]:
                    value = loss_dict[key]
                    if isinstance(value, torch.Tensor):
                        desc_parts.append(f"{key}: {value.item():.2e}")
                pbar.set_description(" ".join(desc_parts))
                if (global_step + 1) % log_latent_every == 0:
                    tqdm.write(f"Logging latent space distributions at step {global_step + 1}")
                    self.log_latent_space(train_dl, writer, global_step, device)
                    self.viz_decoded_prior(writer, global_step, device, train_dl)
                if (global_step + 1) % val_every == 0:
                    self.eval()
                    val_losses = {k: 0.0 for k in loss_dict.keys()}
                    val_count = 0
                    with torch.no_grad():
                        for val_v, val_y in tqdm(val_dl, desc="Validating", leave=False):
                            val_v = val_v.to(device)
                            val_y = val_y.to(device)
                            val_loss_dict = self.loss(val_v, val_y)
                            for k, v in val_loss_dict.items():
                                if isinstance(v, torch.Tensor):
                                    val_losses[k] += v.item()
                            val_count += 1
                    for k in val_losses.keys():
                        val_losses[k] /= max(val_count, 1)
                        writer.add_scalar(f"val/{k}", val_losses[k], global_step)
                    val_msg_parts = [f"Iteration {global_step + 1}/{iterations} |"]
                    for k, v in val_losses.items():
                        val_msg_parts.append(f"Val {k}: {v:.2e}")
                    tqdm.write(" ".join(val_msg_parts))
                    if val_losses["loss"] < best_val_loss:
                        best_val_loss = val_losses["loss"]
                        best_checkpoint_path = os.path.join(save_dir, "arcae_best.pt")
                        torch.save(
                            {
                                "iteration": global_step + 1,
                                "model_state_dict": self.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "val_loss": best_val_loss,
                                "all_losses": val_losses,
                            },
                            best_checkpoint_path,
                        )
                        tqdm.write(f"New best model saved with validation loss: {best_val_loss:.4f}")
                    self.train()
                if (global_step + 1) % save_every == 0:
                    checkpoint_path = os.path.join(save_dir, f"arcae_iter_{global_step + 1}.pt")
                    torch.save(
                        {
                            "iteration": global_step + 1,
                            "model_state_dict": self.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss.item(),
                            "all_losses": {
                                k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()
                            },
                        },
                        checkpoint_path,
                    )
                    tqdm.write(f"Checkpoint saved to {checkpoint_path}")
                global_step += 1
                pbar.update(1)
                if global_step >= iterations:
                    break
        tqdm.write("Logging latent space distributions for final model")
        self.log_latent_space(train_dl, writer, global_step, device)
        self.viz_decoded_prior(writer, global_step, device, train_dl)
        final_checkpoint_path = os.path.join(save_dir, "arcae_final.pt")
        torch.save(
            {
                "iteration": iterations,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item() if "loss" in locals() else None,
                "all_losses": {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
                if "loss_dict" in locals()
                else {},
            },
            final_checkpoint_path,
        )
        tqdm.write(f"Final model saved to {final_checkpoint_path}")
        writer.close()
        return val_losses


def main() -> Any:
    """
    Main.

    Args:
        None

    Returns:
        Any: Function output.
    """
    data = torch.load("exp1404/graph_data/1000000_samples_0415_0240.pt")
    train_dl, val_dl = get_dataloaders(data["V"], data["Y"], train_split=0.99, batch_size=1024, num_workers=0)
    ae_model = ArcAE(search_space=SearchSpace(), z_dim=79, ae_type="WAE")
    ae_model.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        iterations=150000,
        lr=0.0005,
        beta=0.01,
        gamma=0.001,
        log_dir="runs/arc_ae",
        save_dir="checkpoints",
    )


if __name__ == "__main__":
    main()
