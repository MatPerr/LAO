from graph_utils import ArcGraph
from search_space import SearchSpace
from graph_dataloader import get_dataloaders

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime
from tqdm import tqdm
import math


def get_device():
        if torch.backends.mps.is_available():
            return "mps"
        
        elif torch.cuda.is_available():
            return f"cuda:0"
        
        else: 
            return "cpu"


class NonNegativeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.rectifier = nn.Softplus()
        self.in_features = in_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        non_neg_weight = self.rectifier(self.weight) # Apply Softplus to ensure non-negative weights
        gain = 1.0 / self.in_features
        return gain * F.linear(input, non_neg_weight, self.bias)
    

class ICNN(nn.Module):
    def __init__(self, input_size, hs, convex_flag = True):
        super(ICNN, self).__init__()
        self.input_size = input_size
        self.hs = hs
        self.activation = nn.ReLU()
        num_hidden = len(hs)
        
        self.w_zs = nn.ModuleList()
        for i in range(1, num_hidden):
            if convex_flag:
                self.w_zs.append(NonNegativeLinear(hs[i-1], hs[i]))
            else:
                self.w_zs.append(nn.Linear(hs[i-1], hs[i]))
        if convex_flag:
            self.w_zs.append(NonNegativeLinear(hs[-1], 1))
        else:
            self.w_zs.append(nn.Linear(hs[-1], 1))
            
        self.w_xs = nn.ModuleList()
        self.w_xs.append(nn.Linear(input_size, hs[0]))
        for i in range(1, num_hidden):
            self.w_xs.append(nn.Linear(input_size, hs[i]))
        self.w_xs.append(nn.Linear(input_size, 1))
        
    def forward(self, x):
        single = (x.dim() == 1)
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
    

def SMMD_RBF_gaussian_prior(z, scale=1/8, adaptive=True):
    batch_size = z.size(0)
    z_dim = z.size(1)
    norms2 = torch.sum(z ** 2, dim=1, keepdim=True)
    dotprods = torch.matmul(z, z.t())
    dists2 = norms2 + norms2.t() - 2*dotprods
    dists2 = dists2.clamp(min = 0.0)

    if adaptive:
        mean_norms2 = torch.mean(norms2)
        gamma2 = (scale * mean_norms2).detach()
    else:
        gamma2 = scale * z_dim

    variance = (
        (gamma2 / (2. + gamma2)) ** z_dim +
        (gamma2 / (4. + gamma2)) ** (z_dim / 2.) -
        2. * (gamma2 ** 2. / ((1. + gamma2) * (3. + gamma2))) ** (z_dim / 2.)
    )

    variance = 2. * variance / (batch_size * (batch_size - 1.))
    variance_normalization = (variance) ** (-1. / 2.)
    Ekzz = (torch.sum(torch.exp(-dists2 / (2. * gamma2))) - batch_size) / (batch_size * batch_size - batch_size)
    Ekzn = (
        (gamma2 / (1. + gamma2)) ** (z_dim / 2.) *
        torch.mean(torch.exp(-norms2 / (2. * (1. + gamma2))))
    )
    Eknn = (gamma2 / (2. + gamma2)) ** (z_dim / 2.)
    
    return torch.clamp(variance_normalization * (Ekzz - 2. * Ekzn + Eknn), min=0.0)


class CosineAnnealingAlphaLR(_LRScheduler):
    def __init__(self, optimizer, T_max, alpha, last_epoch=-1):
        self.T_max = T_max
        self.alpha = alpha
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * (self.alpha + 0.5 * (1 - self.alpha) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)))
            for base_lr in self.base_lrs
        ]


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Add the residual connection
        out = self.relu(out)
        return out
    

class ArcAE(nn.Module):
    def __init__(self,
                 search_space,
                 ae_type = "WAE",
                 encoder_hs=[1024]*4,
                 decoder_hs=[1024]*4,
                 z_dim=64):
        super().__init__()
        self.search_space = search_space
        self.ae_type = ae_type
        self.z_dim = z_dim

        self.n_nodes = search_space.graph_features.n_nodes[0]
        self.nA = (self.n_nodes*(self.n_nodes - 1))//2
        self.nX = 0
        self.feature_widths = []
        for feature_values in search_space.node_features.__dict__.values():
            w = len(feature_values)
            self.nX += w*self.n_nodes
            self.feature_widths.append(w)
        self.input_size = self.nA + self.nX
        
        # Encoder 
        encoder_layers = []
        prev_size = self.input_size
        encoder_layers.append(nn.Linear(prev_size, encoder_hs[0]))
        encoder_layers.append(nn.BatchNorm1d(encoder_hs[0]))
        encoder_layers.append(nn.ReLU())
        for i in range(len(encoder_hs) - 1):
            if encoder_hs[i] == encoder_hs[i+1]:
                encoder_layers.append(ResidualBlock(encoder_hs[i]))
            else:
                encoder_layers.append(nn.Linear(encoder_hs[i], encoder_hs[i+1]))
                encoder_layers.append(nn.BatchNorm1d(encoder_hs[i+1]))
                encoder_layers.append(nn.ReLU())
        self.encoder_base = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(encoder_hs[-1], z_dim)
        if ae_type == 'VAE':
            self.fc_logvar = nn.Linear(encoder_hs[-1], z_dim)

        # Decoder
        decoder_layers = []
        prev_size = z_dim
        decoder_layers.append(nn.Linear(prev_size, decoder_hs[0]))
        decoder_layers.append(nn.BatchNorm1d(decoder_hs[0]))
        decoder_layers.append(nn.ReLU())
        for i in range(len(decoder_hs) - 1):
            if decoder_hs[i] == decoder_hs[i+1]:
                decoder_layers.append(ResidualBlock(decoder_hs[i]))
            else:
                decoder_layers.append(nn.Linear(decoder_hs[i], decoder_hs[i+1]))
                decoder_layers.append(nn.BatchNorm1d(decoder_hs[i+1]))
                decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(decoder_hs[-1], self.input_size))
        self.decoder_base = nn.Sequential(*decoder_layers)

        # Predictors
        self.params_predictor = ICNN(z_dim, [512]*2)
        self.FLOPs_predictor = ICNN(z_dim, [512]*2)
        
    def encode(self, v):
        v = self.encoder_base(v)
        mu = self.fc_mu(v)
        logvar = None
        if self.ae_type == 'VAE':
            logvar = self.fc_logvar(v)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.ae_type == 'VAE':
            if self.training:
                std = torch.exp(0.5*logvar)
                eps = torch.randn_like(std)
                return mu + eps*std
            else:
                return mu
        else:
            return mu

    def decode(self, z):
        batch_size = z.shape[0]
        h = self.decoder_base(z)
        v_X, v_A = torch.split(h, [self.nX, self.nA], dim=1)
        v_A = (nn.Sigmoid()(v_A) > 0.5)*1.0
        v_X_parts = torch.split(v_X.view(batch_size, self.n_nodes, -1), self.feature_widths, dim=2)
        v_X_processed = []
        for part in v_X_parts:
            part_softmax = nn.functional.softmax(part, dim=2)
            _, indices = torch.max(part_softmax, dim=2, keepdim=True)
            part_one_hot = torch.zeros_like(part_softmax)
            part_one_hot.scatter_(2, indices, 1.0)
            v_X_processed.append(part_one_hot)

        v_X = torch.cat(v_X_processed, dim=2)

        return torch.cat([v_X.view(batch_size, -1), v_A], dim=1)

    def forward(self, v):
        mu, logvar = self.encode(v)
        z = self.reparameterize(mu, logvar)
        v_hat = self.decode(z)

        return v_hat
    
    def loss(self, v, y):
        batch_size = v.shape[0]

        # Encode and decode input
        mu, logvar = self.encode(v)
        z = self.reparameterize(mu, logvar)
        h = self.decoder_base(z)

        # Split node and edge parts
        v_X_hat, v_A_hat = torch.split(h, [self.nX, self.nA], dim=1)
        v_X, v_A = torch.split(v, [self.nX, self.nA], dim=1)

        # Edge loss
        v_A_hat = nn.Sigmoid()(v_A_hat)
        edge_loss = nn.BCELoss()(v_A_hat, v_A)

        # Edge acc
        v_A_hat_binary = (v_A_hat > 0.5).float()
        adj_correct = torch.all(v_A_hat_binary == v_A, dim=1)
        edge_acc = adj_correct.float().mean()
        
        # Split node part per feature
        v_X_hat_parts = torch.split(v_X_hat.view(batch_size, self.n_nodes, -1), self.feature_widths, dim=2)
        v_X_parts = torch.split(v_X.view(batch_size, self.n_nodes, -1), self.feature_widths, dim=2)
        
        # Node loss
        node_loss = 0
        all_features_correct = torch.ones(batch_size, dtype=torch.bool, device=v.device)
        for part, target_part in zip(v_X_hat_parts, v_X_parts):
            target_indices = torch.argmax(target_part, dim=2)
            node_loss += nn.CrossEntropyLoss()(part.reshape(-1, part.size(2)), target_indices.reshape(-1))
            
            predicted_indices = torch.argmax(part, dim=2)
            nodes_correct_for_feature = torch.all(predicted_indices == target_indices, dim=1)
            all_features_correct &= nodes_correct_for_feature
        
        # Node acc
        node_acc = all_features_correct.float().mean()
        
        # Recon loss
        recon_loss = node_loss + len(self.feature_widths)*edge_loss # Multypling by the number of node features empirically works better

        # Recon acc
        perfect_reconstructions = adj_correct & all_features_correct
        recon_acc = perfect_reconstructions.float().mean()

        # Predictor-related losses
        pred_params = self.params_predictor(z)
        pred_FLOPs = self.FLOPs_predictor(z)
        params_loss = nn.MSELoss()(pred_params, torch.log(y[:, 0]))
        FLOPs_loss = nn.MSELoss()(pred_FLOPs, torch.log(y[:, 1]))
        pred_loss = params_loss + FLOPs_loss

        # Regularization
        if self.ae_type == 'VAE':
            reg = (0.5 * (mu**2 + logvar.exp() - 1 - logvar)).sum(1).mean()
            reg_name = 'KL'
        elif self.ae_type == 'WAE':
            reg = SMMD_RBF_gaussian_prior(z)
            reg_name = 'MMD'
        elif self.ae_type == 'AE':
            reg = torch.maximum(torch.abs(z)-3.0, torch.zeros_like(z)).pow(2).sum(1).mean()
            reg += z.mean(0).pow(2).sum()
            reg_name = 'L2'

        # Total loss
        loss = recon_loss
        if self.beta > 0:
            loss += self.beta*reg
        if self.gamma > 0:
            loss += self.gamma*pred_loss

        return {
            'loss': loss, 
            'recon': recon_loss, 
            reg_name: reg, 
            'pred': pred_loss,
            'recon_acc': recon_acc,
            'node_loss': node_loss,
            'edge_loss': edge_loss,
            'node_acc': node_acc,
            'edge_acc': edge_acc,
            'params_loss': params_loss, 
            'FLOPs_loss': FLOPs_loss
        }

    def log_latent_space(self, train_dl, writer, global_step, device, num_samples=10_000):
        self.eval()  # Set model to evaluation mode
        all_z = []
        
        # Create a dataloader with a fixed batch size to collect samples
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
        
        # Combine all batches
        all_z = torch.cat(all_z, dim=0)[:num_samples]  
        
        # Log histograms for each dimension
        z_dim = all_z.size(1)
        for i in range(z_dim):
            writer.add_histogram(f'latent/z_dim_{i}', all_z[:, i], global_step)
        
        self.train()  # Switch back to training mode
        return all_z


    def train_loop(self, train_dl, val_dl, iterations, lr=1e-3, log_dir="runs/arc_ae", beta=1e-2, gamma=1e-3,
        save_dir="checkpoints", save_every=50_000,
        val_every=15_000, log_latent_every=15_000):

        self.beta = beta
        self.gamma = gamma
    
        device = get_device()
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = CosineAnnealingAlphaLR(optimizer, T_max=0.9*iterations, alpha=1e-4)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_dir, f"arcae_{timestamp}")
        writer = SummaryWriter(os.path.join(log_dir, f"{timestamp}"))
        os.makedirs(save_dir, exist_ok=True)
        
        self.train()
        global_step = 0
        pbar = tqdm(range(iterations), desc="Training")
        best_val_loss = float('inf')
        while global_step < iterations:
            for v, y in train_dl:
                # Train logic                   
                v = v.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                loss_dict = self.loss(v, y)
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Log train metrics in Tensorboard
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        writer.add_scalar(f"train/{key}", value.item(), global_step)
                
                # Display train metrics with progress bar
                desc_parts = [""]
                for key in list(loss_dict.keys())[0:5]:
                    value = loss_dict[key]
                    if isinstance(value, torch.Tensor):
                        desc_parts.append(f"{key}: {value.item():.2e}")
                pbar.set_description(" ".join(desc_parts))
                
                # Log latent space distributions
                if (global_step + 1) % log_latent_every == 0:
                    tqdm.write(f"Logging latent space distributions at step {global_step+1}")
                    self.log_latent_space(train_dl, writer, global_step, device)
                
                # Validation
                if (global_step + 1) % val_every == 0:
                    # Val logic
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
                    
                    # Log val metrics in Tensorboard
                    for k in val_losses.keys():
                        val_losses[k] /= max(val_count, 1)
                        writer.add_scalar(f"val/{k}", val_losses[k], global_step)
                    
                    # Display val metrics
                    val_msg_parts = [f"Iteration {global_step+1}/{iterations} |"]
                    for k, v in val_losses.items():
                        val_msg_parts.append(f"Val {k}: {v:.2e}")
                    tqdm.write(" ".join(val_msg_parts))
                    
                    # Save model checkpoint if best so far
                    if val_losses['loss'] < best_val_loss:
                        best_val_loss = val_losses['loss']
                        best_checkpoint_path = os.path.join(save_dir, "arcae_best.pt")
                        torch.save({
                            'iteration': global_step + 1,
                            'model_state_dict': self.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': best_val_loss,
                            'all_losses': val_losses
                        }, best_checkpoint_path)
                        tqdm.write(f"New best model saved with validation loss: {best_val_loss:.4f}")
                    
                    self.train()
                
                # Save model checkpoint as scheduled
                if (global_step + 1) % save_every == 0:
                    checkpoint_path = os.path.join(save_dir, f"arcae_iter_{global_step+1}.pt")
                    torch.save({
                        'iteration': global_step + 1,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'all_losses': {k: v.item() if isinstance(v, torch.Tensor) else v 
                                    for k, v in loss_dict.items()}
                    }, checkpoint_path)
                    tqdm.write(f"Checkpoint saved to {checkpoint_path}")
                
                global_step += 1
                pbar.update(1)
                
                if global_step >= iterations:
                    break
        
        # Log latent space distributions for final model
        tqdm.write("Logging latent space distributions for final model")
        self.log_latent_space(train_dl, writer, global_step, device)
        
        # Save final model
        final_checkpoint_path = os.path.join(save_dir, "arcae_final.pt")
        torch.save({
            'iteration': iterations,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item() if 'loss' in locals() else None,
            'all_losses': {k: v.item() if isinstance(v, torch.Tensor) else v 
                        for k, v in loss_dict.items()} if 'loss_dict' in locals() else {}
        }, final_checkpoint_path)
        tqdm.write(f"Final model saved to {final_checkpoint_path}")
        
        writer.close()
        return val_losses
    

data = torch.load("exp1903/graph_data/1000000_samples_0319_1028.pt")
train_dl, val_dl = get_dataloaders(data["V"], data["Y"], train_split=0.99, batch_size=512, num_workers=0)

from search_space import *

ae_model = ArcAE(search_space = SearchSpace(), z_dim = 139, ae_type = "AE",)
ae_model.train_loop(train_dl = train_dl, val_dl = val_dl, iterations=150_000, lr=5e-4, log_dir="runs/arc_ae", save_dir="checkpoints")