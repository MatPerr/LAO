import torch
import numpy as np
import time
import os
import math
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import AnalyticAcquisitionFunction, LogExpectedImprovement, ExpectedImprovement
from botorch.acquisition.objective import PosteriorTransform
from botorch.optim import optimize_acqf
from botorch.models.model import Model
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel

from graph_utils import ArcGraph
from search_space import SearchSpace
from autoencoder import ArcAE, get_device, CosineAnnealingAlphaLR
from graph_dataloader import MultiEpochsDataLoader

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from torchsummary import summary


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
    

class LogPiExpectedImprovement(AnalyticAcquisitionFunction):
    """
    logPiEI(x) = log([EI*π](x)) = log(EI(x)) + log(π(x)), 
    where π is a provided weighting scheme (preference function)
    """
    _log: bool = True
    def __init__(
        self, 
        model: Model, 
        best_f: float | torch.Tensor,
        posterior_transform: PosteriorTransform | None = None,
        pi_func: torch.nn.Module = None,
        beta: float = 1.0,
        n: int = 1,
        maximize: bool = True,
        bounds: torch.Tensor = None
    ):
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize
        self.bounds = bounds

        self.beta = beta
        self.n = n
        
        self.log_ei = LogExpectedImprovement(
            model=model, 
            best_f=best_f, 
            maximize=maximize
        )
        
        self.pi_func = pi_func or (lambda x: torch.ones_like(x))

    def forward(self, X: torch.Tensor):
        log_ei_values = self.log_ei(X)
        pi_values = self.pi_func(unnormalize(X, self.bounds)).squeeze()
        log_pi_values = torch.log(torch.clamp(pi_values, min=1e-10))

        return log_ei_values + (self.beta/self.n)*log_pi_values
        

class PiExpectedImprovement(AnalyticAcquisitionFunction):
    """
    PiEI(x) = EI(x)*π(x), 
    where π is a provided weighting scheme (preference function)
    """
    _log: bool = True
    def __init__(
        self, 
        model: Model, 
        best_f: float | torch.Tensor,
        posterior_transform: PosteriorTransform | None = None,
        pi_func: torch.nn.Module = None,
        beta: float = 1.0,
        n: int = 1,
        maximize: bool = True,
        mode: str = 'mult',
        bounds: torch.Tensor = None
    ):
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize
        self.mode = mode
        self.bounds = bounds

        self.beta = beta
        self.n = n
        
        self.ei = ExpectedImprovement(
            model=model, 
            best_f=best_f, 
            maximize=maximize
        )
        
        self.pi_func = pi_func or (lambda x: torch.ones_like(x))

    def forward(self, X: torch.Tensor):
        ei_values = self.ei(X)
        pi_values = self.pi_func(unnormalize(X, self.bounds)).squeeze()

        if self.mode == 'mult':
            return ei_values * (pi_values ** (self.beta/self.n))
        elif self.mode == 'add':
            return ei_values + (self.beta/self.n)*pi_values
        elif self.mode == 'pi':
            return pi_values
        else:
            raise ValueError(f"Unsupported PiBO mode: {self.mode}")


def standardize(Y, return_moments = False):
    r"""Standardizes (zero mean, unit variance) a tensor by dim=-2.

    If the tensor is single-dimensional, simply standardizes the tensor.
    If for some batch index all elements are equal (or if there is only a single
    data point), this function will return 0 for that batch index.

    Args:
        Y: A `batch_shape x n x m`-dim tensor.

    Returns:
        The standardized `Y`.

    Example:
        >>> Y = torch.rand(4, 3)
        >>> Y_standardized = standardize(Y)
    """
    stddim = -1 if Y.dim() < 2 else -2
    Y_std = Y.std(dim=stddim, keepdim=True)
    Y_std = Y_std.where(Y_std >= 1e-9, torch.full_like(Y_std, 1.0))
    Y_mean = Y.mean(dim=stddim, keepdim=True)
    if not return_moments:
        return (Y - Y_mean) / Y_std
    else:
        return (Y - Y_mean) / Y_std, Y_mean, Y_std


def ZQL(x, x1, x2, m):
    '''
    A C1 function defined by parts:
        - f1(x) = 0 for x <= x1
        - f2(x) = ax^2 + bx + c for x1 < x <= 2 with f2(x2) = m
        - f3(x) = alpha*log(x)+beta for x2 < x
    '''
    def f1(t):
        return 0.

    a = m /((x2-x1)**2)
    b = -2*x1*a
    c = a*x1**2
    def f2(t):
        return a*t**2 + b*t + c
    
    def df2(t):
        return 2*a*t + b
    alpha = x2*df2(x2)
    beta = m - alpha*np.log(x2)
    def f3(t):
        return alpha*np.log(t)+beta

    return np.less_equal(x,x1)*f1(x) + np.greater(x, x1)*np.less_equal(x, x2)*f2(x) + np.greater(x, x2)*f3(x)


class LSBO_problem:
    def __init__(self, trained_ae, cost_function="accuracy", dataset="cifar10", input_shape=[3, 32], 
                 num_classes=None, custom_cost_fn=None, log_dir="runs/nas", acquisition_type="logEI", pi_func = None):

        self.ae = trained_ae
        self.ae.eval()
        self.cost_function = cost_function
        self.dataset = dataset
        self.input_shape = input_shape
        self.custom_cost_fn = custom_cost_fn
        self.acquisition_type = acquisition_type
        self.pi_func = pi_func
        self.train_device = get_device()
        self.BO_device = "cpu"
        self.z_dim = self.ae.z_dim
        
        if num_classes is None:
            if dataset == "cifar10":
                self.num_classes = 10
            elif dataset == "cifar100":
                self.num_classes = 100
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
        else:
            self.num_classes = num_classes
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(log_dir, f"{cost_function}_{timestamp}"))
        self.save_dir = f"nas_models/{cost_function}_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.X = torch.empty(0, self.z_dim, device=self.BO_device)
        self.Y = torch.empty(0, 1, device=self.BO_device)
        self.Yvar = torch.empty(0, 1, device=self.BO_device)

        self.maximize = (cost_function == "accuracy")
        
        if hasattr(self.ae, 'bounds') and self.ae.bounds is not None:
            self.bounds = self.ae.bounds.to(self.BO_device)
            print(self.bounds)
        else:
            self.bounds = torch.tensor([[-5.0] * self.z_dim, [5.0] * self.z_dim], device=self.BO_device)
        
        self.search_space = self.ae.search_space
            
    def _prepare_dataset(self):
        if self.dataset == "cifar10" or self.dataset == "cifar100":
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        if self.dataset == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=test_transform)
        elif self.dataset == "cifar100":
            train_dataset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=test_transform)
        
        train_loader = MultiEpochsDataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = MultiEpochsDataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        
        return train_loader, test_loader
    
    def _train_and_eval_model(self, model, max_iterations=20_000, model_idx=None):
        model = model.to(self.train_device)
        train_loader, test_loader = self._prepare_dataset()

        tag_prefix = f"model_{model_idx}" if model_idx is not None else "unknown_model"

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingAlphaLR(optimizer, T_max=max_iterations, alpha=1e-4)
        
        best_acc = 0.0
        iteration = 0
        validation_frequency = max_iterations // 20
        
        pbar = tqdm(total=max_iterations, desc="Training...")        
        while iteration < max_iterations:
            for x, y in train_loader:
                model.train()            
                x, y = x.to(self.train_device), y.to(self.train_device)
                
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()            
                scheduler.step()
                
                _, predicted = outputs.max(1)
                acc = predicted.eq(y).sum().item() / y.size(0)

                pbar.set_description(f" n_iter {iteration + 1} | loss {loss.item():.2e} | acc {100*acc:.2f} | lr {optimizer.param_groups[0]['lr']:.2e}")
                pbar.update(1)

                # Log with hierarchical tags
                self.writer.add_scalar(f'Inner_loop/{tag_prefix}/train/loss', loss.item(), iteration)
                self.writer.add_scalar(f'Inner_loop/{tag_prefix}/train/accuracy', acc, iteration)
                self.writer.add_scalar(f'Inner_loop/{tag_prefix}/train/LR', optimizer.param_groups[0]['lr'], iteration)
                
                if iteration % validation_frequency == 0 or iteration == max_iterations - 1:
                    val_acc, avg_val_loss = self._validate_model(model, test_loader, criterion)
                    
                    self.writer.add_scalar(f'Inner_loop/{tag_prefix}/val/loss', avg_val_loss, iteration)
                    self.writer.add_scalar(f'Inner_loop/{tag_prefix}/val/accuracy', val_acc, iteration)
                    
                    if val_acc > best_acc:
                        best_acc = val_acc
                
                iteration += 1
                if iteration >= max_iterations:
                    break
        
        pbar.close()
        print(f"Training completed. Best val accuracy: {best_acc:.2f}%")
            
        return best_acc

    def _validate_model(self, model, test_loader, criterion):
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for val_x, val_y in test_loader:
                val_x, val_y = val_x.to(self.train_device), val_y.to(self.train_device)
                val_outputs = model(val_x)
                val_loss += criterion(val_outputs, val_y).item()
                
                _, val_predicted = val_outputs.max(1)
                total += val_y.size(0)
                correct += val_predicted.eq(val_y).sum().item()
        
        val_acc = correct / total
        avg_val_loss = val_loss / len(test_loader)
        
        return val_acc, avg_val_loss
    
    def _evaluate_architecture(self, z, model_idx=None):
        with torch.no_grad():
            graph_vector = self.ae.decode(z.unsqueeze(0))
        graph = ArcGraph(
            search_space=self.search_space,
            V=graph_vector[0],
            n_nodes=self.search_space.graph_features.n_nodes[0]
        )      
        if self.cost_function == "params":
            # Convert to blueprint without converting to PyTorch model to save time
            blueprint = graph.to_blueprint(input_shape=self.input_shape, num_classes=self.num_classes)
            n_params = blueprint.n_params
            cost = np.log(n_params)
            print(f"n_params: {n_params:,} | log(n_params): {cost:.4f}")

            if model_idx is not None:
                fig_path = os.path.join(self.save_dir, f"arch_{model_idx}.png")
                blueprint.plot(output_path=fig_path)
                
            return cost, None
        
        if self.cost_function == "pred_params":
            cost = np.float64(self.ae.params_predictor(z.unsqueeze(0)).item())
            print(f"pred_n_params: {np.exp(cost):,} | pred_log(n_params): {cost:.4f}")

            blueprint = graph.to_blueprint(input_shape=self.input_shape, num_classes=self.num_classes)
            if model_idx is not None:
                fig_path = os.path.join(self.save_dir, f"arch_{model_idx}.png")
                blueprint.plot(output_path=fig_path)

            return cost, None
            
        elif self.cost_function == "FLOPs":
            blueprint = graph.to_blueprint(input_shape=self.input_shape, num_classes=self.num_classes)
            flops = blueprint.FLOPs
            cost = np.log(flops)
            print(f"FLOPs: {flops:,} | log(FLOPs): {cost:.4f}")
            
            if model_idx is not None:
                fig_path = os.path.join(self.save_dir, f"arch_{model_idx}.png")
                blueprint.plot(output_path=fig_path)
                
            return cost, None
        
        if self.cost_function == "params_x_FLOPs":
            blueprint = graph.to_blueprint(input_shape=self.input_shape, num_classes=self.num_classes)
            n_params = blueprint.n_params
            FLOPs = blueprint.FLOPs
            c_params = np.log(n_params)
            c_FLOPs = np.log(FLOPs)
            cost = c_params * c_FLOPs
            print(f"n_params: {n_params:,} | FLOPs {FLOPs:,} | cost: {cost:.4f} | c_n_params: {c_params:.4f} | c_FLOPs: {c_FLOPs:.4f}")
            
            if model_idx is not None:
                fig_path = os.path.join(self.save_dir, f"arch_{model_idx}.png")
                blueprint.plot(output_path=fig_path)
                
            return cost, None
            
        elif self.cost_function == "accuracy":
            try:
                model = graph.to_torch(input_shape=self.input_shape, num_classes=self.num_classes)
                print(f"n_params : {model.n_params} | FLOPs : {model.FLOPs} | BBGP : {model.BBGP}")
                accuracy = np.float64(self._train_and_eval_model(model))
                print(f"Accuracy: {accuracy*100:.2f}%")
                
                if model_idx is not None:
                    torch.save(model.state_dict(), os.path.join(self.save_dir, f"model_{model_idx}.pth"))
                    blueprint = graph.to_blueprint(input_shape=self.input_shape, num_classes=self.num_classes)
                    fig_path = os.path.join(self.save_dir, f"arch_{model_idx}.png")
                    blueprint.plot(output_path=fig_path)
                
                return accuracy, None
            
            except Exception as e:
                print(f"Error training model: {e}")
                return -float('inf')
        
        elif self.cost_function == "canonical":
            model = graph.to_torch(input_shape=self.input_shape, num_classes=self.num_classes)
            n_params = model.n_params
            FLOPs = model.FLOPs
            BBGP = model.BBGP
            print(f"n_params : {n_params} | FLOPs : {FLOPs} | BBGP : {BBGP}")
            if n_params > 1e6 or FLOPs > 1e8 or BBGP < 2:
                # Assume accuracy centered in 0.7
                accuracy = np.float64(0.7)
                # Assume 3*std = 0.21, hence std = 0.07
                acc_var = np.float64(0.07**2)
            else:
                accuracy = np.float64(self._train_and_eval_model(model))
                # Assume 3*std = 0.009, hence std = 0.003
                acc_var = np.float64(0.003**2)

            cost = - accuracy + ZQL(n_params, 200_000, 250_000, 0.1) + ZQL(FLOPs, 60_000_000, 80_000_000, 0.1)
            
            if model_idx is not None:
                torch.save(model.state_dict(), os.path.join(self.save_dir, f"model_{model_idx}.pth"))
                blueprint = graph.to_blueprint(input_shape=self.input_shape, num_classes=self.num_classes)
                fig_path = os.path.join(self.save_dir, f"arch_{model_idx}.png")
                blueprint.plot(output_path=fig_path)
            
            self.writer.add_scalar('Outer_loop/n_params', n_params, model_idx)
            self.writer.add_scalar('Outer_loop/FLOPs', FLOPs, model_idx)
            self.writer.add_scalar('Outer_loop/BBGP', BBGP, model_idx)
            # Only source of variance considered is accuracy
            return cost, acc_var
                
        elif self.custom_cost_fn is not None:
            return self.custom_cost_fn(graph, input_shape=self.input_shape, num_classes=self.num_classes)
            
        else:
            raise ValueError(f"Unsupported cost function: {self.cost_function}")

    def _initialize_gp(self, n_initial=10):
        print(f"Initializing with {n_initial} random architecture samples...")
        
        for i in range(1, n_initial+1):
            start_time = time.time()
            z = torch.randn(self.z_dim, device=self.BO_device)

            cost, cost_var = self._evaluate_architecture(z, model_idx=i)
            cost_tensor = torch.tensor([[cost]], device=self.BO_device)
            
            self.X = torch.cat([self.X, z.unsqueeze(0)], dim=0).to(torch.float64)
            self.Y = torch.cat([self.Y, cost_tensor], dim=0)
            if cost_var is not None:
                cost_var_tensor = torch.tensor([[cost_var]], device=self.BO_device)
                self.Yvar = torch.cat([self.Yvar, cost_var_tensor], dim = 0)
            else:
                self.Yvar = None
            
            elapsed_time = time.time() - start_time
            print(f"Initial sample {i}/{n_initial} | Cost={cost:.4f} | Iteration time={elapsed_time:.2f}s")
            
            self.writer.add_scalar('Outer_loop/cost', cost, i)
            self.writer.add_scalar('Outer_loop/iteration_time', elapsed_time, i)
            if i == 1:
                best_cost = cost
                self.writer.add_scalar('Outer_loop/best_cost', best_cost, i)
            
            elif i > 1:
                if (self.maximize and cost > best_cost) or (not self.maximize and cost < best_cost):
                    best_cost = cost
                    print(f"New best cost: {best_cost:.4f}")
                    self.writer.add_scalar('Outer_loop/best_cost', best_cost, i)
            
        self._update_gp(self.X, self.Y, self.Yvar)

        self.writer.add_scalar("GP/mean", self.gp.mean_module.constant.item(), n_initial-1)
        self.writer.add_scalar("GP/output_scale", self.gp.covar_module.outputscale.item(), n_initial-1)
        if self.Yvar is None:
            self.writer.add_scalar("GP/noise", self.gp.likelihood.noise.item(), n_initial-1)
        else:
            #TODO: only store last one ?
            self.writer.add_scalar("GP/noise", cost_var, n_initial-1)
        lengthscale = self.gp.covar_module.base_kernel.lengthscale
        if lengthscale.numel() > 1:
            lengthscale = lengthscale.squeeze()
        for k in range(len(lengthscale)):
            self.writer.add_scalar(f"GP/lengthscale_{k}", lengthscale[k].item(), n_initial-1)
        
    def _update_gp(self, X, Y, Yvar = None):
        X_normalized = normalize(X, self.bounds)
        Y_standardized, _, Y_std = standardize(Y, return_moments=True)
        if Yvar is not None:
            train_Yvar = Yvar / (Y_std**2)
        else:
            train_Yvar = None

        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[-1]))

        self.gp = SingleTaskGP(train_X = X_normalized,
                               train_Y = Y_standardized,
                               train_Yvar = train_Yvar,
                               covar_module = covar_module)

        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)
        
    def _optimize_acquisition(self, iteration=None):        
        if self.acquisition_type == "logEI":
            Y_star = self.Y.max().item() if self.maximize else self.Y.min().item()
            acq_func = LogExpectedImprovement(
                model=self.gp, 
                best_f=Y_star,
                maximize=self.maximize
            )
        elif self.acquisition_type == "EI":
            Y_star = self.Y.max().item() if self.maximize else self.Y.min().item()
            acq_func = ExpectedImprovement(
                model=self.gp, 
                best_f=Y_star,
                maximize=self.maximize
            )
        elif self.acquisition_type == "logPiEI":
            assert self.pi_func is not None, "Preference function (pi_func) must be provided for PiBO"
            Y_star = self.Y.max().item() if self.maximize else self.Y.min().item()
            acq_func = LogPiExpectedImprovement(
                model = self.gp, 
                best_f = Y_star,
                pi_func = self.pi_func,
                beta = 1,
                n = iteration,
                maximize=self.maximize,
                bounds = self.bounds
            )
        elif self.acquisition_type == "PiEI":
            assert self.pi_func is not None, "Preference function (pi_func) must be provided for PiBO"
            Y_star = self.Y.max().item() if self.maximize else self.Y.min().item()
            acq_func = PiExpectedImprovement(
                model = self.gp,
                best_f = Y_star,
                pi_func = self.pi_func,
                beta = 1,
                n = iteration,
                maximize=self.maximize,
                mode = 'mult',
                bounds = self.bounds
            )
        else:
            raise ValueError(f"Unsupported acquisition function: {self.acquisition_type}")
        
        candidate, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([
                torch.zeros(self.z_dim, device=self.BO_device),
                torch.ones(self.z_dim, device=self.BO_device)
            ]),
            q=1,
            num_restarts=10,
            raw_samples=10_000,
        )
        next_z = unnormalize(candidate.squeeze(0), self.bounds)
    
        return next_z, acq_value.item()
        
    def run(self, iterations=100, n_initial=10):
        self._initialize_gp(n_initial)
        
        if self.maximize:
            best_idx = torch.argmax(self.Y)
        else:
            best_idx = torch.argmin(self.Y)
            
        best_cost = self.Y[best_idx].item()
        best_z = self.X[best_idx].clone()
        
        print(f"Starting optimization from best initial cost: {best_cost:.4f}")
        
        # Create a directory to store all z vectors
        z_save_dir = os.path.join(self.save_dir, "z_vectors")
        os.makedirs(z_save_dir, exist_ok=True)
        
        # Save all initial z vectors
        for i in range(n_initial):
            torch.save(self.X[i], os.path.join(z_save_dir, f"z_{i+1}.pt"))
        
        for i in range(1, iterations+1):
            start_time = time.time()
            print(f"\n--- Iteration {i}/{iterations} ---")
            
            next_z, acq_value = self._optimize_acquisition(iteration=i)
            cost, cost_var = self._evaluate_architecture(next_z, model_idx=n_initial+i)
            cost_tensor = torch.tensor([[cost]], device=self.BO_device)
            
            self.X = torch.cat([self.X, next_z.unsqueeze(0)], dim=0)
            self.Y = torch.cat([self.Y, cost_tensor], dim=0)
            
            # Save the current z vector
            torch.save(next_z, os.path.join(z_save_dir, f"z_{n_initial+i}.pt"))
            
            if cost_var is not None:
                cost_var_tensor = torch.tensor([[cost_var]], device=self.BO_device)
                self.Yvar = torch.cat([self.Yvar, cost_var_tensor], dim=0)
            else:
                self.Yvar = None
            
            if (self.maximize and cost > best_cost) or (not self.maximize and cost < best_cost):
                best_cost = cost
                best_z = next_z.clone()
                print(f"New best cost: {best_cost:.4f}")
                self.writer.add_scalar('Outer_loop/best_cost', best_cost, n_initial + i)
                
                # Save the current best z with special name
                torch.save(best_z, os.path.join(z_save_dir, f"best_z_at_iteration_{n_initial+i}.pt"))
            
            self._update_gp(self.X, self.Y, self.Yvar)
            elapsed_time = time.time() - start_time
            print(f"Iteration {i}/{iterations} | Cost={cost:.4f} | Iteration time={elapsed_time:.2f}s")

            self.writer.add_scalar("GP/mean", self.gp.mean_module.constant.item(), n_initial + i)
            self.writer.add_scalar("GP/output_scale", self.gp.covar_module.outputscale.item(), n_initial + i)
            if self.Yvar is None:
                self.writer.add_scalar("GP/noise", self.gp.likelihood.noise.item(), n_initial + i)
            else:
                self.writer.add_scalar("GP/noise", cost_var, n_initial + i)
            lengthscale = self.gp.covar_module.base_kernel.lengthscale
            if lengthscale.numel() > 1:
                lengthscale = lengthscale.squeeze()
            for k in range(len(lengthscale)):
                self.writer.add_scalar(f"GP/lengthscale_{k}", lengthscale[k].item(), n_initial + i)
            
            self.writer.add_scalar('Outer_loop/cost', cost, n_initial + i)
            self.writer.add_scalar('Outer_loop/acq_value', acq_value, n_initial + i)
            self.writer.add_scalar('Outer_loop/iteration_time', elapsed_time, n_initial + i)
        
        print("\n--- Final Evaluation ---")
        print(f"Best {'objective' if self.maximize else 'cost'}: {best_cost:.4f}")
        
        with open(os.path.join(self.save_dir, "best_results.txt"), "w") as f:
            f.write(f"Best {'objective' if self.maximize else 'cost'}: {best_cost:.4f}\n")
            f.write(f"Latent vector: {best_z.cpu().numpy()}\n")
        
        # Save the final best z to the root directory as well
        torch.save(best_z, os.path.join(self.save_dir, "best_z.pt"))
        
        # Also save a mapping of iteration number -> cost
        cost_mapping = {i: self.Y[i].item() for i in range(len(self.Y))}
        torch.save(cost_mapping, os.path.join(self.save_dir, "cost_mapping.pt"))
        
        self.writer.close()
        
        return best_z, best_cost

    def visualize_best_architecture(self, z=None, filename="best_architecture.png"):
        if z is None:
            best_z_path = os.path.join(self.save_dir, "best_z.pt")
            if os.path.exists(best_z_path):
                z = torch.load(best_z_path, map_location=self.BO_device)
                z = z.to(dtype=torch.float32) # Cast to float32 to decode
            else:
                raise ValueError("No best architecture found. Run the optimization first or provide a latent vector.")
        
        with torch.no_grad():
            graph_vector = self.ae.decode(z.unsqueeze(0))
        
        graph = ArcGraph(
            search_space=self.search_space,
            V=graph_vector[0],
            n_nodes=self.search_space.graph_features.n_nodes[0]
        )
        
        blueprint = graph.to_blueprint(input_shape=self.input_shape, num_classes=self.num_classes)
        
        output_path = os.path.join(self.save_dir, filename)
        blueprint.plot(output_path=output_path)
        
        print(f"Architecture visualization saved to {output_path}")
        return blueprint
    
    def evaluate_specific_architecture(self, z):
        return self._evaluate_architecture(z)


def find_argmin(model, input_shape=(2,), lr=0.01, n_steps=2_000, device="cpu"):
    model.to(device)
    model.eval()
    x = torch.randn(input_shape, requires_grad=True, device=device)
    
    optimizer = optim.Adam([x], lr=lr)
    scheduler = CosineAnnealingAlphaLR(optimizer, T_max=n_steps, alpha=1e-2)

    trajectory = [x.detach().clone()]
    losses = []
    
    pbar = tqdm(range(n_steps), desc="Optimizing Input")
    for step in pbar:
        optimizer.zero_grad()
        
        output = model(x)      
        loss = output
        loss.backward()       
        optimizer.step()
        scheduler.step()
        
        trajectory.append(x.detach().clone())
        losses.append(loss.item())
        pbar.set_description(f"Iteration: {step} | Loss: {loss.item():.2e}")
            
    
    return x.detach(), loss


if __name__ == "__main__":
    # device = get_device()
    device = "cpu"
    search_space = SearchSpace()
    
    # Create and load the autoencoder
    ae = ArcAE(search_space=search_space, z_dim=99, ae_type="WAE")
    # All ICNN
    # checkpoint = torch.load("checkpoints/arcae_20250402_231330/arcae_final.pt", map_location=device)
    
    # All NO ICNN
    # checkpoint = torch.load("checkpoints/arcae_20250403_092511/arcae_final.pt", map_location=device)
    checkpoint = torch.load("checkpoints/arcae_20250405_103101/arcae_final.pt", map_location=device)

    ae.load_state_dict(checkpoint['model_state_dict'])
    ae.bounds = None
    ae.to(device)
    
    # # Descend gradient of predictor's input
    # predictor = ae.params_predictor
    # z, n_params = find_argmin(predictor, input_shape=(99,), lr=0.01, n_steps=1_000, device="cpu")
    # with torch.no_grad():
    #     ae.eval()
    #     v = ae.decode(z.unsqueeze(0))
    # # Create ArcGraph from the graph vector
    # g = ArcGraph(
    #     search_space=search_space,
    #     V=v[0],
    #     n_nodes=search_space.graph_features.n_nodes[0]
    #     )
    # g2 = g.to_blueprint(input_shape=[3, 32], num_classes=10)
    # g2.plot(display=True)
    # print(g2.n_params)
    # print(n_params)


    # Create preference function
    params_pi_func = nn.Sequential(
        ae.params_predictor,
        Lambda(lambda x: torch.maximum(torch.log(torch.Tensor([200_000])), x)),
        Lambda(lambda x: torch.clamp(x, min=0, max=30)),
        Lambda(lambda x: x / 30),
        Lambda(lambda x: (x-0.5)*8),
        Lambda(lambda x: torch.sigmoid(-x))
    )

    FLOPs_pi_func = nn.Sequential(
        ae.FLOPs_predictor,
        Lambda(lambda x: torch.maximum(torch.log(torch.Tensor([60_000_000])), x)),
        Lambda(lambda x: torch.clamp(x, min=0, max=60)),
        Lambda(lambda x: x / 60),
        Lambda(lambda x: (x-0.5)*8),
        Lambda(lambda x: torch.sigmoid(-x))
    )

    # target_BBGP_pi_func = nn.Sequential(
    #     ae.BBGP_predictor,
    #     Lambda(lambda x: torch.clamp(x, min=0, max=12)),
    #     Lambda(lambda x: x-2),
    #     Lambda(lambda x: torch.abs(x)),
    #     Lambda(lambda x: x / 3),
    #     Lambda(lambda x: (x-0.5)*8),
    #     Lambda(lambda x: torch.sigmoid(-x))
    # )
    
    class prod_pi_func(nn.Module):
        def __init__(self, pi_funcs):
            super().__init__()
            self.pi_funcs = pi_funcs
        
        def forward(self, x):
            result = self.pi_funcs[0](x)           
            for pi_func in self.pi_funcs[1:]:
                result = result * pi_func(x)
            
            return result


    # Create LSBO problem
    lsbo = LSBO_problem(
        trained_ae=ae,
        cost_function="canonical",  # or "accuracy","canonical", "FLOPs", "pred_params" or "params_x_FLOPs_x_BBGP" or "params_x_FLOPs"
        acquisition_type="logPiEI",
        dataset="cifar10",
        input_shape=[3, 32],
        log_dir="runs/nas",
        pi_func = prod_pi_func([params_pi_func, FLOPs_pi_func])
    )
    
    # Run optimization
    best_z, best_cost = lsbo.run(iterations=1000, n_initial=99)
    
    # Visualize the best architecture
    best_blueprint = lsbo.visualize_best_architecture()
    print(f"Best architecture has {best_blueprint.n_params:,} parameters and {best_blueprint.FLOPs:,} FLOPs")