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
from botorch.utils.transforms import standardize, normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, MaternKernel

from graph_utils import ArcGraph
from search_space import SearchSpace
from autoencoder import ArcAE, get_device, CosineAnnealingAlphaLR

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


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
    ):
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize

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
        pi_values = self.pi_func(X).squeeze()
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
    ):
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize

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
        pi_values = self.pi_func(X).squeeze()

        return ei_values * (pi_values ** (self.beta/self.n))
            

class LSBO_problem:
    def __init__(self, trained_ae, cost_function="accuracy", dataset="cifar10", input_shape=[3, 32], 
                 num_classes=None, custom_cost_fn=None, log_dir="runs/nas", acquisition_type="logEI"):

        self.ae = trained_ae
        self.ae.eval()  # Set autoencoder to evaluation mode
        self.cost_function = cost_function
        self.dataset = dataset
        self.input_shape = input_shape
        self.custom_cost_fn = custom_cost_fn
        self.acquisition_type = acquisition_type
        # self.device = get_device()
        self.device = "cpu"
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
        
        # Set up TensorBoard logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(log_dir, f"{cost_function}_{timestamp}"))
        
        # Create a directory for saving models
        self.save_dir = f"nas_models/{cost_function}_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.X = torch.empty(0, self.z_dim, device=self.device)
        self.Y = torch.empty(0, 1, device=self.device)

        self.maximize = (cost_function == "accuracy")
        
        if hasattr(self.ae, 'bounds') and self.ae.bounds is not None:
            self.bounds = self.ae.bounds.to(self.device)
            print(self.bounds)
        else:
            self.bounds = torch.tensor([[-5.0] * self.z_dim, [5.0] * self.z_dim], device=self.device)
        
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
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        
        return train_loader, test_loader
    
    def _train_model(self, model, max_iterations=50_000, val_interval=1_000):
        model = model.to(self.device)        
        train_loader, test_loader = self._prepare_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingAlphaLR(optimizer, T_max=max_iterations, alpha=1e-2)
        
        best_acc = 0.0
        train_iter = iter(train_loader)
        iteration = 0
        
        pbar = tqdm(total=max_iterations, desc="Training...")        
        while iteration < max_iterations:
            model.train()            
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()            
            scheduler.step()
            
            _, predicted = outputs.max(1)
            batch_acc = predicted.eq(targets).sum().item() / targets.size(0) * 100
            
            pbar.set_description(f" iter {iteration + 1} | loss {loss.item():.2e} | acc {batch_acc:.2f} | lr {optimizer.param_groups[0]['lr']:.2e}")
            pbar.update(1)

            self.writer.add_scalar('Inner_loop/train/loss', loss.item(), iteration)
            self.writer.add_scalar('Inner_loop/train/accuracy', batch_acc, iteration)
            self.writer.add_scalar('Inner_loop/train/LR', optimizer.param_groups[0]['lr'], iteration)
            
            if (iteration + 1) % val_interval == 0 or iteration == max_iterations - 1:
                val_acc, val_loss = self._validate_model(model, test_loader, criterion)
                
                self.writer.add_scalar('Inner_loop/val/loss', val_loss, iteration)
                self.writer.add_scalar('Inner_loop/val/accuracy', val_acc, iteration)
                
                if val_acc > best_acc:
                    best_acc = val_acc

            iteration += 1
        
        pbar.close()
        print(f"Training completed. Best accuracy: {best_acc:.2f}%")
        return best_acc / 100.0  # Return as a fraction
    
    def _evaluate_architecture(self, z, model_idx=None):
        # Decode the latent vector to a graph vector
        with torch.no_grad():
            graph_vector = self.ae.decode(z.unsqueeze(0))
        
        # Create ArcGraph from the graph vector
        graph = ArcGraph(
            search_space=self.search_space,
            V=graph_vector[0],
            n_nodes=self.search_space.graph_features.n_nodes[0]
        )
        
        # Compute cost based on the specified cost function
        if self.cost_function == "params":
            # Convert to blueprint without converting to PyTorch model
            blueprint = graph.to_blueprint(input_shape=self.input_shape, num_classes=self.num_classes)
            # blueprint.plot(display=True)
            n_params = blueprint.n_params
            cost = np.log(n_params)
            print(f"n_params: {n_params:,} | log(n_params): {cost:.4f}")
            
            # Log the architecture visualization
            if model_idx is not None:
                fig_path = os.path.join(self.save_dir, f"arch_{model_idx}.png")
                blueprint.plot(output_path=fig_path)
                
            return cost
            
        elif self.cost_function == "FLOPs":
            # Convert to blueprint without converting to PyTorch model
            blueprint = graph.to_blueprint(input_shape=self.input_shape, num_classes=self.num_classes)
            flops = blueprint.FLOPs
            cost = np.log(flops)
            print(f"FLOPs: {flops:,} | log(FLOPs): {cost:.4f}")
            
            # Log the architecture visualization
            if model_idx is not None:
                fig_path = os.path.join(self.save_dir, f"arch_{model_idx}.png")
                blueprint.plot(output_path=fig_path)
                
            return cost
            
        elif self.cost_function == "accuracy":
            # Convert to PyTorch model and train it
            try:
                model = graph.to_torch(input_shape=self.input_shape, num_classes=self.num_classes)
                accuracy = self._train_model(model)
                print(f"Accuracy: {accuracy*100:.2f}%")
                
                # Save the model and log the architecture visualization
                if model_idx is not None:
                    torch.save(model.state_dict(), os.path.join(self.save_dir, f"model_{model_idx}.pth"))
                    
                    blueprint = graph.to_blueprint(input_shape=self.input_shape, num_classes=self.num_classes)
                    fig_path = os.path.join(self.save_dir, f"arch_{model_idx}.png")
                    blueprint.plot(output_path=fig_path)
                
                return accuracy  # Higher is better for accuracy
            except Exception as e:
                print(f"Error training model: {e}")
                return -float('inf')
                
        elif self.custom_cost_fn is not None:
            # Use the provided custom cost function
            return self.custom_cost_fn(graph, input_shape=self.input_shape, num_classes=self.num_classes)
            
        else:
            raise ValueError(f"Unsupported cost function: {self.cost_function}")

    def _initialize_gp(self, n_initial=10):
        print(f"Initializing with {n_initial} random architecture samples...")
        
        for i in range(1, n_initial+1):
            start_time = time.time()
            # Sample random latent vector within bounds
            z = torch.randn(self.z_dim, device=self.device)
            
            # Evaluate architecture
            cost = self._evaluate_architecture(z, model_idx=i)
            
            # Convert cost to tensor
            cost_tensor = torch.tensor([[cost]], device=self.device)
            
            # Store observation
            self.X = torch.cat([self.X, z.unsqueeze(0)], dim=0).to(torch.float64)
            self.Y = torch.cat([self.Y, cost_tensor], dim=0)
            
            # Log
            elapsed_time = time.time() - start_time
            print(f"Initial sample {i}/{n_initial} | Cost={cost:.4f} | Iteration time={elapsed_time:.2f}s")
            
            # Log to TensorBoard
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
        
        # If maximizing (e.g., accuracy), negate the values for the GP (BoTorch minimizes by default)
        Y_for_gp = -self.Y if self.maximize else self.Y
            
        # Initialize GP model
        self._update_gp(self.X, Y_for_gp)

        # Log GP hyperparameters in tensorboard
        self.writer.add_scalar("GP/mean", self.gp.mean_module.constant.item(), n_initial-1)
        self.writer.add_scalar("GP/output_scale", self.gp.covar_module.outputscale.item(), n_initial-1)
        lengthscale = self.gp.covar_module.base_kernel.lengthscale
        if lengthscale.numel() > 1:
            lengthscale = lengthscale.squeeze()
        for k in range(len(lengthscale)):
            self.writer.add_scalar(f"GP/lengthscale_{k}", lengthscale[k].item(), n_initial-1)
        
    def _update_gp(self, X, Y):
        """
        Update the Gaussian Process model with new observations and fit kernel hyperparameters.
        """
        X_normalized = normalize(X, self.bounds)
        Y_standardized = standardize(Y)

        # Initialize and fit GP model
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[-1]))
        self.gp = SingleTaskGP(X_normalized, Y_standardized, covar_module=covar_module)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)
        
    def _optimize_acquisition(self, iteration=None):        
        if self.acquisition_type == "logEI":
            Y_star = -self.Y if self.maximize else self.Y
            Y_star = Y_star.min().item()
            acq_func = LogExpectedImprovement(
                model=self.gp, 
                best_f=Y_star,
                maximize=self.maximize
            )
        elif self.acquisition_type == "EI":
            Y_star = -self.Y if self.maximize else self.Y
            Y_star = Y_star.min().item()
            acq_func = ExpectedImprovement(
                model=self.gp, 
                best_f=Y_star,
                maximize=self.maximize
            )
        elif self.acquisition_type == "logPiEI":
            Y_star = -self.Y if self.maximize else self.Y
            Y_star = Y_star.min().item()
            preference_function = self.ae.params_predictor
            acq_func = LogPiExpectedImprovement(
                model = self.gp, 
                best_f = Y_star,
                pi_func = preference_function,
                beta = 10,
                n = iteration,
                maximize=self.maximize
            )
        elif self.acquisition_type == "PiEI":
            Y_star = -self.Y if self.maximize else self.Y
            Y_star = Y_star.min().item()
            preference_function = self.ae.params_predictor
            acq_func = PiExpectedImprovement(
                model = self.gp, 
                best_f = Y_star,
                pi_func = preference_function,
                beta = 10,
                n = iteration,
                maximize=self.maximize
            )
        else:
            raise ValueError(f"Unsupported acquisition function: {self.acquisition_type}")
        
        # Optimize the acquisition function
        candidate, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([
                torch.zeros(self.z_dim, device=self.device),
                torch.ones(self.z_dim, device=self.device)
            ]),
            q=1,
            num_restarts=10,
            raw_samples=10_000,
        )
        
        # Convert back to original space
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
        
        for i in range(1, iterations+1):
            start_time = time.time()
            print(f"\n--- Iteration {i}/{iterations} ---")
            
            next_z, acq_value = self._optimize_acquisition(iteration = i)
            cost = self._evaluate_architecture(next_z, model_idx=n_initial+i)
            cost_tensor = torch.tensor([[cost]], device=self.device)
            
            # Store observation
            self.X = torch.cat([self.X, next_z.unsqueeze(0)], dim=0)
            self.Y = torch.cat([self.Y, cost_tensor], dim=0)
            
            # Update best architecture
            if (self.maximize and cost > best_cost) or (not self.maximize and cost < best_cost):
                best_cost = cost
                best_z = next_z.clone()
                print(f"New best cost: {best_cost:.4f}")
                
                # Log best to TensorBoard
                self.writer.add_scalar('Outer_loop/best_cost', best_cost, n_initial + i)
            
            # Update GP with all observations
            Y_for_gp = -self.Y if self.maximize else self.Y
            self._update_gp(self.X, Y_for_gp)
            elapsed_time = time.time() - start_time
            print(f"Iteration {i}/{iterations} | Cost={cost:.4f} | Iteration time={elapsed_time:.2f}s")

            self.writer.add_scalar("GP/mean", self.gp.mean_module.constant.item(), n_initial + i)
            self.writer.add_scalar("GP/output_scale", self.gp.covar_module.outputscale.item(), n_initial + i)
            lengthscale = self.gp.covar_module.base_kernel.lengthscale
            if lengthscale.numel() > 1:
                lengthscale = lengthscale.squeeze()
            for k in range(len(lengthscale)):
                self.writer.add_scalar(f"GP/lengthscale_{k}", lengthscale[k].item(), n_initial + i)
            
            self.writer.add_scalar('Outer_loop/cost', cost, n_initial + i)
            self.writer.add_scalar('Outer_loop/acq_value', acq_value, n_initial + i)
            self.writer.add_scalar('Outer_loop/iteration_time', elapsed_time, n_initial + i)
        
        # Final evaluation of best architecture
        print("\n--- Final Evaluation ---")
        print(f"Best {'accuracy' if self.maximize else 'cost'}: {best_cost:.4f}")
        
        # Save the best model
        with open(os.path.join(self.save_dir, "best_results.txt"), "w") as f:
            f.write(f"Best {'accuracy' if self.maximize else 'cost'}: {best_cost:.4f}\n")
            f.write(f"Latent vector: {best_z.cpu().numpy()}\n")
        
        torch.save(best_z, os.path.join(self.save_dir, "best_z.pt"))
        
        self.writer.close()
        
        return best_z, best_cost

    def visualize_best_architecture(self, z=None, filename="best_architecture.png"):
        if z is None:
            # Load the best latent vector if available
            best_z_path = os.path.join(self.save_dir, "best_z.pt")
            if os.path.exists(best_z_path):
                z = torch.load(best_z_path, map_location=self.device)
                z = z.to(dtype=torch.float32) # Cast to float32 to decode
            else:
                raise ValueError("No best architecture found. Run the optimization first or provide a latent vector.")
        
        # Decode the latent vector to a graph vector
        with torch.no_grad():
            graph_vector = self.ae.decode(z.unsqueeze(0))
        
        # Create ArcGraph from the graph vector
        graph = ArcGraph(
            search_space=self.search_space,
            V=graph_vector[0],
            n_nodes=self.search_space.graph_features.n_nodes[0]
        )
        
        # Convert to blueprint
        blueprint = graph.to_blueprint(input_shape=self.input_shape, num_classes=self.num_classes)
        
        # Create visualization
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
    
    optimizer = optim.SGD([x], lr=lr)
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
    checkpoint = torch.load("checkpoints/arcae_20250325_012519/arcae_final.pt", map_location=device)
    ae.load_state_dict(checkpoint['model_state_dict'])
    ae.bounds = None
    ae.to(device)
    

    # # Descend gradient of predictor's input
    # predictor = ae.FLOPs_predictor
    # z, FLOPs = find_argmin(predictor, input_shape=(99,), lr=0.1, n_steps=2_000, device="cpu")
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
    # print(g2.FLOPs)
    # print(FLOPs)


    # Create LSBO problem
    lsbo = LSBO_problem(
        trained_ae=ae,
        cost_function="params",  # or "accuracy", "FLOPs"
        acquisition_type="PiEI",
        dataset="cifar10",
        input_shape=[3, 32],
        log_dir="runs/nas",
    )
    
    # Run optimization
    best_z, best_cost = lsbo.run(iterations=1000, n_initial=99)
    
    # Visualize the best architecture
    best_blueprint = lsbo.visualize_best_architecture()
    print(f"Best architecture has {best_blueprint.n_params:,} parameters and {best_blueprint.FLOPs:,} FLOPs")