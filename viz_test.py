import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from graph_utils import ArcGraph
from search_space import SearchSpace
from autoencoder import ArcAE, get_device
from nas import Lambda

def sample_and_visualize_latent_vectors(
    trained_ae, 
    n_samples=1000, 
    input_shape=[3, 32], 
    num_classes=10, 
    save_dir="latent_analysis", 
    fig_size=(18, 14),
    device=None
):
    """
    Sample n latent vectors and display histograms of their metrics.
    
    Args:
        trained_ae: Trained autoencoder model
        n_samples: Number of latent vectors to sample
        input_shape: Input shape for the architectures
        num_classes: Number of classes for the architectures
        save_dir: Directory to save the visualization results
        fig_size: Size of the figure for plotting
        device: Device to run on (if None, will use get_device())
    
    Returns:
        Dictionary containing the sampled metrics
    """
    if device is None:
        device = get_device()
    
    trained_ae.eval()
    trained_ae.to(device)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"latent_analysis_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize preference functions (same as in the original code)
    params_pi_func = torch.nn.Sequential(
        trained_ae.params_predictor,
        Lambda(lambda x: torch.clamp(x, min=0, max=30)),
        Lambda(lambda x: x / 26),
        Lambda(lambda x: (x-0.5)*8),
        Lambda(lambda x: torch.sigmoid(-x))
    )

    FLOPs_pi_func = torch.nn.Sequential(
        trained_ae.FLOPs_predictor,
        Lambda(lambda x: torch.clamp(x, min=0, max=60)),
        Lambda(lambda x: x / 60),
        Lambda(lambda x: (x-0.5)*8),
        Lambda(lambda x: torch.sigmoid(-x))
    )

    target_BBGP_pi_func = torch.nn.Sequential(
        trained_ae.BBGP_predictor,
        Lambda(lambda x: torch.clamp(x, min=0, max=12)),
        Lambda(lambda x: x-2),
        Lambda(lambda x: torch.abs(x)),
        Lambda(lambda x: x / 3),
        Lambda(lambda x: (x-0.5)*8),
        Lambda(lambda x: torch.sigmoid(-x))
    )
    
    class prod_pi_func(torch.nn.Module):
        def __init__(self, pi_funcs):
            super().__init__()
            self.pi_funcs = pi_funcs
        
        def forward(self, x):
            result = self.pi_funcs[0](x)           
            for pi_func in self.pi_funcs[1:]:
                result = result * pi_func(x)
            
            return result
    
    # Create product pi function
    prod_func = prod_pi_func([params_pi_func, FLOPs_pi_func, target_BBGP_pi_func])
    
    # Prepare data collection
    metrics = {
        "z": [],
        "log_params": [],
        "log_FLOPs": [],
        "log2_BBGP": [],
        "pred_log_params": [],
        "pred_log_FLOPs": [],
        "pred_log2_BBGP": [],
        "params_pi": [],
        "FLOPs_pi": [],
        "BBGP_pi": [],
        "prod_pi": [],
        "params": [],
        "FLOPs": [],
        "BBGP": []
    }
    
    # Sample latent vectors and collect metrics
    print(f"Sampling {n_samples} architectures from the latent space...")
    for i in tqdm(range(n_samples)):
        # Sample latent vector
        z = torch.randn(trained_ae.z_dim, device=device)
        
        # Store the latent vector
        metrics["z"].append(z.cpu().numpy())
        
        # Get predictions from autoencoder's predictors
        with torch.no_grad():
            pred_params = trained_ae.params_predictor(z.unsqueeze(0)).item()
            pred_FLOPs = trained_ae.FLOPs_predictor(z.unsqueeze(0)).item()
            pred_BBGP = trained_ae.BBGP_predictor(z.unsqueeze(0)).item()
            
            params_pi_value = params_pi_func(z.unsqueeze(0)).item()
            FLOPs_pi_value = FLOPs_pi_func(z.unsqueeze(0)).item()
            BBGP_pi_value = target_BBGP_pi_func(z.unsqueeze(0)).item()
            prod_pi_value = prod_func(z.unsqueeze(0)).item()
            
            # Decode the latent vector
            graph_vector = trained_ae.decode(z.unsqueeze(0))
        
        # Create architecture graph
        graph = ArcGraph(
            search_space=trained_ae.search_space,
            V=graph_vector[0],
            n_nodes=trained_ae.search_space.graph_features.n_nodes[0]
        )
        
        # Convert to blueprint to get actual metrics
        try:
            blueprint = graph.to_blueprint(input_shape=input_shape, num_classes=num_classes)
            actual_params = blueprint.n_params
            actual_FLOPs = blueprint.FLOPs
            actual_BBGP = blueprint.BBGP
            
            # Store metrics
            metrics["log_params"].append(np.log(actual_params))
            metrics["log_FLOPs"].append(np.log(actual_FLOPs))
            metrics["log2_BBGP"].append(np.log2(actual_BBGP))
            
            metrics["pred_log_params"].append(pred_params)
            metrics["pred_log_FLOPs"].append(pred_FLOPs)
            metrics["pred_log2_BBGP"].append(pred_BBGP)
            
            metrics["params_pi"].append(params_pi_value)
            metrics["FLOPs_pi"].append(FLOPs_pi_value)
            metrics["BBGP_pi"].append(BBGP_pi_value)
            metrics["prod_pi"].append(prod_pi_value)
            
            metrics["params"].append(actual_params)
            metrics["FLOPs"].append(actual_FLOPs)
            metrics["BBGP"].append(actual_BBGP)
            
        except Exception as e:
            print(f"Error creating blueprint for sample {i}: {e}")
            # Skip this sample
            continue
            
    # Convert lists to numpy arrays
    for key in metrics.keys():
        if key != "z":
            metrics[key] = np.array(metrics[key])
    metrics["z"] = np.array(metrics["z"])
    
    # Create visualization
    print("Creating visualizations...")
    fig, axs = plt.subplots(4, 3, figsize=fig_size)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Plot actual metrics
    axs[0, 0].hist(metrics["log_params"], bins=50, alpha=0.7)
    axs[0, 0].set_title(f"Log Parameters Distribution\nMean: {np.mean(metrics['log_params']):.2f}")
    axs[0, 0].set_xlabel("log(params)")
    axs[0, 0].set_ylabel("Count")
    
    axs[0, 1].hist(metrics["log_FLOPs"], bins=50, alpha=0.7)
    axs[0, 1].set_title(f"Log FLOPs Distribution\nMean: {np.mean(metrics['log_FLOPs']):.2f}")
    axs[0, 1].set_xlabel("log(FLOPs)")
    axs[0, 1].set_ylabel("Count")
    
    axs[0, 2].hist(metrics["log2_BBGP"], bins=50, alpha=0.7)
    axs[0, 2].set_title(f"Log2 BBGP Distribution\nMean: {np.mean(metrics['log2_BBGP']):.2f}")
    axs[0, 2].set_xlabel("log2(BBGP)")
    axs[0, 2].set_ylabel("Count")
    
    # Plot predicted metrics
    axs[1, 0].hist(metrics["pred_log_params"], bins=50, alpha=0.7)
    axs[1, 0].set_title(f"Predicted Log Parameters\nMean: {np.mean(metrics['pred_log_params']):.2f}")
    axs[1, 0].set_xlabel("pred_log(params)")
    axs[1, 0].set_ylabel("Count")
    
    axs[1, 1].hist(metrics["pred_log_FLOPs"], bins=50, alpha=0.7)
    axs[1, 1].set_title(f"Predicted Log FLOPs\nMean: {np.mean(metrics['pred_log_FLOPs']):.2f}")
    axs[1, 1].set_xlabel("pred_log(FLOPs)")
    axs[1, 1].set_ylabel("Count")
    
    axs[1, 2].hist(metrics["pred_log2_BBGP"], bins=50, alpha=0.7)
    axs[1, 2].set_title(f"Predicted Log2 BBGP\nMean: {np.mean(metrics['pred_log2_BBGP']):.2f}")
    axs[1, 2].set_xlabel("pred_log2(BBGP)")
    axs[1, 2].set_ylabel("Count")
    
    # Plot preference function values
    axs[2, 0].hist(metrics["params_pi"], bins=50, alpha=0.7)
    axs[2, 0].set_title(f"Parameters Preference\nMean: {np.mean(metrics['params_pi']):.2f}")
    axs[2, 0].set_xlabel("params_pi_func")
    axs[2, 0].set_ylabel("Count")
    
    axs[2, 1].hist(metrics["FLOPs_pi"], bins=50, alpha=0.7)
    axs[2, 1].set_title(f"FLOPs Preference\nMean: {np.mean(metrics['FLOPs_pi']):.2f}")
    axs[2, 1].set_xlabel("flops_pi_func")
    axs[2, 1].set_ylabel("Count")
    
    axs[2, 2].hist(metrics["BBGP_pi"], bins=50, alpha=0.7)
    axs[2, 2].set_title(f"BBGP Preference\nMean: {np.mean(metrics['BBGP_pi']):.2f}")
    axs[2, 2].set_xlabel("target_bbgp_pi_func")
    axs[2, 2].set_ylabel("Count")
    
    # Plot product preference function and raw metrics
    axs[3, 0].hist(metrics["prod_pi"], bins=50, alpha=0.7)
    axs[3, 0].set_title(f"Product Preference\nMean: {np.mean(metrics['prod_pi']):.2f}")
    axs[3, 0].set_xlabel("prod_pi_func")
    axs[3, 0].set_ylabel("Count")
    
    # Plot raw parameters and FLOPs with log scale
    axs[3, 1].hist(metrics["params"], bins=50, alpha=0.7)
    axs[3, 1].set_title(f"Raw Parameters\nMean: {np.mean(metrics['params']):,.0f}")
    axs[3, 1].set_xlabel("params")
    axs[3, 1].set_xscale('log')
    axs[3, 1].set_ylabel("Count")
    
    axs[3, 2].hist(metrics["FLOPs"], bins=50, alpha=0.7)
    axs[3, 2].set_title(f"Raw FLOPs\nMean: {np.mean(metrics['FLOPs']):,.0f}")
    axs[3, 2].set_xlabel("FLOPs")
    axs[3, 2].set_xscale('log')
    axs[3, 2].set_ylabel("Count")
    
    # Add a main title
    plt.suptitle(f"Analysis of {len(metrics['log_params'])} Sampled Architectures", 
                fontsize=16, y=0.98)
    
    # Save figure
    fig_path = os.path.join(save_path, "latent_metrics_histograms.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    
    # Create scatter plots to show relationships between metrics
    plt.figure(figsize=(18, 16))
    
    plt.subplot(2, 2, 1)
    plt.scatter(metrics["log_params"], metrics["log_FLOPs"], alpha=0.5)
    plt.title("Log Parameters vs Log FLOPs")
    plt.xlabel("log(params)")
    plt.ylabel("log(FLOPs)")
    
    plt.subplot(2, 2, 2)
    plt.scatter(metrics["log_params"], metrics["log2_BBGP"], alpha=0.5)
    plt.title("Log Parameters vs Log2 BBGP")
    plt.xlabel("log(params)")
    plt.ylabel("log2(BBGP)")
    
    plt.subplot(2, 2, 3)
    plt.scatter(metrics["log_FLOPs"], metrics["log2_BBGP"], alpha=0.5)
    plt.title("Log FLOPs vs Log2 BBGP")
    plt.xlabel("log(FLOPs)")
    plt.ylabel("log2(BBGP)")
    
    plt.subplot(2, 2, 4)
    plt.scatter(metrics["params_pi"] * metrics["FLOPs_pi"] * metrics["BBGP_pi"], 
               metrics["prod_pi"], alpha=0.5)
    plt.title("Product of Individual Pi Functions vs Prod Pi Function")
    plt.xlabel("params_pi * FLOPs_pi * BBGP_pi")
    plt.ylabel("prod_pi_func")
    
    corr_path = os.path.join(save_path, "metric_correlations.png")
    plt.tight_layout()
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    print(f"Saved correlation plots to {corr_path}")
    
    # Also create scatter plots of predicted vs actual metrics
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.scatter(metrics["pred_log_params"], metrics["log_params"], alpha=0.5)
    plt.title("Predicted vs Actual Log Parameters")
    plt.xlabel("Predicted log(params)")
    plt.ylabel("Actual log(params)")
    
    # Add a diagonal line for perfect prediction
    min_val = min(np.min(metrics["pred_log_params"]), np.min(metrics["log_params"]))
    max_val = max(np.max(metrics["pred_log_params"]), np.max(metrics["log_params"]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.subplot(1, 3, 2)
    plt.scatter(metrics["pred_log_FLOPs"], metrics["log_FLOPs"], alpha=0.5)
    plt.title("Predicted vs Actual Log FLOPs")
    plt.xlabel("Predicted log(FLOPs)")
    plt.ylabel("Actual log(FLOPs)")
    
    # Add a diagonal line for perfect prediction
    min_val = min(np.min(metrics["pred_log_FLOPs"]), np.min(metrics["log_FLOPs"]))
    max_val = max(np.max(metrics["pred_log_FLOPs"]), np.max(metrics["log_FLOPs"]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.subplot(1, 3, 3)
    plt.scatter(metrics["pred_log2_BBGP"], metrics["log2_BBGP"], alpha=0.5)
    plt.title("Predicted vs Actual Log2 BBGP")
    plt.xlabel("Predicted log2(BBGP)")
    plt.ylabel("Actual log2(BBGP)")
    
    # Add a diagonal line for perfect prediction
    min_val = min(np.min(metrics["pred_log2_BBGP"]), np.min(metrics["log2_BBGP"]))
    max_val = max(np.max(metrics["pred_log2_BBGP"]), np.max(metrics["log2_BBGP"]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    pred_path = os.path.join(save_path, "prediction_accuracy.png")
    plt.tight_layout()
    plt.savefig(pred_path, dpi=300, bbox_inches='tight')
    print(f"Saved prediction accuracy plots to {pred_path}")
    
    # Calculate correlation coefficients and save to a text file
    corr_file = os.path.join(save_path, "correlation_coefficients.txt")
    with open(corr_file, "w") as f:
        f.write("Correlation Coefficients between Metrics:\n\n")
        
        # Calculate correlations between actual metrics
        corr_params_flops = np.corrcoef(metrics["log_params"], metrics["log_FLOPs"])[0, 1]
        corr_params_bbgp = np.corrcoef(metrics["log_params"], metrics["log2_BBGP"])[0, 1]
        corr_flops_bbgp = np.corrcoef(metrics["log_FLOPs"], metrics["log2_BBGP"])[0, 1]
        
        f.write(f"log(params) vs log(FLOPs): {corr_params_flops:.4f}\n")
        f.write(f"log(params) vs log2(BBGP): {corr_params_bbgp:.4f}\n")
        f.write(f"log(FLOPs) vs log2(BBGP): {corr_flops_bbgp:.4f}\n\n")
        
        # Calculate correlations between predicted and actual
        corr_pred_params = np.corrcoef(metrics["pred_log_params"], metrics["log_params"])[0, 1]
        corr_pred_flops = np.corrcoef(metrics["pred_log_FLOPs"], metrics["log_FLOPs"])[0, 1]
        corr_pred_bbgp = np.corrcoef(metrics["pred_log2_BBGP"], metrics["log2_BBGP"])[0, 1]
        
        f.write(f"Predicted vs Actual log(params): {corr_pred_params:.4f}\n")
        f.write(f"Predicted vs Actual log(FLOPs): {corr_pred_flops:.4f}\n")
        f.write(f"Predicted vs Actual log2(BBGP): {corr_pred_bbgp:.4f}\n\n")
        
        # Calculate other useful statistics
        f.write("Summary Statistics:\n")
        f.write(f"Number of samples: {len(metrics['log_params'])}\n")
        f.write(f"Mean log(params): {np.mean(metrics['log_params']):.4f}\n")
        f.write(f"Mean log(FLOPs): {np.mean(metrics['log_FLOPs']):.4f}\n")
        f.write(f"Mean log2(BBGP): {np.mean(metrics['log2_BBGP']):.4f}\n")
        f.write(f"Mean prod_pi: {np.mean(metrics['prod_pi']):.4f}\n\n")
        
        # Calculate percentage of architectures with desired BBGP
        target_bbgp = 4  # Target BBGP is 2^2 = 4
        bbgp_values = 2 ** np.array(metrics["log2_BBGP"])
        close_to_target = np.abs(bbgp_values - target_bbgp) <= 1
        percent_close = np.mean(close_to_target) * 100
        
        f.write(f"Percentage of architectures with BBGP within ±1 of target (4): {percent_close:.2f}%\n")
    
    print(f"Saved correlation coefficients and statistics to {corr_file}")
    
    # Save the raw metrics data
    np.save(os.path.join(save_path, "latent_metrics_data.npy"), metrics)
    print(f"Saved raw metrics data to {save_path}/latent_metrics_data.npy")
    
    return metrics

# Example usage
if __name__ == "__main__":
    device = get_device()
    search_space = SearchSpace()
    
    # Create and load the autoencoder
    ae = ArcAE(search_space=search_space, z_dim=99, ae_type="WAE")
    # checkpoint = torch.load("checkpoints/arcae_20250403_083352/arcae_best.pt", map_location=device)
    checkpoint = torch.load("checkpoints/arcae_20250403_092511/arcae_best.pt", map_location=device)
    ae.load_state_dict(checkpoint['model_state_dict'])
    ae.to(device)
    
    # Sample and visualize latent vectors
    metrics = sample_and_visualize_latent_vectors(
        trained_ae=ae,
        n_samples=10000,
        input_shape=[3, 32],
        num_classes=10,
        save_dir="latent_analysis",
        device=device
    )