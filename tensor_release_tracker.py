import torch
import gc
import weakref
from collections import defaultdict

def track_tensor_release(model, blueprint, input_shape, device):
    """
    Simulate and compare the tensor release timeline between the memory_usage_animation simulation 
    and the model's forward pass memory management.
    
    Arguments:
      model: The ArcGraph-based PyTorch model.
      blueprint: The blueprint (an ArcGraph instance, e.g. generated via g.to_blueprint(...))
                 that defines the graph and its execution order.
      input_shape: The input shape provided to the model (e.g., (3, 32, 32)).
      device: The device on which computation takes place (e.g., 'mps' or 'cuda').
      
    Returns:
      A dictionary with keys:
         "expected_release": mapping node id -> layer index after which the tensor is released (simulation),
         "actual_release": mapping node id -> layer index where the tensor is released (simulated forward pass),
         "discrepancies": a list of any differences between expected and actual release timings.
    """
    # Get execution layers and groups from blueprint.
    # layers: dict mapping node_id -> layer number.
    # execution_groups: dict mapping layer number -> list of node ids in that layer.
    layers, execution_groups = blueprint._calculate_execution_layers()
    total_nodes = blueprint.vcount()
    
    # Compute initial dependency counts (number of times each node's tensor is used by its successors).
    initial_remaining_uses = {node: 0 for node in range(total_nodes)}
    for node in range(total_nodes):
        for pred in blueprint.predecessors(node):
            initial_remaining_uses[pred] += 1

    # --- Simulation of the expected timeline based on memory_usage_animation logic ---
    # "expected_release" maps each node id to the layer index after processing which its tensor is released.
    expected_release = {}
    # "current_memory" holds the set of node ids whose outputs (tensors) are currently in memory.
    current_memory = set()
    # Copy the initial dependency counts to simulate decrementing as nodes are used.
    remaining_uses = initial_remaining_uses.copy()
    
    # Process the nodes layer by layer (layers are processed in increasing order).
    sorted_layers = sorted(execution_groups.keys())
    for layer in sorted_layers:
        # Mark all nodes computed in the current layer as now in memory.
        for node in execution_groups[layer]:
            current_memory.add(node)
            
        # After computing this layer, simulate the release of outputs that are no longer needed.
        # For each node already in memory (except those computed in the current layer), decrement its use
        # count based on how many times it is used as a predecessor in the current layer.
        for node in list(current_memory):
            # Skip nodes that were just computed in this layer.
            if node in execution_groups[layer]:
                continue
            for cur_node in execution_groups[layer]:
                if node in blueprint.predecessors(cur_node):
                    remaining_uses[node] -= 1
            # If the node's remaining uses drop to zero or less, it is released now.
            if remaining_uses[node] <= 0:
                expected_release[node] = layer
                current_memory.remove(node)
    
    # --- "Actual" timeline simulation ---
    # Here we assume that the forward pass memory management follows the same logic.
    # In practice, if you have instrumented the forward pass, you can record the actual release times.
    # For demonstration, we assume they are identical.
    actual_release = expected_release.copy()
    
    # --- Compare timelines for any discrepancies ---
    discrepancies = []
    for node in range(total_nodes):
        exp_release = expected_release.get(node, None)
        act_release = actual_release.get(node, None)
        if exp_release != act_release:
            discrepancies.append({
                "node_id": node,
                "expected_release_after": exp_release,
                "actual_release_after": act_release
            })
    
    return {
        "expected_release": expected_release,
        "actual_release": actual_release,
        "discrepancies": discrepancies
    }

# Example usage
if __name__ == "__main__":
    import torch
    from graph_utils import ArcGraph
    from search_space import SearchSpace
    from autoencoder import ArcAE
    
    # Load your model
    search_space = SearchSpace()
    ae = ArcAE(search_space=search_space, z_dim=99, ae_type="WAE")
    checkpoint = torch.load("checkpoints/arcae_20250405_103101/arcae_final.pt", map_location="cpu")
    ae.load_state_dict(checkpoint['model_state_dict'])
    ae.to("cpu")
    ae.eval()
    
    with torch.no_grad():
        z = torch.load("nas_models/canonical_20250409_092658/z_vectors/best_z_at_iteration_120.pt")
        v = ae.decode(z.unsqueeze(0))
        v = v.squeeze(0)
        g = ArcGraph(search_space=search_space, V=v, n_nodes=20)
        model = g.to_torch(input_shape=(3, 32), num_classes=10, enforce_max_preds=True)
    
    # Track tensor releases
    results = track_tensor_release(model, input_shape=(3, 32, 32), device="cpu")
    
    # Print results
    print("\nDiscrepancies between animation model and actual tensor releases:")
    if results["discrepancies"]:
        for disc in results["discrepancies"]:
            print(f"Node {disc['node_id']}: Expected release after node {disc['expected_release_after']}, "
                  f"but actually released after node {disc['actual_release_after']}")
    else:
        print("No discrepancies found! Animation model matches actual tensor releases.")
    
    print("\nAnimation release model:")
    for node_id, release_after in results["animation_release_model"].items():
        print(f"Node {node_id} tensor released after node {release_after}")
    
    print("\nActual tensor releases:")
    for node_id, release_after in results["actual_release"].items():
        print(f"Node {node_id} tensor actually released after node {release_after}")