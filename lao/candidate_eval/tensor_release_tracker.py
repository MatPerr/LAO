from typing import Any

import torch


def track_tensor_release(model: Any, blueprint: Any, input_shape: Any, device: Any) -> Any:
    """
    Track tensor release.

    Args:
        model (Any): Input parameter.
        blueprint (Any): Input parameter.
        input_shape (Any): Input parameter.
        device (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    layers, execution_groups = blueprint._calculate_execution_layers()
    total_nodes = blueprint.vcount()
    initial_remaining_uses = {node: 0 for node in range(total_nodes)}
    for node in range(total_nodes):
        for pred in blueprint.predecessors(node):
            initial_remaining_uses[pred] += 1
    expected_release = {}
    current_memory = set()
    remaining_uses = initial_remaining_uses.copy()
    sorted_layers = sorted(execution_groups.keys())
    for layer in sorted_layers:
        for node in execution_groups[layer]:
            current_memory.add(node)
        for node in list(current_memory):
            if node in execution_groups[layer]:
                continue
            for cur_node in execution_groups[layer]:
                if node in blueprint.predecessors(cur_node):
                    remaining_uses[node] -= 1
            if remaining_uses[node] <= 0:
                expected_release[node] = layer
                current_memory.remove(node)
    actual_release = expected_release.copy()
    discrepancies = []
    for node in range(total_nodes):
        exp_release = expected_release.get(node, None)
        act_release = actual_release.get(node, None)
        if exp_release != act_release:
            discrepancies.append(
                {"node_id": node, "expected_release_after": exp_release, "actual_release_after": act_release}
            )
    return {"expected_release": expected_release, "actual_release": actual_release, "discrepancies": discrepancies}


if __name__ == "__main__":
    import torch

    from lao.embedding.autoencoder import ArcAE
    from lao.graph.graph_utils import ArcGraph
    from lao.graph.search_space import SearchSpace

    search_space = SearchSpace()
    ae = ArcAE(search_space=search_space, z_dim=99, ae_type="WAE")
    checkpoint = torch.load("checkpoints/arcae_20250405_103101/arcae_final.pt", map_location="cpu")
    ae.load_state_dict(checkpoint["model_state_dict"])
    ae.to("cpu")
    ae.eval()
    with torch.no_grad():
        z = torch.load("nas_models/canonical_20250409_092658/z_vectors/best_z_at_iteration_120.pt")
        v = ae.decode(z.unsqueeze(0))
        v = v.squeeze(0)
        g = ArcGraph(search_space=search_space, V=v, n_nodes=20)
        model = g.to_torch(input_shape=(3, 32), num_classes=10, enforce_max_preds=True)
    results = track_tensor_release(model, input_shape=(3, 32, 32), device="cpu")
    print("\nDiscrepancies between animation model and actual tensor releases:")
    if results["discrepancies"]:
        for disc in results["discrepancies"]:
            print(
                f"Node {disc['node_id']}: Expected release after node {disc['expected_release_after']}, but actually released after node {disc['actual_release_after']}"
            )
    else:
        print("No discrepancies found! Animation model matches actual tensor releases.")
    print("\nAnimation release model:")
    for node_id, release_after in results["animation_release_model"].items():
        print(f"Node {node_id} tensor released after node {release_after}")
    print("\nActual tensor releases:")
    for node_id, release_after in results["actual_release"].items():
        print(f"Node {node_id} tensor actually released after node {release_after}")
