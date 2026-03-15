from typing import Any

import torch
from deepspeed.profiling.flops_profiler import get_model_profile


def count_aggtensors_flops_from_topology(model: Any, input_shape: Any = (1, 3, 32, 32)) -> Any:
    """
    Count aggtensors flops from topology.

    Args:
        model (Any): Input parameter.
        input_shape (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    agg_flops = 0
    if not hasattr(model, "topology"):
        print("Model doesn't have a topology attribute.")
        return 0
    dummy_input = torch.zeros(*input_shape)
    outputs = {}
    handles = []

    def capture_output_hook(name: Any) -> Any:
        """Capture output hook."""

        def hook(module: Any, input: Any, output: Any) -> Any:
            """Hook."""
            outputs[name] = output

        return hook

    for name, module in model.named_modules():
        if "conv" in name or "bn" in name or "act" in name:
            handles.append(module.register_forward_hook(capture_output_hook(name)))
    with torch.no_grad():
        model(dummy_input)
    for handle in handles:
        handle.remove()
    for name, module in model.named_modules():
        if hasattr(module, "aggregation"):
            if module.aggregation == "sum":
                try:
                    node_idx = int(name.split("_")[-1])
                    for topology_node_idx, predecessors in model.topology:
                        if topology_node_idx == node_idx:
                            if len(predecessors) > 1:
                                pred_shapes = []
                                for pred_idx in predecessors:
                                    relevant_outputs = [
                                        out for out_name, out in outputs.items() if f"_{pred_idx}" in out_name
                                    ]
                                    if relevant_outputs:
                                        pred_shapes.append(relevant_outputs[-1].shape)
                                if not pred_shapes:
                                    continue
                                if pred_shapes:
                                    shape = pred_shapes[0]
                                    num_elements = 1
                                    for dim in shape:
                                        num_elements *= dim
                                    module_flops = (len(predecessors) - 1) * num_elements
                                    agg_flops += module_flops
                            break
                except (ValueError, AttributeError, IndexError) as e:
                    print(f"Error processing {name}: {e}")
            else:
                raise NotImplementedError(f"FLOPs count for aggregation = {module.aggregation} not implemented")
    return agg_flops


def count_sigmoid_and_se_flops(model: Any, input_shape: Any = (1, 3, 32, 32)) -> Any:
    """
    Count sigmoid and se flops.

    Args:
        model (Any): Input parameter.
        input_shape (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    sigmoid_flops = 0
    se_multiplication_flops = 0
    customse_sigmoid_modules = set()
    for name, module in model.named_modules():
        if "CustomSE" in module.__class__.__name__:
            for subname, submodule in module.named_modules():
                if isinstance(submodule, torch.nn.Sigmoid):
                    if subname:
                        full_path = f"{name}.{subname}"
                    else:
                        full_path = f"{name}.sigmoid"
                    customse_sigmoid_modules.add(full_path)
    dummy_input = torch.zeros(*input_shape)
    sigmoid_outputs = {}
    se_inputs = {}
    sigmoid_handles = []
    se_handles = []

    def capture_sigmoid_output_hook(name: Any) -> Any:
        """Capture sigmoid output hook."""

        def hook(module: Any, input: Any, output: Any) -> Any:
            """Hook."""
            sigmoid_outputs[name] = output

        return hook

    def capture_se_input_hook(name: Any) -> Any:
        """Capture se input hook."""

        def hook(module: Any, input: Any, output: Any) -> Any:
            """Hook."""
            se_inputs[name] = input[0]

        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Sigmoid) and name not in customse_sigmoid_modules:
            sigmoid_handles.append(module.register_forward_hook(capture_sigmoid_output_hook(name)))
        if "CustomSE" in module.__class__.__name__:
            se_handles.append(module.register_forward_hook(capture_se_input_hook(name)))
            if hasattr(module, "sigmoid"):
                sigmoid_handles.append(
                    module.sigmoid.register_forward_hook(capture_sigmoid_output_hook(f"{name}.sigmoid"))
                )
    with torch.no_grad():
        model(dummy_input)
    for handle in sigmoid_handles:
        handle.remove()
    for handle in se_handles:
        handle.remove()
    for name, output in sigmoid_outputs.items():
        num_elements = output.numel()
        module_flops = 4 * num_elements
        sigmoid_flops += module_flops
    for name, module in model.named_modules():
        if "CustomSE" in module.__class__.__name__:
            if name in se_inputs:
                input_tensor = se_inputs[name]
                num_elements = input_tensor.numel()
                module_flops = num_elements
                se_multiplication_flops += module_flops
    return (sigmoid_flops, se_multiplication_flops)


def profile_model(model: Any, input_shape: Any = (1, 3, 32, 32)) -> Any:
    """
    Profile model.

    Args:
        model (Any): Input parameter.
        input_shape (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    try:
        flops, _, params = get_model_profile(model, input_shape, print_profile=False, as_string=False)
    except Exception as e:
        print(f"DeepSpeed profiler error: {e}")
        params = sum((p.numel() for p in model.parameters()))
        flops = 0
    agg_flops = count_aggtensors_flops_from_topology(model, input_shape)
    sigmoid_flops, se_multiplication_flops = count_sigmoid_and_se_flops(model, input_shape)
    detailed_flops = {
        "deepspeed_flops": flops,
        "aggtensors_flops": agg_flops,
        "sigmoid_flops": sigmoid_flops,
        "se_multiplication_flops": se_multiplication_flops,
        "total_flops": flops + agg_flops + sigmoid_flops + se_multiplication_flops,
    }
    return (params, detailed_flops["total_flops"])
