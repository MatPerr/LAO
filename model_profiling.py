import torch
from deepspeed.profiling.flops_profiler import get_model_profile


def count_aggtensors_flops_from_topology(model, input_shape=(1, 3, 32, 32)):
    """
    Count FLOPs for AggTensors modules by examining the model topology directly.
    """
    agg_flops = 0
    
    # Check if the model has the topology attribute
    if not hasattr(model, 'topology'):
        print("Model doesn't have a topology attribute.")
        return 0
    
    # First, run a forward pass to fill the model's outputs dictionary
    # print("Running forward pass to collect tensor shapes...")
    dummy_input = torch.zeros(*input_shape)
    outputs = {}
    
    # Create a hook to capture outputs from each layer
    handles = []
    
    def capture_output_hook(name):
        def hook(module, input, output):
            outputs[name] = output
        return hook
    
    # Register hooks for capturing tensor shapes
    for name, module in model.named_modules():
        if "conv" in name or "bn" in name or "act" in name:
            handles.append(module.register_forward_hook(capture_output_hook(name)))
    
    # Run the forward pass
    with torch.no_grad():
        model(dummy_input)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Find AggTensors modules
    for name, module in model.named_modules():
        if hasattr(module, 'aggregation'):
            if module.aggregation == 'sum':
                # print(f"Found AggTensors module: {name}")
                try:
                    # Extract the node index from the name (e.g., "agg_2" -> 2)
                    node_idx = int(name.split('_')[-1])
                    
                    # Find the predecessors in the model's topology
                    for topology_node_idx, predecessors in model.topology:
                        if topology_node_idx == node_idx:
                            # print(f"Found node {node_idx} in topology with {len(predecessors)} predecessors")
                            
                            if len(predecessors) > 1:
                                # Get shapes of predecessors' outputs if available
                                pred_shapes = []
                                for pred_idx in predecessors:
                                    # Look for outputs from layers that would connect to this node
                                    relevant_outputs = [
                                        out for out_name, out in outputs.items() 
                                        if f"_{pred_idx}" in out_name
                                    ]
                                    
                                    if relevant_outputs:
                                        # Use the last relevant output as the shape
                                        pred_shapes.append(relevant_outputs[-1].shape)
                                
                                if not pred_shapes:
                                    # print(f"Could not find predecessor shapes for node {node_idx}")
                                    continue
                                
                                # Use the first predecessor shape as reference (all predecessors are mde to have the same size before summation)
                                if pred_shapes:
                                    # Calculate elements in the tensor
                                    shape = pred_shapes[0]
                                    num_elements = 1
                                    for dim in shape:
                                        num_elements *= dim
                                    
                                    # For sum: (n-1) additions per element
                                    module_flops = (len(predecessors) - 1) * num_elements
                                    agg_flops += module_flops
                                    # print(f"Node {node_idx} processes tensors of shape {shape}, estimated {module_flops} FLOPs")
                            
                            break
                except (ValueError, AttributeError, IndexError) as e:
                    print(f"Error processing {name}: {e}")
            
            else:
                raise NotImplementedError(f"FLOPs count for aggregation = {module.aggregation} not implemented")
    
    return agg_flops


def profile_model(model, input_shape=(1, 3, 32, 32)):
    # Step 1: Get baseline FLOPs from DeepSpeed
    try:
        # Try to run DeepSpeed's profiler
        flops, _, params = get_model_profile(model, input_shape, print_profile=False, as_string=False)
    except Exception as e:
        print(f"DeepSpeed profiler error: {e}")
        # Fallback to just parameter counting
        params = sum(p.numel() for p in model.parameters())
        flops = 0
    
    # Step 2: Count AggTensors FLOPs using the topology
    agg_flops = count_aggtensors_flops_from_topology(model, input_shape)
    
    return params, flops + agg_flops