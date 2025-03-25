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
                                
                                # Use the first predecessor shape as reference (all predecessors are made to have the same size before summation)
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


def count_sigmoid_and_se_flops(model, input_shape=(1, 3, 32, 32)):
    """
    Count FLOPs for Sigmoid operations (4 FLOPs per element) and CustomSE modules.
    """
    sigmoid_flops = 0
    se_multiplication_flops = 0
    
    # Keep track of sigmoid modules within CustomSE to avoid double counting
    customse_sigmoid_modules = set()
    
    # First identify all sigmoid modules that are part of CustomSE
    for name, module in model.named_modules():
        if "CustomSE" in module.__class__.__name__:
            # Find the sigmoid submodule within CustomSE
            for subname, submodule in module.named_modules():
                if isinstance(submodule, torch.nn.Sigmoid):
                    # Store the full path to this sigmoid module
                    if subname:
                        full_path = f"{name}.{subname}"
                    else:
                        full_path = f"{name}.sigmoid"  # If it's the direct .sigmoid attribute
                    customse_sigmoid_modules.add(full_path)
    
    # Run a forward pass to collect tensor shapes
    dummy_input = torch.zeros(*input_shape)
    sigmoid_outputs = {}
    se_inputs = {}
    
    # Create hooks to capture outputs from sigmoid and inputs to CustomSE
    sigmoid_handles = []
    se_handles = []
    
    def capture_sigmoid_output_hook(name):
        def hook(module, input, output):
            sigmoid_outputs[name] = output
        return hook
    
    def capture_se_input_hook(name):
        def hook(module, input, output):
            se_inputs[name] = input[0]  # input is a tuple, so we take the first element
        return hook
    
    # Register hooks for capturing tensor shapes
    for name, module in model.named_modules():
        # Only hook standalone sigmoids (not part of CustomSE)
        if isinstance(module, torch.nn.Sigmoid) and name not in customse_sigmoid_modules:
            sigmoid_handles.append(module.register_forward_hook(capture_sigmoid_output_hook(name)))
        if "CustomSE" in module.__class__.__name__:
            se_handles.append(module.register_forward_hook(capture_se_input_hook(name)))
            # Also specifically hook the sigmoid within CustomSE to get its output shape
            if hasattr(module, 'sigmoid'):
                sigmoid_handles.append(module.sigmoid.register_forward_hook(
                    capture_sigmoid_output_hook(f"{name}.sigmoid")))
    
    # Run the forward pass
    with torch.no_grad():
        model(dummy_input)
    
    # Remove hooks
    for handle in sigmoid_handles:
        handle.remove()
    for handle in se_handles:
        handle.remove()
    
    # Count standalone sigmoid FLOPs (4 FLOPs per element)
    for name, output in sigmoid_outputs.items():
        num_elements = output.numel()
        module_flops = 4 * num_elements  # 4 FLOPs per element for sigmoid
        sigmoid_flops += module_flops
        # print(f"Sigmoid {name} processes {num_elements} elements, estimated {module_flops} FLOPs")
    
    # Count CustomSE element-wise multiplication FLOPs
    for name, module in model.named_modules():
        if "CustomSE" in module.__class__.__name__:
            if name in se_inputs:
                input_tensor = se_inputs[name]
                num_elements = input_tensor.numel()
                # One multiplication per element for module_input * x
                module_flops = num_elements
                se_multiplication_flops += module_flops
                # print(f"CustomSE {name} multiplication: {num_elements} elements, estimated {module_flops} FLOPs")
    
    return sigmoid_flops, se_multiplication_flops


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
    
    # Step 3: Count Sigmoid and CustomSE FLOPs
    sigmoid_flops, se_multiplication_flops = count_sigmoid_and_se_flops(model, input_shape)
    
    # Detailed breakdown
    detailed_flops = {
        'deepspeed_flops': flops,
        'aggtensors_flops': agg_flops,
        'sigmoid_flops': sigmoid_flops,
        'se_multiplication_flops': se_multiplication_flops,
        'total_flops': flops + agg_flops + sigmoid_flops + se_multiplication_flops
    }
    
    return params, detailed_flops['total_flops']