from typing import Any

import numpy as np


def make_node_valid(graph: Any, node_idx: Any, input_shape: Any, feature_index_fn: Any) -> Any:
    """
    Make node valid.

    Args:
        graph (Any): Input parameter.
        node_idx (Any): Input parameter.
        input_shape (Any): Input parameter.
        feature_index_fn (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    assert "features" in graph.vs[node_idx].attribute_names() and graph.vs[node_idx]["features"] is not None, (
        "Node must have features to make it valid"
    )
    features = graph.vs[node_idx]["features"]
    input_channels, input_breadth = input_shape
    stride_idx = feature_index_fn("stride", graph.search_space)
    stride_values = graph.search_space.node_features.stride
    valid_strides = [s for s in stride_values if input_breadth % s == 0]
    if not valid_strides:
        features[stride_idx] = 1
    elif features[stride_idx] not in valid_strides:
        features[stride_idx] = min(valid_strides, key=lambda s: abs(s - features[stride_idx]))
    out_channels_idx = feature_index_fn("out_channels", graph.search_space)
    groups_idx = feature_index_fn("groups", graph.search_space)
    out_channels = features[out_channels_idx]
    groups = features[groups_idx]
    if groups == -1:
        valid_out_channels = [oc for oc in graph.search_space.node_features.out_channels if oc % input_channels == 0]
        if valid_out_channels:
            features[out_channels_idx] = min(valid_out_channels, key=lambda oc: abs(oc - out_channels))
            features[groups_idx] = input_channels
        else:
            max_divisor = max(
                [
                    g
                    for g in graph.search_space.node_features.groups
                    if g != -1 and input_channels % g == 0 and (g <= input_channels)
                ],
                default=1,
            )
            features[groups_idx] = max_divisor
    else:
        valid_groups = [
            g
            for g in graph.search_space.node_features.groups
            if g != -1 and input_channels % g == 0 and (out_channels % g == 0)
        ]
        if groups not in valid_groups:
            if valid_groups:
                features[groups_idx] = min(valid_groups, key=lambda g: abs(g - groups))
            else:
                for g in sorted(graph.search_space.node_features.groups, reverse=True):
                    if g != -1 and input_channels % g == 0:
                        valid_out_channels = [oc for oc in graph.search_space.node_features.out_channels if oc % g == 0]
                        if valid_out_channels:
                            features[groups_idx] = g
                            features[out_channels_idx] = min(valid_out_channels, key=lambda oc: abs(oc - out_channels))
                            break
                else:
                    features[groups_idx] = 1


def sample_node_features(
    graph: Any, input_shape: Any, constraints: Any, feature_index_fn: Any, constraints_satisfied_fn: Any
) -> Any:
    """
    Sample node features.

    Args:
        graph (Any): Input parameter.
        input_shape (Any): Input parameter.
        constraints (Any): Input parameter.
        feature_index_fn (Any): Input parameter.
        constraints_satisfied_fn (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    n_nodes = len(graph.vs)

    def sample_for_node(node_idx: Any, node_input_shape: Any) -> Any:
        """
        Sample for node.

        Args:
            node_idx (Any): Input parameter.
            node_input_shape (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        graph.vs[node_idx]["input_shape"] = node_input_shape
        if graph.search_space.node_feature_probs.out_channels == "uniform":
            out_channels = np.random.choice(graph.search_space.node_features.out_channels)
        elif isinstance(graph.search_space.node_feature_probs.out_channels, list):
            out_channels = np.random.choice(
                graph.search_space.node_features.out_channels, p=graph.search_space.node_feature_probs.out_channels
            )
        else:
            raise ValueError("Unsupported out_channels probability configuration")
        if graph.search_space.node_feature_probs.stride == "uniform":
            possible_strides = [s for s in graph.search_space.node_features.stride if node_input_shape[1] % s == 0]
            stride = np.random.choice(possible_strides)
        elif isinstance(graph.search_space.node_feature_probs.stride, list):
            all_strides = graph.search_space.node_features.stride
            possible_strides_idx = [all_strides.index(s) for s in all_strides if node_input_shape[1] % s == 0]
            possible_strides = [all_strides[i] for i in possible_strides_idx]
            possible_strides_weights = [graph.search_space.node_feature_probs.stride[i] for i in possible_strides_idx]
            possible_strides_probs = np.array(possible_strides_weights) / np.sum(possible_strides_weights)
            stride = np.random.choice(possible_strides, p=possible_strides_probs)
        else:
            raise ValueError("Unsupported stride probability configuration")
        if graph.search_space.node_feature_probs.groups == "uniform":
            possible_groups = [
                g
                for g in graph.search_space.node_features.groups
                if node_input_shape[0] % g == 0 and out_channels % g == 0
            ]
            groups = np.random.choice(possible_groups)
        elif isinstance(graph.search_space.node_feature_probs.groups, list):
            all_groups = graph.search_space.node_features.groups
            possible_groups_idx = [
                all_groups.index(g) for g in all_groups if node_input_shape[0] % g == 0 and out_channels % g == 0
            ]
            possible_groups = [all_groups[i] for i in possible_groups_idx]
            possible_groups_weights = [graph.search_space.node_feature_probs.groups[i] for i in possible_groups_idx]
            possible_groups_probs = np.array(possible_groups_weights) / np.sum(possible_groups_weights)
            groups = np.random.choice(possible_groups, p=possible_groups_probs)
        else:
            raise ValueError("Unsupported groups probability configuration")
        feature_dict = graph.search_space.node_features.__dict__
        node_features = np.empty(len(feature_dict), dtype="object")
        for feature_name, feature_values in feature_dict.items():
            index = feature_index_fn(feature_name, graph.search_space)
            if feature_name == "out_channels":
                node_features[index] = out_channels
            elif feature_name == "stride":
                node_features[index] = stride
            elif feature_name == "groups":
                node_features[index] = groups
            else:
                probs = graph.search_space.node_feature_probs.__dict__[feature_name]
                if probs == "uniform":
                    node_features[index] = np.random.choice(feature_values)
                elif isinstance(probs, list):
                    node_features[index] = np.random.choice(feature_values, p=probs)
                else:
                    raise ValueError(f"Unsupported probability config for feature '{feature_name}'")
        graph.vs[node_idx]["features"] = node_features.tolist()
        graph.vs[node_idx]["output_shape"] = [out_channels, node_input_shape[1] // stride]

    graph.constraints_met = True
    n_params = 0
    flops = 0
    sample_for_node(0, input_shape)
    graph._add_node_n_params_and_FLOPs(0)
    n_params += graph.vs[0]["n_params"]
    flops += graph.vs[0]["FLOPs"]
    if constraints is not None:
        if not constraints_satisfied_fn(
            {"n_params": n_params, "FLOPs": flops}, constraints, allowed_keys={"n_params", "FLOPs"}
        ):
            graph.constraints_met = False
    for i in range(1, n_nodes):
        current_input_shape = graph._get_input_shape(i)
        sample_for_node(i, current_input_shape)
        graph._add_node_n_params_and_FLOPs(i)
        n_params += graph.vs[i]["n_params"]
        flops += graph.vs[i]["FLOPs"]
        if constraints is not None:
            if not constraints_satisfied_fn(
                {"n_params": n_params, "FLOPs": flops}, constraints, allowed_keys={"n_params", "FLOPs"}
            ):
                graph.constraints_met = False
                break
