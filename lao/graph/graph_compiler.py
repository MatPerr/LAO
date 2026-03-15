from typing import Any


def layer_blueprint(graph: Any) -> Any:
    """
    Layer blueprint.

    Args:
        graph (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    layer_of_node_dict = {}
    visited = set()

    def dfs(node_idx: Any) -> Any:
        """
        Dfs.

        Args:
            node_idx (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        if node_idx in visited:
            return layer_of_node_dict[node_idx]
        visited.add(node_idx)
        predecessors = graph.predecessors(node_idx)
        if not predecessors:
            layer_of_node_dict[node_idx] = 0
            return 0
        max_layer = -1
        for pred in predecessors:
            pred_layer = dfs(pred)
            max_layer = max(max_layer, pred_layer)
        layer_of_node_dict[node_idx] = max_layer + 1
        return layer_of_node_dict[node_idx]

    for i in range(graph.vcount()):
        if i not in visited:
            dfs(i)
    nodes_of_layer_dict = {}
    for node, layer in layer_of_node_dict.items():
        nodes_of_layer_dict.setdefault(layer, []).append(node)
    return (layer_of_node_dict, nodes_of_layer_dict)


def to_blueprint(
    graph: Any,
    input_shape: Any = (3, 32),
    num_classes: Any = 10,
    enforce_max_preds: Any = False,
    topological_sort: Any = False,
    feature_index_fn: Any = None,
    make_node_valid_fn: Any = None,
) -> Any:
    """
    To blueprint.

    Args:
        graph (Any): Input parameter.
        input_shape (Any): Input parameter.
        num_classes (Any): Input parameter.
        enforce_max_preds (Any): Input parameter.
        topological_sort (Any): Input parameter.
        feature_index_fn (Any): Input parameter.
        make_node_valid_fn (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    assert graph.search_space is not None, "search_space must be provided to create a blueprint"
    blueprint = graph.__class__(search_space=graph.search_space)
    node_mapping = {}
    for i in range(graph.vcount()):
        blueprint.add_vertex()
        node_mapping[i] = i
        if "features" in graph.vs[i].attribute_names() and graph.vs[i]["features"] is not None:
            blueprint.vs[i]["features"] = list(graph.vs[i]["features"])
        if "n_params" in graph.vs[i].attribute_names() and graph.vs[i]["n_params"] is not None:
            blueprint.vs[i]["n_params"] = graph.vs[i]["n_params"]
        if "FLOPs" in graph.vs[i].attribute_names() and graph.vs[i]["FLOPs"] is not None:
            blueprint.vs[i]["FLOPs"] = graph.vs[i]["FLOPs"]
    for i in range(blueprint.vcount()):
        blueprint.vs[i]["node_type"] = "original"
    blueprint.vs[node_mapping[0]]["input_shape"] = input_shape
    make_node_valid_fn(blueprint, node_mapping[0], input_shape)
    node_features = blueprint.vs[node_mapping[0]]["features"]
    stride = node_features[feature_index_fn("stride", graph.search_space)]
    out_channels = node_features[feature_index_fn("out_channels", graph.search_space)]
    blueprint.vs[node_mapping[0]]["output_shape"] = [out_channels, input_shape[1] // stride]
    blueprint._add_node_n_params_and_FLOPs(node_mapping[0])
    for i in range(1, graph.vcount()):
        original_predecessors = graph.predecessors(i)
        if enforce_max_preds and len(original_predecessors) > graph.search_space.graph_features.max_preds:
            original_predecessors = sorted(original_predecessors, reverse=True)
            original_predecessors = original_predecessors[: graph.search_space.graph_features.max_preds]
        pred_shapes = []
        for pred_idx in original_predecessors:
            mapped_pred = node_mapping[pred_idx]
            pred_shapes.append(blueprint.vs[mapped_pred]["output_shape"])
        max_channels = max([shape[0] for shape in pred_shapes])
        min_spatial = min([shape[1] for shape in pred_shapes])
        for pred_idx in original_predecessors:
            mapped_pred = node_mapping[pred_idx]
            pred_shape = blueprint.vs[mapped_pred]["output_shape"]
            current_src = mapped_pred
            if pred_shape[1] > min_spatial:
                blueprint.add_vertex()
                adapter_idx = blueprint.vcount() - 1
                blueprint.vs[adapter_idx]["node_type"] = "breadth_adapter"
                reduction_factor = pred_shape[1] // min_spatial
                blueprint.vs[adapter_idx]["input_shape"] = list(pred_shape)
                blueprint.vs[adapter_idx]["output_shape"] = [pred_shape[0], min_spatial]
                blueprint.vs[adapter_idx]["reduc_factor"] = reduction_factor
                blueprint.vs[adapter_idx]["FLOPs"] = (
                    reduction_factor**2 * (pred_shape[1] // reduction_factor) ** 2 * pred_shape[0]
                )
                blueprint.add_edge(current_src, adapter_idx)
                current_src = adapter_idx
            if pred_shape[0] < max_channels:
                blueprint.add_vertex()
                adapter_idx = blueprint.vcount() - 1
                blueprint.vs[adapter_idx]["node_type"] = "channel_adapter"
                if current_src != mapped_pred:
                    current_shape = blueprint.vs[current_src]["output_shape"]
                else:
                    current_shape = pred_shape
                blueprint.vs[adapter_idx]["input_shape"] = list(current_shape)
                blueprint.vs[adapter_idx]["output_shape"] = [max_channels, current_shape[1]]
                blueprint.vs[adapter_idx]["padding"] = max_channels - current_shape[0]
                blueprint.add_edge(current_src, adapter_idx)
                current_src = adapter_idx
            blueprint.add_edge(current_src, node_mapping[i])
        blueprint.vs[node_mapping[i]]["input_shape"] = [max_channels, min_spatial]
        make_node_valid_fn(blueprint, node_mapping[i], [max_channels, min_spatial])
        current_input_shape = blueprint.vs[node_mapping[i]]["input_shape"]
        node_features = blueprint.vs[node_mapping[i]]["features"]
        stride = node_features[feature_index_fn("stride", graph.search_space)]
        out_channels = node_features[feature_index_fn("out_channels", graph.search_space)]
        blueprint.vs[node_mapping[i]]["output_shape"] = [out_channels, current_input_shape[1] // stride]
        blueprint._add_node_n_params_and_FLOPs(node_mapping[i])
    sink_nodes = []
    for i in range(blueprint.vcount()):
        if (
            "node_type" in blueprint.vs[i].attribute_names()
            and blueprint.vs[i]["node_type"] == "original"
            and (blueprint.outdegree(i) == 0)
        ):
            sink_nodes.append(i)
    if not sink_nodes and blueprint.vcount() > 0:
        sink_nodes = [node_mapping[graph.vcount() - 1]]
    blueprint.add_vertex()
    gap_idx = blueprint.vcount() - 1
    blueprint.vs[gap_idx]["node_type"] = "global_avg_pool"
    if len(sink_nodes) > 1:
        all_sink_shapes = [blueprint.vs[node]["output_shape"] for node in sink_nodes]
        max_channels = max([shape[0] for shape in all_sink_shapes])
        for sink_idx in sink_nodes:
            sink_shape = blueprint.vs[sink_idx]["output_shape"]
            current_src = sink_idx
            if sink_shape[0] < max_channels:
                blueprint.add_vertex()
                adapter_idx = blueprint.vcount() - 1
                blueprint.vs[adapter_idx]["node_type"] = "channel_adapter"
                blueprint.vs[adapter_idx]["input_shape"] = list(sink_shape)
                blueprint.vs[adapter_idx]["output_shape"] = [max_channels, sink_shape[1]]
                blueprint.vs[adapter_idx]["padding"] = max_channels - sink_shape[0]
                blueprint.add_edge(current_src, adapter_idx)
                current_src = adapter_idx
                sink_shape = blueprint.vs[adapter_idx]["output_shape"]
            blueprint.add_edge(current_src, gap_idx)
        blueprint.vs[gap_idx]["input_shape"] = [max_channels, sink_shape[1]]
    else:
        sink_idx = sink_nodes[0]
        sink_shape = blueprint.vs[sink_idx]["output_shape"]
        blueprint.vs[gap_idx]["input_shape"] = list(sink_shape)
        blueprint.add_edge(sink_idx, gap_idx)
    gap_input_shape = blueprint.vs[gap_idx]["input_shape"]
    blueprint.BBGP = gap_input_shape[1]
    blueprint.vs[gap_idx]["output_shape"] = [gap_input_shape[0], 1]
    gap_flops = gap_input_shape[0] * gap_input_shape[1] ** 2
    if len(blueprint.predecessors(gap_idx)) > 1:
        gap_flops += (len(blueprint.predecessors(gap_idx)) - 1) * gap_input_shape[0] * gap_input_shape[1] ** 2
    blueprint.vs[gap_idx]["FLOPs"] = gap_flops
    blueprint.add_vertex()
    classifier_idx = blueprint.vcount() - 1
    blueprint.vs[classifier_idx]["node_type"] = "classifier"
    in_features = blueprint.vs[gap_idx]["output_shape"][0]
    blueprint.vs[classifier_idx]["input_shape"] = [in_features, 1]
    blueprint.vs[classifier_idx]["output_shape"] = [num_classes, 1]
    blueprint.vs[classifier_idx]["n_params"] = in_features * num_classes + num_classes
    blueprint.vs[classifier_idx]["FLOPs"] = 2 * in_features * num_classes
    blueprint.add_edge(gap_idx, classifier_idx)
    blueprint.valid = True
    blueprint.shapes_added = True
    blueprint.params_and_FLOPs_added = True
    blueprint.n_params = blueprint._count_params()
    blueprint.FLOPs = blueprint._count_FLOPs()
    if not topological_sort:
        return blueprint
    sorted_vertices = blueprint.topological_sorting()
    sorted_blueprint = graph.__class__(search_space=graph.search_space)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_vertices)}
    for new_idx, old_idx in enumerate(sorted_vertices):
        sorted_blueprint.add_vertex()
        for attr in blueprint.vs[old_idx].attribute_names():
            sorted_blueprint.vs[new_idx][attr] = blueprint.vs[old_idx][attr]
    for old_source, old_target in blueprint.get_edgelist():
        new_source, new_target = (old_to_new[old_source], old_to_new[old_target])
        sorted_blueprint.add_edge(new_source, new_target)
    sorted_blueprint.valid = blueprint.valid
    sorted_blueprint.shapes_added = blueprint.shapes_added
    sorted_blueprint.params_and_FLOPs_added = blueprint.params_and_FLOPs_added
    sorted_blueprint.n_params = blueprint.n_params
    sorted_blueprint.FLOPs = blueprint.FLOPs
    if hasattr(blueprint, "BBGP"):
        sorted_blueprint.BBGP = blueprint.BBGP
    return sorted_blueprint
