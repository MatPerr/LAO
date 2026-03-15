from typing import Any


def build_torch_model(
    graph: Any,
    input_shape: Any = (3, 32),
    num_classes: Any = 10,
    enforce_max_preds: Any = False,
    feature_index_fn: Any = None,
) -> Any:
    """
    Build torch model.

    Args:
        graph (Any): Input parameter.
        input_shape (Any): Input parameter.
        num_classes (Any): Input parameter.
        enforce_max_preds (Any): Input parameter.
        feature_index_fn (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    import torch
    import torch.nn as nn

    from lao.candidate_eval.custom_layers import AggTensors, ChannelPad, CustomSE

    blueprint = graph.to_blueprint(
        input_shape=input_shape, num_classes=num_classes, enforce_max_preds=enforce_max_preds
    )

    class ArcGraphModel(nn.Module):
        def __init__(self, blueprint_graph: Any) -> None:
            """
            Init.

            Args:
                blueprint_graph (Any): Input parameter.
            """
            super(ArcGraphModel, self).__init__()
            self.layers = nn.ModuleDict()
            self.topology = []
            execution_order = []
            visited = set()
            self.n_params = blueprint_graph.n_params
            self.FLOPs = blueprint_graph.FLOPs
            self.BBGP = blueprint_graph.BBGP

            def visit(node_idx: Any) -> Any:
                """
                Visit.

                Args:
                    node_idx (Any): Input parameter.

                Returns:
                    Any: Function output.
                """
                if node_idx in visited:
                    return
                visited.add(node_idx)
                for pred in blueprint_graph.predecessors(node_idx):
                    visit(pred)
                execution_order.append(node_idx)

            for i in range(blueprint_graph.vcount()):
                if i not in visited:
                    visit(i)
            for node_idx in execution_order:
                node = blueprint_graph.vs[node_idx]
                node_type = node["node_type"] if "node_type" in node.attribute_names() else "original"
                if node_type == "original":
                    features = node["features"]
                    in_channels, _ = node["input_shape"]
                    out_channels = features[feature_index_fn("out_channels", blueprint_graph.search_space)]
                    kernel_size = features[feature_index_fn("kernel_size", blueprint_graph.search_space)]
                    stride = features[feature_index_fn("stride", blueprint_graph.search_space)]
                    groups = features[feature_index_fn("groups", blueprint_graph.search_space)]
                    if groups == -1:
                        groups = in_channels
                    if len(blueprint_graph.predecessors(node_idx)) > 1:
                        if "aggregation" in blueprint_graph.search_space.node_features.__dict__.keys():
                            aggregation = features[feature_index_fn("aggregation", blueprint_graph.search_space)]
                            self.layers[f"agg_{node_idx}"] = AggTensors(aggregation)
                        else:
                            self.layers[f"agg_{node_idx}"] = AggTensors("sum")
                    self.layers[f"conv_{node_idx}"] = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        groups=groups,
                        bias=True,
                    )
                    self.layers[f"bn_{node_idx}"] = nn.BatchNorm2d(out_channels)
                    self.layers[f"act_{node_idx}"] = nn.ReLU(inplace=True)
                    if "squeeze_excitation" in blueprint_graph.search_space.node_features.__dict__.keys():
                        se = features[feature_index_fn("squeeze_excitation", blueprint_graph.search_space)]
                        if se == 1:
                            self.layers[f"se_{node_idx}"] = CustomSE(out_channels)
                elif node_type == "breadth_adapter":
                    reduc_factor = node["reduc_factor"]
                    self.layers[f"maxpool_{node_idx}"] = nn.MaxPool2d(kernel_size=reduc_factor, stride=reduc_factor)
                elif node_type == "channel_adapter":
                    in_channels, _ = node["input_shape"]
                    out_channels, _ = node["output_shape"]
                    self.layers[f"chpad_{node_idx}"] = ChannelPad(in_channels, out_channels)
                elif node_type == "global_avg_pool":
                    if len(blueprint_graph.predecessors(node_idx)) > 1:
                        self.layers[f"agg_{node_idx}"] = AggTensors("sum")
                    self.layers[f"gap_{node_idx}"] = nn.AdaptiveAvgPool2d(1)
                elif node_type == "classifier":
                    in_features = node["input_shape"][0]
                    out_classes = node["output_shape"][0]
                    self.layers[f"classifier_{node_idx}"] = nn.Linear(in_features, out_classes)
                self.topology.append((node_idx, blueprint_graph.predecessors(node_idx)))
            self.successors = {i: [] for i in range(blueprint_graph.vcount())}
            for node_idx, predecessors in self.topology:
                for pred_idx in predecessors:
                    self.successors[pred_idx].append(node_idx)
            self.output_node = self.topology[-1][0]
            self._blueprint = blueprint_graph

        def forward(self, x: Any) -> Any:
            """
            Forward.

            Args:
                x (Any): Input parameter.

            Returns:
                Any: Function output.
            """
            outputs = {}
            remaining_uses = {i: len(self.successors[i]) for i in range(len(self.topology))}
            for node_idx, predecessors in self.topology:
                node_type = self._blueprint.vs[node_idx]["node_type"]
                if len(predecessors) == 0:
                    if node_idx != 0:
                        raise ValueError(f"Node {node_idx}, but only the first node may have indegree = 0")
                    if node_type != "original":
                        raise ValueError(f"Unexpected node type {node_type} for input node")
                    x = self.layers[f"conv_{node_idx}"](x)
                    x = self.layers[f"bn_{node_idx}"](x)
                    x = self.layers[f"act_{node_idx}"](x)
                    if f"se_{node_idx}" in self.layers:
                        x = self.layers[f"se_{node_idx}"](x)
                    outputs[node_idx] = x
                    continue
                if len(predecessors) == 1:
                    pred_idx = predecessors[0]
                    x = outputs[pred_idx]
                    remaining_uses[pred_idx] -= 1
                    if remaining_uses[pred_idx] == 0 and pred_idx != self.output_node:
                        outputs[pred_idx] = None
                else:
                    pred_outputs = []
                    for pred_idx in predecessors:
                        pred_outputs.append(outputs[pred_idx])
                        remaining_uses[pred_idx] -= 1
                        if remaining_uses[pred_idx] == 0 and pred_idx != self.output_node:
                            outputs[pred_idx] = None
                    if node_type in ("original", "global_avg_pool"):
                        x = self.layers[f"agg_{node_idx}"](torch.stack(pred_outputs))
                    else:
                        raise ValueError(
                            f"{node_type} nodes may not have >1 predecessors, received {len(predecessors)}"
                        )
                    pred_outputs = None
                if node_type == "original":
                    x = self.layers[f"conv_{node_idx}"](x)
                    x = self.layers[f"bn_{node_idx}"](x)
                    x = self.layers[f"act_{node_idx}"](x)
                    if f"se_{node_idx}" in self.layers:
                        x = self.layers[f"se_{node_idx}"](x)
                elif node_type == "breadth_adapter":
                    x = self.layers[f"maxpool_{node_idx}"](x)
                elif node_type == "channel_adapter":
                    x = self.layers[f"chpad_{node_idx}"](x)
                elif node_type == "global_avg_pool":
                    x = self.layers[f"gap_{node_idx}"](x)
                    x = torch.flatten(x, 1)
                elif node_type == "classifier":
                    x = self.layers[f"classifier_{node_idx}"](x)
                outputs[node_idx] = x
            result = outputs[self.output_node]
            outputs.clear()
            return result

    return ArcGraphModel(blueprint)


def build_keras_model(
    graph: Any,
    input_shape: Any = (3, 32),
    num_classes: Any = 10,
    enforce_max_preds: Any = False,
    backend: Any = "torch",
    feature_index_fn: Any = None,
) -> Any:
    """
    Build keras model.

    Args:
        graph (Any): Input parameter.
        input_shape (Any): Input parameter.
        num_classes (Any): Input parameter.
        enforce_max_preds (Any): Input parameter.
        backend (Any): Input parameter.
        feature_index_fn (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    import os

    os.environ["KERAS_BACKEND"] = backend
    import keras
    from keras.models import Model

    class KerasCustomSE(keras.layers.Layer):
        def __init__(self, channels: Any, reduction: Any = 16, **kwargs: Any) -> None:
            """
            Init.

            Args:
                channels (Any): Input parameter.
                reduction (Any): Input parameter.
                **kwargs (Any): Variable keyword arguments.
            """
            super(KerasCustomSE, self).__init__(**kwargs)
            self.channels = channels
            self.global_avg_pool = keras.layers.GlobalAveragePooling2D()
            self.fc1 = keras.layers.Dense(channels // reduction, activation="relu")
            self.fc2 = keras.layers.Dense(channels, activation="sigmoid")

        def call(self, inputs: Any) -> Any:
            """
            Call.

            Args:
                inputs (Any): Input parameter.

            Returns:
                Any: Function output.
            """
            se = self.global_avg_pool(inputs)
            se = self.fc1(se)
            se = self.fc2(se)
            se = se.reshape(se.shape[0], self.channels, 1, 1)
            return inputs * se

    class KerasChannelPad(keras.layers.Layer):
        def __init__(self, in_channels: Any, out_channels: Any, **kwargs: Any) -> None:
            """
            Init.

            Args:
                in_channels (Any): Input parameter.
                out_channels (Any): Input parameter.
                **kwargs (Any): Variable keyword arguments.
            """
            super(KerasChannelPad, self).__init__(**kwargs)
            self.in_channels = in_channels
            self.out_channels = out_channels
            if in_channels > out_channels:
                raise ValueError(
                    f"in_channels (={in_channels}) must be inferior out_channels (={out_channels}) to use channel padding"
                )
            if in_channels == out_channels:
                self.pad_layer = keras.layers.Identity()
            else:
                self.pad_layer = keras.layers.ZeroPadding2D(padding=((0, out_channels - in_channels), (0, 0)))
            self.perm = (0, 2, 1, 3)

        def call(self, inputs: Any) -> Any:
            """
            Call.

            Args:
                inputs (Any): Input parameter.

            Returns:
                Any: Function output.
            """
            output = keras.ops.transpose(inputs, axes=self.perm)
            output = self.pad_layer(output)
            output = keras.ops.transpose(output, axes=self.perm)
            return output

    class KerasAggTensors(keras.layers.Layer):
        def __init__(self, aggregation: Any = "sum", **kwargs: Any) -> None:
            """
            Init.

            Args:
                aggregation (Any): Input parameter.
                **kwargs (Any): Variable keyword arguments.
            """
            super(KerasAggTensors, self).__init__(**kwargs)
            if aggregation != "sum":
                raise NotImplementedError("Only 'sum' aggregation is implemented.")
            self.aggregator = keras.layers.Add()

        def call(self, inputs: Any) -> Any:
            """Call."""
            return self.aggregator(inputs)

    blueprint = graph.to_blueprint(
        input_shape=input_shape, num_classes=num_classes, enforce_max_preds=enforce_max_preds
    )
    visited = set()
    execution_order = []

    def visit(node_idx: Any) -> Any:
        """
        Visit.

        Args:
            node_idx (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        if node_idx in visited:
            return
        visited.add(node_idx)
        for pred in blueprint.predecessors(node_idx):
            visit(pred)
        execution_order.append(node_idx)

    for i in range(blueprint.vcount()):
        visit(i)
    topology = [(node_idx, blueprint.predecessors(node_idx)) for node_idx in execution_order]

    class KerasArcGraphModel(Model):
        def __init__(self, blueprint_graph: Any, model_topology: Any) -> None:
            """
            Init.

            Args:
                blueprint_graph (Any): Input parameter.
                model_topology (Any): Input parameter.
            """
            super(KerasArcGraphModel, self).__init__()
            self.n_params = blueprint_graph.n_params
            self.FLOPs = blueprint_graph.FLOPs
            self.BBGP = blueprint_graph.BBGP
            self.topology = model_topology
            self.layers_dict = {}
            self.agg_layer = KerasAggTensors("sum")
            for node_idx in execution_order:
                node = blueprint_graph.vs[node_idx]
                node_type = node["node_type"] if "node_type" in node.attribute_names() else "original"
                if node_type == "original":
                    features = node["features"]
                    in_channels = node["input_shape"][0]
                    out_channels = features[feature_index_fn("out_channels", blueprint_graph.search_space)]
                    kernel_size = features[feature_index_fn("kernel_size", blueprint_graph.search_space)]
                    stride = features[feature_index_fn("stride", blueprint_graph.search_space)]
                    groups = features[feature_index_fn("groups", blueprint_graph.search_space)]
                    if groups == -1:
                        groups = in_channels
                    block = keras.Sequential()
                    block.add(
                        keras.layers.Conv2D(
                            filters=out_channels,
                            kernel_size=kernel_size,
                            strides=stride,
                            padding="same",
                            groups=groups,
                            use_bias=True,
                        )
                    )
                    block.add(keras.layers.BatchNormalization())
                    block.add(keras.layers.ReLU())
                    if "squeeze_excitation" in blueprint_graph.search_space.node_features.__dict__:
                        se = features[feature_index_fn("squeeze_excitation", blueprint_graph.search_space)]
                        if se == 1:
                            block.add(KerasCustomSE(out_channels))
                    self.layers_dict[node_idx] = block
                elif node_type == "breadth_adapter":
                    reduc_factor = node["reduc_factor"]
                    self.layers_dict[node_idx] = keras.layers.MaxPooling2D(
                        pool_size=(reduc_factor, reduc_factor), strides=reduc_factor, padding="same"
                    )
                elif node_type == "channel_adapter":
                    in_channels = node["input_shape"][0]
                    out_channels = node["output_shape"][0]
                    self.layers_dict[node_idx] = KerasChannelPad(in_channels, out_channels)
                elif node_type == "global_avg_pool":
                    self.layers_dict[node_idx] = keras.layers.GlobalAveragePooling2D()
                elif node_type == "classifier":
                    self.layers_dict[node_idx] = keras.layers.Dense(num_classes)
                else:
                    raise ValueError(f"Unknown node type: {node_type}")

        def call(self, inputs: Any, training: Any = False) -> Any:
            """
            Call.

            Args:
                inputs (Any): Input parameter.
                training (Any): Input parameter.

            Returns:
                Any: Function output.
            """
            outputs = {}
            for node_idx, preds in self.topology:
                if not preds:
                    outputs[node_idx] = self.layers_dict[node_idx](inputs)
                else:
                    pred_outputs = [outputs[p] for p in preds]
                    aggregated = self.agg_layer(pred_outputs) if len(pred_outputs) > 1 else pred_outputs[0]
                    outputs[node_idx] = self.layers_dict[node_idx](aggregated)
            return outputs[self.topology[-1][0]]

    input_tensor = keras.layers.Input(shape=(input_shape[0], input_shape[1], input_shape[1]))
    keras_arc_model = KerasArcGraphModel(blueprint, topology)
    output_tensor = keras_arc_model(input_tensor)
    return keras.Model(inputs=input_tensor, outputs=output_tensor)
