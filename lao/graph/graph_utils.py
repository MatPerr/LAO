from typing import Any

import igraph as ig
import numpy as np
import torch


def constraints_satisfied(metrics_source: Any, constraints: Any, allowed_keys: Any = None) -> Any:
    """
    Constraints satisfied.

    Args:
        metrics_source (Any): Input parameter.
        constraints (Any): Input parameter.
        allowed_keys (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    if constraints is None:
        return True
    if allowed_keys is None:
        allowed_keys = {"n_params", "FLOPs", "BBGP"}
    for key, bounds in constraints.items():
        if key not in allowed_keys:
            raise ValueError(f"Unsupported constraint key: {key}")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Constraint for {key} must be a 2-item [min, max] pair")
        low, high = bounds
        if low > high:
            raise ValueError(f"Constraint for {key} has min > max: {low} > {high}")
        if isinstance(metrics_source, dict):
            if key not in metrics_source:
                raise KeyError(f"Missing metric '{key}' in metrics source")
            metric_value = metrics_source[key]
        else:
            if not hasattr(metrics_source, key):
                raise AttributeError(f"Metrics source has no attribute '{key}'")
            metric_value = getattr(metrics_source, key)
        if metric_value < low or metric_value > high:
            return False
    return True


def one_hot_encode(value: Any, possible_values: Any) -> Any:
    """
    One hot encode.

    Args:
        value (Any): Input parameter.
        possible_values (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    try:
        index = possible_values.index(value)
    except ValueError:
        raise ValueError(f"Value {value} not found in possible values: {possible_values}")
    one_hot = np.zeros(len(possible_values), dtype=int)
    one_hot[index] = 1
    return one_hot


def quantized_encode(value: Any, possible_values: Any) -> Any:
    """
    Quantized encode.

    Args:
        value (Any): Input parameter.
        possible_values (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    try:
        index = possible_values.index(value)
    except ValueError:
        raise ValueError(f"Value {value} not found in possible values: {possible_values}")
    n_bins = len(possible_values)
    if n_bins == 1:
        return np.array([0.5], dtype=float)
    bin_width = 1.0 / n_bins
    bin_center = (index + 0.5) * bin_width
    return np.array([bin_center], dtype=float)


def quantized_decode(quantized_value: Any, possible_values: Any) -> Any:
    """
    Quantized decode.

    Args:
        quantized_value (Any): Input parameter.
        possible_values (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    n_bins = len(possible_values)
    if n_bins == 1:
        return possible_values[0]
    if quantized_value <= 0:
        return possible_values[0]
    if quantized_value >= 1:
        return possible_values[-1]
    bin_width = 1.0 / n_bins
    bin_centers = [(i + 0.5) * bin_width for i in range(n_bins)]
    closest_bin = min(range(n_bins), key=lambda i: abs(bin_centers[i] - quantized_value))
    return possible_values[closest_bin]


def feature_index(feature_name: Any, search_space: Any) -> Any:
    """Feature index."""
    return list(search_space.node_features.__dict__.keys()).index(feature_name)


def format_number_spaces(number: Any) -> Any:
    """
    Format number spaces.

    Args:
        number (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    number_string = str(number)
    formatted_string = ""
    for i, char in enumerate(number_string[::-1]):
        if i > 0 and i % 3 == 0:
            formatted_string += " "
        formatted_string += char
    return formatted_string[::-1]


class ArcGraph(ig.Graph):
    """
    Directed architecture graph used throughout LAO.

    This class stores node-level architecture features, edge connectivity, shape
    propagation metadata, and derived cost statistics (parameters/FLOPs/BBGP). It
    also exposes conversion helpers that compile the graph into downstream
    representations (blueprint, PyTorch, Keras).

    Args:
        search_space (Any): Search-space definition used to interpret and validate
            node features.
        X (Any): Optional node-feature matrix used to initialize graph vertices.
        A (Any): Optional adjacency matrix used to initialize graph edges.
        V (Any): Optional flattened graph vector encoding (features + adjacency).
        n_nodes (Any): Number of nodes to reconstruct when `V` is provided.
    """

    def __init__(
        self, search_space: Any = None, X: Any = None, A: Any = None, V: Any = None, n_nodes: Any = None
    ) -> None:
        """
        Init.

        Args:
            search_space (Any): Input parameter.
            X (Any): Input parameter.
            A (Any): Input parameter.
            V (Any): Input parameter.
            n_nodes (Any): Input parameter.
        """
        super().__init__(directed=True)
        self.search_space = search_space
        if V is None:
            if X is not None:
                self.add_vertices(len(X))
                for i in range(self.vcount()):
                    self.vs[i]["features"] = X[i].tolist()
            if A is not None:
                g = ig.Graph.Adjacency(A.tolist() if isinstance(A, np.ndarray) else A, mode="directed")
                if self.vcount() == 0:
                    self.add_vertices(len(A))
                self.add_edges(g.get_edgelist())
        elif V is not None:
            assert search_space is not None, "search_space must be provided to create a graph from graphvector"
            assert n_nodes is not None, "n_nodes must be provided to create a graph from graphvector"
            V = V.cpu().numpy()
            self.add_vertices(n_nodes)
            n_adj_comp = n_nodes * (n_nodes - 1) // 2
            adj_list = V[-n_adj_comp:]
            A = np.zeros((n_nodes, n_nodes), dtype=int)
            index = 0
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if adj_list[index] != 0:
                        self.add_edge(i, j)
                    index += 1
            node_encoding_type = search_space.graph_features.node_encoding_type
            if node_encoding_type == "categorical":
                X_onehot = V[:-n_adj_comp].reshape(n_nodes, -1)
                X = np.empty((n_nodes, len(search_space.node_features.__dict__)), dtype="object")
                for i in range(n_nodes):
                    shift = 0
                    for j, possible_values in enumerate(search_space.node_features.__dict__.values()):
                        onehot_val = X_onehot[i, shift : shift + len(possible_values)]
                        X[i, j] = possible_values[np.argmax(onehot_val)]
                        shift += len(possible_values)
                    self.vs[i]["features"] = X[i].tolist()
            elif node_encoding_type == "quantized":
                feature_count = len(search_space.node_features.__dict__)
                X_quantized = V[:-n_adj_comp].reshape(n_nodes, feature_count)
                X = np.empty((n_nodes, feature_count), dtype="object")
                for i in range(n_nodes):
                    for j, possible_values in enumerate(search_space.node_features.__dict__.values()):
                        quantized_val = X_quantized[i, j]
                        X[i, j] = quantized_decode(quantized_val, possible_values)
                    self.vs[i]["features"] = X[i].tolist()

    def plot(self, output_path: Any = None, backbone: Any = True, display: Any = False) -> Any:
        """Plot."""
        from lao.graph.graph_viz import plot_graph

        return plot_graph(
            self,
            output_path=output_path,
            backbone=backbone,
            display=display,
            feature_index_fn=feature_index,
            format_number_spaces_fn=format_number_spaces,
        )

    def to_V(self) -> Any:
        """
        To v.

        Args:
            None

        Returns:
            Any: Function output.
        """
        assert self.search_space is not None, "search_space must be provided to convert a graph to graphvector"
        n_nodes = self.vcount()
        V = []
        node_encoding_type = self.search_space.graph_features.node_encoding_type
        for i in range(n_nodes):
            for j, possible_values in enumerate(self.search_space.node_features.__dict__.values()):
                val = self.vs[i]["features"][j]
                if node_encoding_type == "categorical":
                    V += one_hot_encode(val, possible_values).tolist()
                elif node_encoding_type == "quantized":
                    V += quantized_encode(val, possible_values).tolist()
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                V.append(1 if self.are_adjacent(i, j) else 0)
        return torch.Tensor(V)

    def _get_input_shape(self, i: Any) -> Any:
        """
        Get input shape.

        Args:
            i (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        all_input_shapes = np.array([self.vs[x]["output_shape"] for x in self.predecessors(i)])
        input_channels = all_input_shapes[:, 0].max()
        input_breadth = all_input_shapes[:, 1].min()
        return [input_channels, input_breadth]

    def _get_output_shape(self, i: Any) -> Any:
        """
        Get output shape.

        Args:
            i (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        all_input_shapes = np.array([self.vs[x]["output_shape"] for x in self.predecessors(i)])
        input_breadth = all_input_shapes[:, 1].min()
        node_features = self.vs[i]["features"]
        stride = node_features[feature_index("stride", self.search_space)]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        return [out_channels, input_breadth // stride]

    def add_latent_shapes(self, input_shape: Any) -> Any:
        """
        Add latent shapes.

        Args:
            input_shape (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        assert self.search_space is not None, "search_space must be provided to propagate input shape"
        assert self.valid, "the graph must be valid to propagate input shape"
        self.vs[0]["input_shape"] = input_shape
        node_features = self.vs[0]["features"]
        stride = node_features[feature_index("stride", self.search_space)]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        input_breadth = input_shape[1]
        self.vs[0]["output_shape"] = [out_channels, input_breadth // stride]
        for i in range(1, self.vcount()):
            self.vs[i]["input_shape"] = self._get_input_shape(i)
            self.vs[i]["output_shape"] = self._get_output_shape(i)
        self.BBGP = self.vs[-1]["output_shape"][1]
        self.shapes_added = True

    def add_n_params_and_FlOPs(self) -> Any:
        """
        Add n params and flops.

        Args:
            None

        Returns:
            Any: Function output.
        """
        assert self.search_space is not None, "search_space must be provided to count node params"
        assert self.valid, "the graph must be valid to count node params"
        assert self.shapes_added, "the latent shapes must be added to count node params"
        for i in range(self.vcount()):
            self._add_node_n_params_and_FLOPs(i)
        self.params_and_FLOPs_added = True
        self.n_params = self._count_params()
        self.FLOPs = self._count_FLOPs()

    def _add_node_n_params_and_FLOPs(self, node_idx: Any) -> Any:
        """
        Add node n params and flops.

        Args:
            node_idx (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        node = self.vs[node_idx]
        input_channels, input_breadth = node["input_shape"]
        node_features = node["features"]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        kernel_size = node_features[feature_index("kernel_size", self.search_space)]
        stride = node_features[feature_index("stride", self.search_space)]
        groups = node_features[feature_index("groups", self.search_space)]
        if groups == -1:
            groups = input_channels
        node["n_params"] = self._calculate_node_params(node_features, input_channels, out_channels, kernel_size, groups)
        node["FLOPs"] = self._calculate_node_FLOPs(
            node, node_features, input_channels, input_breadth, out_channels, kernel_size, stride, groups
        )

    def _calculate_node_params(
        self, node_features: Any, input_channels: Any, out_channels: Any, kernel_size: Any, groups: Any
    ) -> Any:
        """
        Calculate node params.

        Args:
            node_features (Any): Input parameter.
            input_channels (Any): Input parameter.
            out_channels (Any): Input parameter.
            kernel_size (Any): Input parameter.
            groups (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        conv_params = out_channels * (kernel_size**2 * (input_channels // groups) + 1)
        bn_params = 2 * out_channels
        n_params = conv_params + bn_params
        if "squeeze_excitation" in self.search_space.node_features.__dict__.keys():
            if node_features[feature_index("squeeze_excitation", self.search_space)] == 1:
                se_reduce = out_channels // 16
                n_params += se_reduce * (out_channels + 1) + out_channels * (se_reduce + 1)
        return n_params

    def _calculate_node_FLOPs(
        self,
        node: Any,
        node_features: Any,
        input_channels: Any,
        input_breadth: Any,
        out_channels: Any,
        kernel_size: Any,
        stride: Any,
        groups: Any,
    ) -> Any:
        """
        Calculate node flops.

        Args:
            node (Any): Input parameter.
            node_features (Any): Input parameter.
            input_channels (Any): Input parameter.
            input_breadth (Any): Input parameter.
            out_channels (Any): Input parameter.
            kernel_size (Any): Input parameter.
            stride (Any): Input parameter.
            groups (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        output_breadth = input_breadth // stride
        conv_FLOPs = 2 * input_breadth**2 * kernel_size**2 * input_channels * out_channels // (stride**2 * groups)
        bn_FLOPs = 2 * out_channels * output_breadth**2
        act_flops = 2 * out_channels * output_breadth**2
        FLOPs = conv_FLOPs + bn_FLOPs + act_flops
        if "squeeze_excitation" in self.search_space.node_features.__dict__.keys():
            if node_features[feature_index("squeeze_excitation", self.search_space)] == 1:
                FLOPs += self._calculate_se_FLOPs(out_channels, output_breadth)
        FLOPs += self._calculate_aggregation_FLOPs(node, input_channels, input_breadth)
        return FLOPs

    def _calculate_se_FLOPs(self, out_channels: Any, output_breadth: Any) -> Any:
        """
        Calculate se flops.

        Args:
            out_channels (Any): Input parameter.
            output_breadth (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        se_reduce = out_channels // 16
        FLOPs = 0
        FLOPs += out_channels * output_breadth**2
        FLOPs += 2 * out_channels * se_reduce
        FLOPs += 2 * se_reduce
        FLOPs += 2 * se_reduce * out_channels + out_channels
        FLOPs += 4 * out_channels
        FLOPs += out_channels * output_breadth**2
        return FLOPs

    def _calculate_aggregation_FLOPs(self, node: Any, input_channels: Any, input_breadth: Any) -> Any:
        """
        Calculate aggregation flops.

        Args:
            node (Any): Input parameter.
            input_channels (Any): Input parameter.
            input_breadth (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        if "aggregation" in self.search_space.node_features.__dict__.keys():
            agg_type = node["features"][feature_index("aggregation", self.search_space)]
            if agg_type != "sum":
                raise NotImplementedError(
                    f"Aggregation type {agg_type} not implemented, only 'sum' is supported for now"
                )
        predecessors = self.predecessors(node.index)
        if len(predecessors) > 1:
            input_elements = input_channels * input_breadth**2
            agg_flops = (len(predecessors) - 1) * input_elements
            return agg_flops
        else:
            return 0

    def _count_params(self) -> Any:
        """Count params."""
        assert self.params_and_FLOPs_added, "n_params must be added to count total params"
        return np.sum([node["n_params"] for node in self.vs if node["n_params"] is not None])

    def _count_FLOPs(self) -> Any:
        """Count flops."""
        assert self.params_and_FLOPs_added, "FLOPs must be added to count total FLOPs"
        return np.sum([node["FLOPs"] for node in self.vs if node["FLOPs"] is not None])

    def _is_node_valid(self, i: Any, verbose: Any) -> Any:
        """
        Is node valid.

        Args:
            i (Any): Input parameter.
            verbose (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        node_features = self.vs[i]["features"]
        stride = node_features[feature_index("stride", self.search_space)]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        groups = node_features[feature_index("groups", self.search_space)]
        input_shape = self.vs[i]["input_shape"]
        input_channels, input_breadth = (input_shape[0], input_shape[1])
        if input_breadth % stride != 0:
            if verbose:
                print(
                    f"The architecture is not valid because node {i} has input breadth = {input_breadth} and the stride = {stride}, must be divisible"
                )
            return False
        if input_channels % groups != 0:
            if verbose:
                print(
                    f"The architecture is not valid because node {i} has input channels = {input_channels} and the groups = {groups}, must be divisible"
                )
            return False
        if out_channels % groups != 0:
            if verbose:
                print(
                    f"The architecture is not valid because node {i} has output channels = {out_channels} and the groups = {groups}, must be divisible"
                )
            return False
        return True

    def is_valid(self, input_shape: Any, verbose: Any = False) -> Any:
        """
        Is valid.

        Args:
            input_shape (Any): Input parameter.
            verbose (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        assert self.search_space is not None, "search_space must be provided to test graph validity"
        if self.vcount() < 1:
            if verbose:
                print(f"The architecture is not valid because it has {self.vcount()} nodes, expected at least 1")
            return False
        if self.search_space.graph_features.traceable:
            A = np.array(self.get_adjacency().data)
            if not np.all(np.diag(A, k=1) == 1):
                if verbose:
                    print("The architecture is not valid because the graph is not traceable")
                return False
            loose_end_vertices = set([v.index for v in self.vs.select(_outdegree_eq=0)])
            if len(loose_end_vertices) > 1:
                if verbose:
                    print(
                        f"The architecture is not valid because it has {loose_end_vertices} outdegree = 0 vertices, expected at most 1"
                    )
                return False
        loose_start_vertices = set([v.index for v in self.vs.select(_indegree_eq=0)])
        if len(loose_start_vertices) > 1:
            if verbose:
                print(
                    f"The architecture is not valid because it has {loose_start_vertices} indegree = 0 vertices, expected at most 1"
                )
            return False
        self.vs[0]["input_shape"] = input_shape
        if not self._is_node_valid(0, verbose):
            return False
        node_features = self.vs[0]["features"]
        stride = node_features[feature_index("stride", self.search_space)]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        input_breadth = input_shape[1]
        self.vs[0]["output_shape"] = [out_channels, input_breadth // stride]
        for i in range(1, self.vcount()):
            self.vs[i]["input_shape"] = self._get_input_shape(i)
            if not self._is_node_valid(i, verbose):
                return False
            self.vs[i]["output_shape"] = self._get_output_shape(i)
        self.valid = True
        return True

    def sample_node_features(self, input_shape: Any, constraints: Any = None) -> Any:
        """Sample node features."""
        from lao.graph.graph_sampling import sample_node_features

        return sample_node_features(
            self,
            input_shape=input_shape,
            constraints=constraints,
            feature_index_fn=feature_index,
            constraints_satisfied_fn=constraints_satisfied,
        )

    def _make_node_valid(self, graph: Any, node_idx: Any, input_shape: Any) -> Any:
        """Make node valid."""
        from lao.graph.graph_sampling import make_node_valid

        return make_node_valid(graph, node_idx=node_idx, input_shape=input_shape, feature_index_fn=feature_index)

    def to_blueprint(
        self,
        input_shape: Any = [3, 32],
        num_classes: Any = 10,
        enforce_max_preds: Any = False,
        topological_sort: Any = False,
    ) -> Any:
        """To blueprint."""
        from lao.graph.graph_compiler import to_blueprint

        return to_blueprint(
            self,
            input_shape=input_shape,
            num_classes=num_classes,
            enforce_max_preds=enforce_max_preds,
            topological_sort=topological_sort,
            feature_index_fn=feature_index,
            make_node_valid_fn=self._make_node_valid,
        )

    def _layer_blueprint(self) -> Any:
        """Layer blueprint."""
        from lao.graph.graph_compiler import layer_blueprint

        return layer_blueprint(self)

    def to_torch(self, input_shape: Any = [3, 32], num_classes: Any = 10, enforce_max_preds: Any = False) -> Any:
        """To torch."""
        from lao.graph.graph_builders import build_torch_model

        return build_torch_model(
            self,
            input_shape=input_shape,
            num_classes=num_classes,
            enforce_max_preds=enforce_max_preds,
            feature_index_fn=feature_index,
        )

    def to_keras(
        self, input_shape: Any = (3, 32), num_classes: Any = 10, enforce_max_preds: Any = False, backend: Any = "torch"
    ) -> Any:
        """To keras."""
        from lao.graph.graph_builders import build_keras_model

        return build_keras_model(
            self,
            input_shape=input_shape,
            num_classes=num_classes,
            enforce_max_preds=enforce_max_preds,
            backend=backend,
            feature_index_fn=feature_index,
        )

    def memory_usage_animation(
        self, output_path: Any = None, delays: Any = 50, backbone: Any = True, plot_memory: Any = True
    ) -> Any:
        """Memory usage animation."""
        from lao.graph.graph_viz import memory_usage_animation

        return memory_usage_animation(
            self,
            output_path=output_path,
            delays=delays,
            backbone=backbone,
            plot_memory=plot_memory,
            feature_index_fn=feature_index,
            format_number_spaces_fn=format_number_spaces,
        )
