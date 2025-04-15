import numpy as np
import igraph as ig
import pygraphviz as pgv
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import tempfile

from search_space import *


def one_hot_encode(value, possible_values):
    try:
        index = possible_values.index(value)
    except ValueError:
        raise ValueError(f"Value {value} not found in possible values: {possible_values}")
    one_hot = np.zeros(len(possible_values), dtype=int)
    one_hot[index] = 1
    
    return one_hot


def quantized_encode(value, possible_values):
    try:
        index = possible_values.index(value)
    except ValueError:
        raise ValueError(f"Value {value} not found in possible values: {possible_values}")
    
    n_bins = len(possible_values)
    if n_bins == 1:
        return np.array([0.5], dtype=float)
    bin_width = 1.0 / n_bins
    
    # Calculate bin center (middle point of the bin)
    # For example, with 4 bins, bin centers would be at 0.125, 0.375, 0.625, 0.875
    bin_center = (index + 0.5) * bin_width
    
    return np.array([bin_center], dtype=float)


def quantized_decode(quantized_value, possible_values):
    """
    Decode a quantized value back to the closest value in possible_values.
    Handles values outside the [0,1] range.
    
    Arguments:
        quantized_value: A float value (can be outside the [0, 1] range)
        possible_values: List of all possible values for this feature
    
    Returns:
        The decoded feature value from possible_values
    """
    n_bins = len(possible_values)
    if n_bins == 1:
        return possible_values[0]  # Special case for single value
    
    # Handle values outside [0,1]
    if quantized_value <= 0:
        return possible_values[0]
    if quantized_value >= 1:
        return possible_values[-1]
    
    # Calculate bin width and the bin centers
    bin_width = 1.0 / n_bins
    bin_centers = [(i + 0.5) * bin_width for i in range(n_bins)]
    
    # Find the closest bin center
    closest_bin = min(range(n_bins), key=lambda i: abs(bin_centers[i] - quantized_value))
    
    return possible_values[closest_bin]


def feature_index(feature_name, search_space):
    return list(search_space.node_features.__dict__.keys()).index(feature_name)


def format_number_spaces(number):
    number_string = str(number)
    formatted_string = ""
    for i, char in enumerate(number_string[::-1]):
        if i > 0 and i % 3 == 0:
            formatted_string += " "
        formatted_string += char
    
    return formatted_string[::-1]


class ArcGraph(ig.Graph):
    def __init__(self, search_space=None, X=None, A=None, V=None, n_nodes=None):
        super().__init__(directed=True)

        self.search_space = search_space

        if V is None:
            if X is not None:
                self.add_vertices(len(X))
                for i in range(self.vcount()):
                    self.vs[i]["features"] = X[i].tolist()
            if A is not None:
                g = ig.Graph.Adjacency(A.tolist() if isinstance(A, np.ndarray) else A, mode="directed")
                if self.vcount() == 0: # In case only A is provided
                    self.add_vertices(len(A))
                self.add_edges(g.get_edgelist())

        elif V is not None:
            assert search_space is not None, "search_space must be provided to create a graph from graphvector"
            assert n_nodes is not None, "n_nodes must be provided to create a graph from graphvector"
            V = V.cpu().numpy()
            self.add_vertices(n_nodes)

            # The adjacency matrix is strictly upper triangular
            n_adj_comp = (n_nodes*(n_nodes - 1)) // 2
            adj_list = V[-n_adj_comp:]
            A = np.zeros((n_nodes, n_nodes), dtype=int)
            index = 0
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if adj_list[index] != 0:
                        self.add_edge(i, j)
                    index += 1

            # Process node features
            node_encoding_type = search_space.graph_features.node_encoding_type
            if node_encoding_type == "categorical":
                X_onehot = V[:-n_adj_comp].reshape(n_nodes, -1)
                X = np.empty((n_nodes, len(search_space.node_features.__dict__)), dtype = 'object')
                for i in range(n_nodes):
                    shift = 0
                    for j, possible_values in enumerate(search_space.node_features.__dict__.values()):
                        onehot_val = X_onehot[i, shift:shift + len(possible_values)]
                        X[i, j] = possible_values[np.argmax(onehot_val)]
                        shift += len(possible_values)
                    self.vs[i]["features"] = X[i].tolist()
            elif node_encoding_type == "quantized":
                feature_count = len(search_space.node_features.__dict__)
                X_quantized = V[:-n_adj_comp].reshape(n_nodes, feature_count)
                X = np.empty((n_nodes, feature_count), dtype='object')
                for i in range(n_nodes):
                    for j, possible_values in enumerate(search_space.node_features.__dict__.values()):
                        quantized_val = X_quantized[i, j]
                        X[i, j] = quantized_decode(quantized_val, possible_values)
                    self.vs[i]["features"] = X[i].tolist()
    

    def plot(self, output_path=None, backbone=True, display=False):
        graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
        cmap = plt.get_cmap('cool')
        
        for idx in range(self.vcount()):
            # Use attribute check rather than get method
            node_type = "original"
            if "node_type" in self.vs[idx].attribute_names():
                node_type = self.vs[idx]["node_type"]
            
            if node_type == "original" and "features" in self.vs[idx].attribute_names():
                features = self.vs[idx]["features"]
                if self.search_space and hasattr(self.search_space, "aliases"):
                    label_parts = []
                    feature_names = list(self.search_space.node_features.__dict__.keys())              
                    for i, name in enumerate(feature_names):
                        if i < len(features):
                            alias = self.search_space.aliases.get(name, name)
                            label_parts.append(f"{alias}{features[i]}")
                    
                    label = f"({idx})" + "\n" + " ".join(label_parts)
                else:
                    label = f"({idx})\n{features}"
                    
                color = "goldenrod"
                fixedsize = False
                width = None
                height = None
                stride = features[feature_index("stride", self.search_space)]
                shape = 'box'
                if stride > 1:
                    fixedsize = True
                    width = 2.5
                    height = 0.9
                    shape = 'invtrapezium'
                if "n_params" in self.vs[idx].attribute_names():
                    n_params = self.vs[idx]["n_params"]
                    norm = mcolors.LogNorm(vmin=1, vmax=1_000_000) 
                    color = mcolors.to_hex(cmap(norm(n_params)))
                    label += f"\nparams: {format_number_spaces(n_params)}"
                if "FLOPs" in self.vs[idx].attribute_names():
                    FLOPs = self.vs[idx]["FLOPs"]
                    label += f"\nFLOPs: {format_number_spaces(FLOPs)}"
                
                    
                graph.add_node(
                    idx, 
                    label=label, 
                    color='black', 
                    fillcolor=color, 
                    shape=shape,
                    style='filled',
                    fixedsize=fixedsize,
                    width=width,
                    height=height,
                    fontsize=12
                )
            elif node_type == "breadth_adapter":
                reduc_factor = "?"
                if "reduc_factor" in self.vs[idx].attribute_names():
                    reduc_factor = self.vs[idx]["reduc_factor"]
                label = f"({idx})\nMaxPool {reduc_factor}"
                graph.add_node(
                    idx, 
                    label=label, 
                    color='black', 
                    fillcolor='grey85', 
                    shape='invtrapezium',
                    style='filled',
                    width=1.4, 
                    height=0.5,
                    fixedsize=True,
                    fontsize=12
                )
            elif node_type == "channel_adapter":
                padding = "?"
                if "padding" in self.vs[idx].attribute_names():
                    padding = self.vs[idx]["padding"]
                
                label = f"({idx})\nChannelPad {padding}"
                graph.add_node(
                    idx, 
                    label=label, 
                    color='black', 
                    fillcolor='grey85', 
                    shape='box',
                    style='filled', 
                    fontsize=12
                )
            elif node_type == "global_avg_pool":
                label = f"({idx})\nGlobalAvgPool"
                if "FLOPs" in self.vs[idx].attribute_names():
                    FLOPs = self.vs[idx]["FLOPs"]
                    label += f"\nFLOPs: {format_number_spaces(FLOPs)}"
                graph.add_node(
                    idx, 
                    label=label, 
                    color='black', 
                    fillcolor='grey85', 
                    shape='invtrapezium',
                    style='filled',
                    width=2, 
                    height=0.7,
                    fixedsize=True,
                    fontsize=12
                )
            elif node_type == "classifier":
                label = f"({idx})\nLinear"
                if "n_params" in self.vs[idx].attribute_names():
                    n_params = self.vs[idx]["n_params"]
                    label += f"\nparams: {format_number_spaces(n_params)}"
                if "FLOPs" in self.vs[idx].attribute_names():
                    FLOPs = self.vs[idx]["FLOPs"]
                    label += f"\nFLOPs: {format_number_spaces(FLOPs)}"
                graph.add_node(
                    idx, 
                    label=label, 
                    color='black', 
                    fillcolor='grey85', 
                    shape='box',
                    style='filled', 
                    fontsize=12
                )
            else:
                label = f"({idx})"                
                graph.add_node(
                    idx, 
                    label=label, 
                    color='black', 
                    fillcolor='white', 
                    shape='box',
                    style='filled', 
                    fontsize=12
                )
        
        # Add edges with latent size labels
        for edge in self.get_edgelist():
            source, target = edge
            
            # Add both input and output shapes on edge labels
            input_shape_label = ""
            output_shape_label = ""
            
            # Get the output shape of the source node (edge start)
            if "output_shape" in self.vs[source].attribute_names():
                source_output = self.vs[source]["output_shape"]
                output_shape_label = f" {source_output[0]}×{source_output[1]}²"
            
            # Get the input shape of the target node (edge end)
            if "input_shape" in self.vs[target].attribute_names():
                target_input = self.vs[target]["input_shape"]
                input_shape_label = f" {target_input[0]}×{target_input[1]}²"
            
            # Combine labels if both exist
            if output_shape_label and input_shape_label:
                if output_shape_label == input_shape_label:
                    # If they're the same, just use one
                    label = output_shape_label
                else:
                    # If different, show both
                    label = f" {output_shape_label}\n{input_shape_label}"
            elif output_shape_label:
                label = output_shape_label
            elif input_shape_label:
                label = input_shape_label
            else:
                label = ""
            
            edge_weight = 1
            if backbone and source == target - 1:
                edge_weight = 3
            
            if label:
                graph.add_edge(
                    source, 
                    target, 
                    weight=edge_weight,
                    label=label,
                    fontsize=12,
                    labeldistance=1.5,
                    labelangle=0
                )
            else:
                graph.add_edge(source, target, weight=edge_weight)
        
        graph.layout(prog='dot')
        
        if display:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
            graph.draw(temp_path, prog='dot', args='-Gdpi=300')
            img = mpimg.imread(temp_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            try:
                os.remove(temp_path)
            except:
                pass
            
            return None
            
        elif output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            graph.draw(output_path, prog='dot', args='-Gdpi=300')
            return output_path
        else:
            return graph

    def to_V(self):
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
    
    def _get_input_shape(self, i):
        all_input_shapes = np.array([self.vs[x]['output_shape'] for x in self.predecessors(i)])
        # The channels are the maximum channels among predecessing latents
        input_channels = all_input_shapes[:,0].max()
        # The breadth is the smallest breadth among predecessing latents. The larger ones are downsampled
        input_breadth = all_input_shapes[:,1].min()

        return [input_channels, input_breadth]
    
    def _get_output_shape(self, i):
        all_input_shapes = np.array([self.vs[x]['output_shape'] for x in self.predecessors(i)])
        # The breadth is the smallest breadth among predecessing latents. The larger ones are downsampled
        input_breadth = all_input_shapes[:,1].min()
        node_features = self.vs[i]["features"]
        stride = node_features[feature_index("stride", self.search_space)]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        return [out_channels, input_breadth // stride] 

    def add_latent_shapes(self, input_shape):
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

        # BBGP is the Breadth Before Global (average) Pooling
        self.BBGP = self.vs[-1]["output_shape"][1]
        self.shapes_added = True

    def add_n_params_and_FlOPs(self):
        """Calculate and add parameters and FLOPs for each node in the graph."""
        assert self.search_space is not None, "search_space must be provided to count node params"
        assert self.valid, "the graph must be valid to count node params"
        assert self.shapes_added, "the latent shapes must be added to count node params"
        
        for i in range(self.vcount()):
            self._add_node_n_params_and_FLOPs(i)
        
        self.params_and_FLOPs_added = True
        self.n_params = self._count_params()
        self.FLOPs = self._count_FLOPs()

    def _add_node_n_params_and_FLOPs(self, node_idx):
        """Calculate and add parameters and FLOPs for a single node."""
        node = self.vs[node_idx]
        input_channels, input_breadth = node['input_shape']
        node_features = node["features"]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        kernel_size = node_features[feature_index("kernel_size", self.search_space)]
        stride = node_features[feature_index("stride", self.search_space)]
        groups = node_features[feature_index("groups", self.search_space)]
        
        if groups == -1:
            groups = input_channels
        
        node['n_params'] = self._calculate_node_params(
            node_features, input_channels, out_channels, kernel_size, groups
        )
        
        node['FLOPs'] = self._calculate_node_FLOPs(
            node, node_features, input_channels, input_breadth, out_channels, 
            kernel_size, stride, groups
        )

    def _calculate_node_params(self, node_features, input_channels, out_channels, kernel_size, groups):
        conv_params = out_channels * ((kernel_size**2) * (input_channels // groups) + 1)
        bn_params = 2 * out_channels
        n_params = conv_params + bn_params
        
        if "squeeze_excitation" in self.search_space.node_features.__dict__.keys():
            if node_features[feature_index("squeeze_excitation", self.search_space)] == 1:
                se_reduce = out_channels // 16
                # First FC: (out_channels//16) * (out_channels + bias)
                # Second FC: out_channels * (out_channels//16 + bias)
                n_params += se_reduce * (out_channels + 1) + out_channels * (se_reduce + 1)
        
        return n_params

    def _calculate_node_FLOPs(self, node, node_features, input_channels, input_breadth, 
                            out_channels, kernel_size, stride, groups):
        output_breadth = input_breadth // stride
        conv_FLOPs = (2 * (input_breadth**2) * (kernel_size**2) * input_channels * out_channels) // ((stride**2) * groups)     
        bn_FLOPs = 2 * out_channels * output_breadth**2
        # ReLU FlOPs
        act_flops = 2 * out_channels * output_breadth**2
        FLOPs = conv_FLOPs + bn_FLOPs + act_flops   
        if "squeeze_excitation" in self.search_space.node_features.__dict__.keys():
            if node_features[feature_index("squeeze_excitation", self.search_space)] == 1:
                FLOPs += self._calculate_se_FLOPs(out_channels, output_breadth)
        
        FLOPs += self._calculate_aggregation_FLOPs(node, input_channels, input_breadth)
        
        return FLOPs

    def _calculate_se_FLOPs(self, out_channels, output_breadth):
        se_reduce = out_channels // 16
        FLOPs = 0
        
        # Global average pooling
        FLOPs += out_channels * output_breadth**2
        
        # First FC
        FLOPs += 2 * out_channels * se_reduce
        
        # ReLU
        FLOPs += 2 * se_reduce
        
        # Second FC
        # NOTE: "+ out_channels" is a quickfix to be consistent with deepspeed
        FLOPs += 2 * se_reduce * out_channels + out_channels
        
        # Sigmoid: 1 FLOP for negation, 1 for exponentiation, 1 for addition, 1 for division
        FLOPs += 4 * out_channels
        
        # Pointwise multiplication
        FLOPs += out_channels * output_breadth**2
        
        return FLOPs

    def _calculate_aggregation_FLOPs(self, node, input_channels, input_breadth):
        if "aggregation" in self.search_space.node_features.__dict__.keys():
            agg_type = node["features"][feature_index("aggregation", self.search_space)]
            if agg_type != "sum":
                raise NotImplementedError(
                    f"Aggregation type {agg_type} not implemented, only 'sum' is supported for now"
                )
        
        # Calculate aggregation FLOPs for nodes with multiple predecessors
        predecessors = self.predecessors(node.index)
        if len(predecessors) > 1:
            input_elements = input_channels * (input_breadth**2)
            # (n-1) additions for n inputs
            agg_flops = (len(predecessors) - 1) * input_elements
            return agg_flops
        
        else:
            return 0
    
    def _count_params(self):
        assert self.params_and_FLOPs_added, "n_params must be added to count total params"
        return np.sum([node['n_params'] for node in self.vs if node['n_params'] is not None])
    
    def _count_FLOPs(self):
        assert self.params_and_FLOPs_added, "FLOPs must be added to count total FLOPs"
        return np.sum([node['FLOPs'] for node in self.vs if node['FLOPs'] is not None])
        
    def _is_node_valid(self, i, verbose):
        node_features = self.vs[i]["features"]
        stride = node_features[feature_index("stride", self.search_space)]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        groups = node_features[feature_index("groups", self.search_space)]
        input_shape = self.vs[i]["input_shape"]
        input_channels, input_breadth = input_shape[0], input_shape[1]

        if input_breadth % stride != 0:
            if verbose:
                print(f'The architecture is not valid because node {i} has input breadth = {input_breadth} and the stride = {stride}, must be divisible')
            return False
        if input_channels % groups != 0:
            if verbose:
                print(f'The architecture is not valid because node {i} has input channels = {input_channels} and the groups = {groups}, must be divisible')
            return False
        if out_channels % groups != 0:
            if verbose:
                print(f'The architecture is not valid because node {i} has output channels = {out_channels} and the groups = {groups}, must be divisible')
            return False

        return True
    
    def is_valid(self, input_shape, verbose = False):
        assert self.search_space is not None, "search_space must be provided to test graph validity"
        if self.vcount() < 1:
            if verbose:
                print(f'The architecture is not valid because it has {g.vs.count()} nodes, expected at least 1')
            return False
        
        if self.search_space.graph_features.traceable:
            A = np.array(self.get_adjacency().data)
            if not np.all(np.diag(A, k = 1) == 1):
                if verbose:
                    print(f'The architecture is not valid because the graph is not traceable')
                return False
            
            loose_end_vertices = set([v.index for v in self.vs.select(_outdegree_eq=0)])
            if len(loose_end_vertices) > 1:
                if verbose:
                    print(f'The architecture is not valid because it has {loose_end_vertices} outdegree = 0 vertices, expected at most 1')
                return False
            
        loose_start_vertices = set([v.index for v in self.vs.select(_indegree_eq=0)])
        if len(loose_start_vertices) > 1:
            if verbose:
                print(f'The architecture is not valid because it has {loose_start_vertices} indegree = 0 vertices, expected at most 1')
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
    
    def sample_node_features(self, input_shape, constraints = None):
        n_nodes = len(self.vs)
        
        def sample_for_node(node_idx, input_shape):
            self.vs[node_idx]["input_shape"] = input_shape
            if self.search_space.node_feature_probs.out_channels == "uniform":
                out_channels = np.random.choice(self.search_space.node_features.out_channels)
            elif type(self.search_space.node_feature_probs.out_channels) == list:
                out_channels = np.random.choice(self.search_space.node_features.out_channels, p=self.search_space.node_feature_probs.out_channels)
            
            if self.search_space.node_feature_probs.stride == "uniform":
                possible_strides = [s for s in self.search_space.node_features.stride if input_shape[1] % s == 0]
                stride = np.random.choice(possible_strides)
            elif type(self.search_space.node_feature_probs.stride) == list:
                all_strides = self.search_space.node_features.stride
                possible_strides_idx = [all_strides.index(s) for s in all_strides if input_shape[1] % s == 0]
                possible_strides = [all_strides[i] for i in possible_strides_idx]
                possible_strides_weights = [self.search_space.node_feature_probs.stride[i] for i in possible_strides_idx]
                possible_strides_probs = np.array(possible_strides_weights) / np.sum(possible_strides_weights)
                stride = np.random.choice(possible_strides, p=possible_strides_probs)
            
            if self.search_space.node_feature_probs.groups == "uniform":
                possible_groups = [g for g in self.search_space.node_features.groups if input_shape[0] % g == 0 and out_channels % g == 0]
                groups = np.random.choice(possible_groups)
            elif type(self.search_space.node_feature_probs.groups) == list:
                all_groups = self.search_space.node_features.groups
                possible_groups_idx = [all_groups.index(g) for g in all_groups if input_shape[0] % g == 0 and out_channels % g == 0]
                possible_groups = [all_groups[i] for i in possible_groups_idx]
                possible_groups_weights = [self.search_space.node_feature_probs.groups[i] for i in possible_groups_idx]
                possible_groups_probs = np.array(possible_groups_weights) / np.sum(possible_groups_weights)
                groups = np.random.choice(possible_groups, p=possible_groups_probs)
            
            feature_dict = self.search_space.node_features.__dict__
            node_features = np.empty(len(feature_dict), dtype='object')
            
            for feature_name, feature_values in feature_dict.items():
                index = feature_index(feature_name, self.search_space)
                if feature_name == "out_channels":
                    node_features[index] = out_channels
                elif feature_name == "stride":
                    node_features[index] = stride
                elif feature_name == "groups":
                    node_features[index] = groups
                else:
                    if self.search_space.node_feature_probs.__dict__[feature_name] == "uniform":
                        node_features[index] = np.random.choice(feature_values)
                    elif type(self.search_space.node_feature_probs.__dict__[feature_name]) == list: 
                        node_features[index] = np.random.choice(feature_values, p=self.search_space.node_feature_probs.__dict__[feature_name])            
            self.vs[node_idx]["features"] = node_features.tolist()
            self.vs[node_idx]["output_shape"] = [out_channels, input_shape[1] // stride]

        self.constraints_met = True        
        n_params = 0
        FLOPs = 0

        sample_for_node(0, input_shape)
        self._add_node_n_params_and_FLOPs(0)

        n_params += self.vs[0]['n_params']
        FLOPs += self.vs[0]['FLOPs']
        if constraints is not None:
            if not all([eval(f"{key} <= {value[1]} and {key} >= {value[0]}") for key, value in constraints.items()]):
                self.constraints_met = False

        for i in range(1, n_nodes):
            current_input_shape = self._get_input_shape(i)
            sample_for_node(i, current_input_shape)
            self._add_node_n_params_and_FLOPs(i)

            n_params += self.vs[i]['n_params']
            FLOPs += self.vs[i]['FLOPs']
            if constraints is not None:
                if not all([eval(f"{key} <= {value[1]} and {key} >= {value[0]}") for key, value in constraints.items()]):
                    self.constraints_met = False
                    break


    def _make_node_valid(self, graph, node_idx, input_shape):
        """
        Adjust node features to make them valid based on input shape
        """
        assert "features" in graph.vs[node_idx].attribute_names() and graph.vs[node_idx]["features"] is not None, "Node must have features to make it valid"
        features = graph.vs[node_idx]["features"]
        
        input_channels, input_breadth = input_shape
        
        # Fix stride to be divisible by input_breadth
        stride_idx = feature_index("stride", self.search_space)
        stride_values = self.search_space.node_features.stride
        valid_strides = [s for s in stride_values if input_breadth % s == 0]
        if not valid_strides:
            features[stride_idx] = 1
        elif features[stride_idx] not in valid_strides:
            features[stride_idx] = min(valid_strides, key=lambda s: abs(s - features[stride_idx]))
        
        # Fix groups and out_channels compatibility
        out_channels_idx = feature_index("out_channels", self.search_space)
        groups_idx = feature_index("groups", self.search_space)
        out_channels = features[out_channels_idx]
        groups = features[groups_idx]
        
        # Handle depthwise convolution (groups = -1)
        if groups == -1:
            # Attempt to adjust out_channels to be a multiple of input_channels
            valid_out_channels = [oc for oc in self.search_space.node_features.out_channels 
                                if oc % input_channels == 0]
            if valid_out_channels:
                features[out_channels_idx] = min(valid_out_channels, 
                                            key=lambda oc: abs(oc - out_channels))
                # Set the actual number of groups rather than keeping -1
                features[groups_idx] = input_channels
            else:
                # Fallback to maximum possible number of groups
                max_divisor = max([g for g in self.search_space.node_features.groups 
                                if g != -1 and input_channels % g == 0 and g <= input_channels], default=1)
                features[groups_idx] = max_divisor
        else:
            # For standard grouped convolutions
            # Find all valid group values
            valid_groups = [g for g in self.search_space.node_features.groups 
                        if g != -1 and input_channels % g == 0 and out_channels % g == 0]
            
            if groups not in valid_groups:
                if valid_groups:
                    # Use closest valid group value
                    features[groups_idx] = min(valid_groups, key=lambda g: abs(g - groups))
                else:
                    # No valid groups with current out_channels, try adjusting out_channels
                    for g in sorted(self.search_space.node_features.groups, reverse=True):
                        if g != -1 and input_channels % g == 0:
                            valid_out_channels = [oc for oc in self.search_space.node_features.out_channels 
                                                if oc % g == 0]
                            if valid_out_channels:
                                features[groups_idx] = g
                                features[out_channels_idx] = min(valid_out_channels, 
                                                            key=lambda oc: abs(oc - out_channels))
                                break
                    else:
                        # Last resort: use groups=1
                        features[groups_idx] = 1


    def to_blueprint(self, input_shape=[3, 32], num_classes=10, enforce_max_preds=False, topological_sort=False):
        assert self.search_space is not None, "search_space must be provided to create a blueprint"
        blueprint = ArcGraph(search_space=self.search_space)
        
        # Track the mapping from original nodes to nodes in the blueprint
        node_mapping = {}
        
        # Add all original nodes to the blueprint
        for i in range(self.vcount()):
            blueprint.add_vertex()
            node_mapping[i] = i  # Since vertices are added in order, the index matches
            
            # Copy node features if they exist
            if "features" in self.vs[i].attribute_names() and self.vs[i]["features"] is not None:
                blueprint.vs[i]["features"] = list(self.vs[i]["features"])
            # Copy n_params and FLOPs if they exist
            if "n_params" in self.vs[i].attribute_names() and self.vs[i]["n_params"] is not None:
                blueprint.vs[i]["n_params"] = self.vs[i]["n_params"]
            if "FLOPs" in self.vs[i].attribute_names() and self.vs[i]["FLOPs"] is not None:
                blueprint.vs[i]["FLOPs"] = self.vs[i]["FLOPs"]
        
        # Add a special node type field to distinguish adapters
        for i in range(blueprint.vcount()):
            blueprint.vs[i]["node_type"] = "original"
        
        # Process nodes in topological order
        adapter_count = 0
        blueprint.vs[node_mapping[0]]["input_shape"] = input_shape
        self._make_node_valid(blueprint, node_mapping[0], input_shape)        
        node_features = blueprint.vs[node_mapping[0]]["features"]
        stride = node_features[feature_index("stride", self.search_space)]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        blueprint.vs[node_mapping[0]]["output_shape"] = [out_channels, input_shape[1] // stride]
        blueprint._add_node_n_params_and_FLOPs(node_mapping[0])
        
        # Process remaining nodes
        for i in range(1, self.vcount()):
            original_predecessors = self.predecessors(i)
            
            # Enforce maximum predecessors if requested
            if enforce_max_preds and len(original_predecessors) > self.search_space.graph_features.max_preds:
                # Sort predecessors by index in descending order (prioritize later nodes)
                original_predecessors = sorted(original_predecessors, reverse=True)
                # Keep only the max allowed number of predecessors
                original_predecessors = original_predecessors[:self.search_space.graph_features.max_preds]
            
            # Get all predecessor output shapes in the blueprint graph
            pred_shapes = []
            for pred_idx in original_predecessors:
                mapped_pred = node_mapping[pred_idx]
                pred_shapes.append(blueprint.vs[mapped_pred]["output_shape"])
            
            # Find maximum channel dimension and minimum spatial dimension
            max_channels = max([shape[0] for shape in pred_shapes])
            min_spatial = min([shape[1] for shape in pred_shapes])
            
            # Process each predecessor and add adapters if needed
            for pred_idx in original_predecessors:
                mapped_pred = node_mapping[pred_idx]
                pred_shape = blueprint.vs[mapped_pred]["output_shape"]
                
                # Start with the original predecessor as the current source
                current_src = mapped_pred
                
                # Check if spatial downsampling is needed
                if pred_shape[1] > min_spatial:
                    # Create maxpool downsampling adapter
                    blueprint.add_vertex()
                    adapter_idx = blueprint.vcount() - 1
                    adapter_count += 1
                    blueprint.vs[adapter_idx]["node_type"] = "breadth_adapter"
                    
                    # Calculate required spatial reduction factor
                    reduction_factor = pred_shape[1] // min_spatial
                    
                    # Set maxpool adapter properties
                    blueprint.vs[adapter_idx]["input_shape"] = list(pred_shape)
                    blueprint.vs[adapter_idx]["output_shape"] = [pred_shape[0], min_spatial]
                    blueprint.vs[adapter_idx]["reduc_factor"] = reduction_factor
                    # NOTE: I think it should be (reduction_factor**2 -1)*((pred_shape[1]// reduction_factor)**2)*pred_shape[0] (count comparisons) but changed to be consistent with deepspeed
                    blueprint.vs[adapter_idx]["FLOPs"] = (reduction_factor**2)*((pred_shape[1]// reduction_factor)**2)*pred_shape[0]
                    
                    # Add edge from predecessor to maxpool adapter
                    blueprint.add_edge(current_src, adapter_idx)
                    current_src = adapter_idx
                
                # Check if channel padding is needed
                if pred_shape[0] < max_channels:
                    # Create channel padding adapter
                    blueprint.add_vertex()
                    adapter_idx = blueprint.vcount() - 1
                    adapter_count += 1
                    blueprint.vs[adapter_idx]["node_type"] = "channel_adapter"
                    
                    # Get current shape (after possible downsampling)
                    if current_src != mapped_pred:
                        current_shape = blueprint.vs[current_src]["output_shape"]
                    else:
                        current_shape = pred_shape
                    
                    # Set channel padding adapter properties
                    blueprint.vs[adapter_idx]["input_shape"] = list(current_shape)
                    blueprint.vs[adapter_idx]["output_shape"] = [max_channels, current_shape[1]]
                    blueprint.vs[adapter_idx]["padding"] = max_channels - current_shape[0]
                    
                    # Add edge from current source to channel adapter
                    blueprint.add_edge(current_src, adapter_idx)
                    current_src = adapter_idx
                
                # Add edge from the last adapter (or original predecessor) to current node
                blueprint.add_edge(current_src, node_mapping[i])
            
            # Set input shape for current node based on adapted predecessors
            blueprint.vs[node_mapping[i]]["input_shape"] = [max_channels, min_spatial]
            
            # Ensure current node is valid
            self._make_node_valid(blueprint, node_mapping[i], [max_channels, min_spatial])
            
            # Compute output shape using the existing method
            input_shape = blueprint.vs[node_mapping[i]]["input_shape"]
            node_features = blueprint.vs[node_mapping[i]]["features"]
            stride = node_features[feature_index("stride", self.search_space)]
            out_channels = node_features[feature_index("out_channels", self.search_space)]
            blueprint.vs[node_mapping[i]]["output_shape"] = [out_channels, input_shape[1] // stride]
            blueprint._add_node_n_params_and_FLOPs(node_mapping[i])
        
        # Add global average pooling layer
        # First, find all nodes with outdegree 0 (sink nodes)
        sink_nodes = []
        for i in range(blueprint.vcount()):
            if "node_type" in blueprint.vs[i].attribute_names() and blueprint.vs[i]["node_type"] == "original" and blueprint.outdegree(i) == 0:
                sink_nodes.append(i)
        
        # If there are no sink nodes, all nodes are already connected in a chain
        # In this case, the last node becomes our sink
        if not sink_nodes and blueprint.vcount() > 0:
            sink_nodes = [node_mapping[self.vcount() - 1]]
        
        # Create global average pooling node
        blueprint.add_vertex()
        gap_idx = blueprint.vcount() - 1
        blueprint.vs[gap_idx]["node_type"] = "global_avg_pool"
        
        # If we have multiple sink nodes, we need to handle potential differences in dimensions
        if len(sink_nodes) > 1:
            # Find the max channels among all sink nodes
            all_sink_shapes = [blueprint.vs[node]["output_shape"] for node in sink_nodes]
            max_channels = max([shape[0] for shape in all_sink_shapes])
            
            # Connect each sink node to GAP, with adapters if needed
            for sink_idx in sink_nodes:
                sink_shape = blueprint.vs[sink_idx]["output_shape"]
                current_src = sink_idx
                
                # Check if channel padding is needed
                if sink_shape[0] < max_channels:
                    # Create channel padding adapter
                    blueprint.add_vertex()
                    adapter_idx = blueprint.vcount() - 1
                    adapter_count += 1
                    blueprint.vs[adapter_idx]["node_type"] = "channel_adapter"
                    
                    # Set channel padding adapter properties
                    blueprint.vs[adapter_idx]["input_shape"] = list(sink_shape)
                    blueprint.vs[adapter_idx]["output_shape"] = [max_channels, sink_shape[1]]
                    blueprint.vs[adapter_idx]["padding"] = max_channels - sink_shape[0]
                    
                    # Add edge from sink to channel adapter
                    blueprint.add_edge(current_src, adapter_idx)
                    current_src = adapter_idx
                    
                    # Update shape for GAP input
                    sink_shape = blueprint.vs[adapter_idx]["output_shape"]
                
                # Connect to global average pooling
                blueprint.add_edge(current_src, gap_idx)
            
            # Set GAP input shape (spatial dimension from last sink, max channels from all sinks)
            blueprint.vs[gap_idx]["input_shape"] = [max_channels, sink_shape[1]]
        else:
            # Single sink case - straightforward connection
            sink_idx = sink_nodes[0]
            sink_shape = blueprint.vs[sink_idx]["output_shape"]
            blueprint.vs[gap_idx]["input_shape"] = list(sink_shape)
            blueprint.add_edge(sink_idx, gap_idx)
        
        # Set GAP output shape - reduces spatial dimensions to 1
        input_shape = blueprint.vs[gap_idx]["input_shape"]
        blueprint.BBGP = input_shape[1]
        blueprint.vs[gap_idx]["output_shape"] = [input_shape[0], 1]
        gap_input_shape = blueprint.vs[gap_idx]["input_shape"]
        gap_FLOPs = gap_input_shape[0]*gap_input_shape[1]**2
        if len(blueprint.predecessors(gap_idx)) > 1:
            gap_FLOPs += (len(blueprint.predecessors(gap_idx)) - 1)*gap_input_shape[0]*(gap_input_shape[1]**2)
        blueprint.vs[gap_idx]["FLOPs"] = gap_FLOPs
        
        # Add linear classifier layer
        blueprint.add_vertex()
        classifier_idx = blueprint.vcount() - 1
        blueprint.vs[classifier_idx]["node_type"] = "classifier"
        
        # Set classifier layer properties
        in_features = blueprint.vs[gap_idx]["output_shape"][0]
        blueprint.vs[classifier_idx]["input_shape"] = [in_features, 1]
        blueprint.vs[classifier_idx]["output_shape"] = [num_classes, 1]
        blueprint.vs[classifier_idx]["n_params"] = in_features * num_classes + num_classes
        blueprint.vs[classifier_idx]["FLOPs"] = 2 * in_features * num_classes 
        
        # Connect GAP to classifier
        blueprint.add_edge(gap_idx, classifier_idx)
        
        blueprint.valid = True
        blueprint.shapes_added = True
        blueprint.params_and_FLOPs_added = True
        blueprint.n_params = blueprint._count_params()
        blueprint.FLOPs = blueprint._count_FLOPs()

        # Create a topologically sorted version of the blueprint
        if not topological_sort:
            return blueprint
        else:
            sorted_vertices = blueprint.topological_sorting()
            sorted_blueprint = ArcGraph(search_space=self.search_space)
            
            # Create a mapping from old indices to new indices
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_vertices)}
            
            # Add vertices in sorted order and copy their attributes
            for new_idx, old_idx in enumerate(sorted_vertices):
                sorted_blueprint.add_vertex()
                # Copy all attributes from the original blueprint
                for attr in blueprint.vs[old_idx].attribute_names():
                    sorted_blueprint.vs[new_idx][attr] = blueprint.vs[old_idx][attr]
            
            # Add edges, translating from old indices to new indices
            for edge in blueprint.get_edgelist():
                old_source, old_target = edge
                new_source, new_target = old_to_new[old_source], old_to_new[old_target]
                sorted_blueprint.add_edge(new_source, new_target)
            
            # Copy global attributes
            sorted_blueprint.valid = blueprint.valid
            sorted_blueprint.shapes_added = blueprint.shapes_added
            sorted_blueprint.params_and_FLOPs_added = blueprint.params_and_FLOPs_added
            sorted_blueprint.n_params = blueprint.n_params
            sorted_blueprint.FLOPs = blueprint.FLOPs
            if hasattr(blueprint, 'BBGP'):
                sorted_blueprint.BBGP = blueprint.BBGP
            
            return sorted_blueprint


    def _layer_blueprint(self):
        """
        Calculate the execution layers for parallel processing.
        Each node is assigned to a layer based on the longest path from input.
        Nodes in the same layer can be executed in parallel.
        
        Returns:
            dict: Node indices mapped to their execution layer
            dict: Execution layers (indices) mapped to lists of nodes in that layer
        """
        layer_of_node_dict = {}
        visited = set()
        
        def dfs(node_idx):
            if node_idx in visited:
                return layer_of_node_dict[node_idx]
            
            visited.add(node_idx)
            
            # If no predecessors, this is layer 0
            predecessors = self.predecessors(node_idx)
            if not predecessors:
                layer_of_node_dict[node_idx] = 0
                return 0
            
            # Find the maximum layer among predecessors
            max_layer = -1
            for pred in predecessors:
                pred_layer = dfs(pred)
                max_layer = max(max_layer, pred_layer)
            
            # This node's layer is one more than its deepest predecessor
            layer_of_node_dict[node_idx] = max_layer + 1
            return layer_of_node_dict[node_idx]
        
        # Calculate layer for each node
        for i in range(self.vcount()):
            if i not in visited:
                dfs(i)
        
        # Group nodes by layer for parallel execution
        nodes_of_layer_dict = {}
        for node, layer in layer_of_node_dict.items():
            if layer not in nodes_of_layer_dict:
                nodes_of_layer_dict[layer] = []
            nodes_of_layer_dict[layer].append(node)
        
        return layer_of_node_dict, nodes_of_layer_dict


    def to_torch(self, input_shape=[3, 32], num_classes=10, enforce_max_preds=False):
        import torch
        import torch.nn as nn
        
        blueprint = self.to_blueprint(input_shape=input_shape, num_classes=num_classes, enforce_max_preds=enforce_max_preds)
        
        class ArcGraphModel(nn.Module):
            def __init__(self, blueprint):
                super(ArcGraphModel, self).__init__()
                from custom_layers import CustomSE, ChannelPad, AggTensors
                self.layers = nn.ModuleDict()
                self.topology = []
                
                execution_order = []
                visited = set()

                self.n_params = blueprint.n_params
                self.FLOPs = blueprint.FLOPs
                self.BBGP = blueprint.BBGP
                
                def visit(node_idx):
                    if node_idx in visited:
                        return
                    visited.add(node_idx)
                    for pred in blueprint.predecessors(node_idx):
                        visit(pred)
                    execution_order.append(node_idx)
                
                for i in range(blueprint.vcount()):
                    if i not in visited:
                        visit(i)
                
                for node_idx in execution_order:
                    node = blueprint.vs[node_idx]
                    node_type = "original"
                    if "node_type" in node.attribute_names():
                        node_type = node["node_type"]
                    
                    if node_type == 'original':
                        features = node['features']
                        in_channels, _ = node['input_shape']
                        out_channels = features[feature_index('out_channels', blueprint.search_space)]
                        kernel_size = features[feature_index('kernel_size', blueprint.search_space)]
                        stride = features[feature_index('stride', blueprint.search_space)]
                        groups = features[feature_index('groups', blueprint.search_space)]
                        if groups == -1:
                            groups = in_channels                            

                        if len(blueprint.predecessors(node_idx)) > 1:
                            if 'aggregation' in blueprint.search_space.node_features.__dict__.keys():
                                    aggregation = features[feature_index('aggregation', blueprint.search_space)]
                                    agg = AggTensors(aggregation)
                                    self.layers[f'agg_{node_idx}'] = agg
                            else:
                                agg = AggTensors('sum')
                                self.layers[f'agg_{node_idx}'] = agg
                        
                        conv = nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=kernel_size // 2,
                            groups=groups,
                            bias=True
                        )
                        
                        batchnorm = nn.BatchNorm2d(out_channels)
                        activation = nn.ReLU(inplace=True)
                        
                        self.layers[f'conv_{node_idx}'] = conv
                        self.layers[f'bn_{node_idx}'] = batchnorm
                        self.layers[f'act_{node_idx}'] = activation
                        if 'squeeze_excitation' in blueprint.search_space.node_features.__dict__.keys():
                            se = features[feature_index('squeeze_excitation', blueprint.search_space)]
                            if se == 1:
                                self.layers[f'se_{node_idx}'] = CustomSE(out_channels)
                    
                    elif node_type == 'breadth_adapter':
                        reduc_factor = node['reduc_factor']
                        self.layers[f'maxpool_{node_idx}'] = nn.MaxPool2d(kernel_size=reduc_factor, stride=reduc_factor)
                    
                    elif node_type == 'channel_adapter':
                        in_channels, _ = node['input_shape']
                        out_channels, _ = node['output_shape']
                        self.layers[f'chpad_{node_idx}'] = ChannelPad(in_channels, out_channels)
                    
                    elif node_type == 'global_avg_pool':
                        if len(blueprint.predecessors(node_idx)) > 1:
                            self.layers[f'agg_{node_idx}'] = AggTensors('sum')
                        self.layers[f'gap_{node_idx}'] = nn.AdaptiveAvgPool2d(1)
                    
                    elif node_type == 'classifier':
                        in_features = node['input_shape'][0]
                        num_classes = node['output_shape'][0]
                        self.layers[f'classifier_{node_idx}'] = nn.Linear(in_features, num_classes)
                    
                    self.topology.append((node_idx, blueprint.predecessors(node_idx)))
                
                # Store the successor information to determine when tensors can be freed
                self.successors = {i: [] for i in range(blueprint.vcount())}
                for node_idx, predecessors in self.topology:
                    for pred_idx in predecessors:
                        self.successors[pred_idx].append(node_idx)
                
                # Store the final output node for reference
                self.output_node = self.topology[-1][0]
            
            def forward(self, x):
                outputs = {}
                # Keep track of how many successors still need each tensor
                remaining_uses = {i: len(self.successors[i]) for i in range(len(self.topology))}
                
                for node_idx, predecessors in self.topology:
                    node_type = blueprint.vs[node_idx]["node_type"]
                    
                    if len(predecessors) == 0:
                        if not node_idx == 0:
                            raise ValueError(f"Node {node_idx}, but only the first node may have indegree = 0")
                        elif node_type == 'original':
                            x = self.layers[f'conv_{node_idx}'](x)
                            x = self.layers[f'bn_{node_idx}'](x)
                            x = self.layers[f'act_{node_idx}'](x)
                            
                            if f'se_{node_idx}' in self.layers:
                                x = self.layers[f'se_{node_idx}'](x)
                            
                            outputs[node_idx] = x
                        else:
                            raise ValueError(f"Unexpected node type {node_type} for input node")
                    else:
                        if len(predecessors) == 1:
                            pred_idx = predecessors[0]
                            x = outputs[pred_idx]
                            
                            # Decrement usage count of predecessor
                            remaining_uses[pred_idx] -= 1
                            # Free memory if no longer needed
                            if remaining_uses[pred_idx] == 0 and pred_idx != self.output_node:
                                outputs[pred_idx] = None
                        else:
                            pred_outputs = []
                            for pred_idx in predecessors:
                                pred_outputs.append(outputs[pred_idx])
                                # Decrement usage count
                                remaining_uses[pred_idx] -= 1
                                # Free memory if no longer needed
                                if remaining_uses[pred_idx] == 0 and pred_idx != self.output_node:
                                    outputs[pred_idx] = None
                            
                            if node_type == 'original' or node_type == 'global_avg_pool':
                                x = self.layers[f'agg_{node_idx}'](torch.stack(pred_outputs))
                            else:
                                raise ValueError(f'{node_type} nodes may not have >1 predecessors, received {len(predecessors)}')
                            
                            # Clear the list to free up memory
                            pred_outputs = None
                        
                        if node_type == 'original':
                            x = self.layers[f'conv_{node_idx}'](x)
                            x = self.layers[f'bn_{node_idx}'](x)
                            x = self.layers[f'act_{node_idx}'](x)
                            
                            if f'se_{node_idx}' in self.layers:
                                x = self.layers[f'se_{node_idx}'](x)
                        
                        elif node_type == 'breadth_adapter':
                            x = self.layers[f'maxpool_{node_idx}'](x)
                        
                        elif node_type == 'channel_adapter':
                            x = self.layers[f'chpad_{node_idx}'](x)
                        
                        elif node_type == 'global_avg_pool':
                            x = self.layers[f'gap_{node_idx}'](x)
                            x = torch.flatten(x, 1)
                        
                        elif node_type == 'classifier':
                            x = self.layers[f'classifier_{node_idx}'](x)
                        
                        outputs[node_idx] = x
                
                result = outputs[self.output_node]
                # Clear the outputs dictionary to free remaining memory
                outputs.clear()
                return result
        
        model = ArcGraphModel(blueprint)
        return model
    
   
    def to_keras(self, input_shape=(3, 32), num_classes=10, enforce_max_preds=False, backend = "torch"):
        """
        Convert the graph blueprint into a Keras model using the torch backend.

        Args:
            input_shape (tuple): Shape of the input tensor in channels-first format.
                                (e.g., (channels, height))
            num_classes (int): Number of output classes.
            enforce_max_preds (bool): Whether to enforce maximum predecessor constraints.

        Returns:
            keras.Model: The constructed Keras model.
        """
        import os
        os.environ["KERAS_BACKEND"] = backend
        import keras
        from keras.models import Model

        # Custom SE layer rewritten for torch backend.
        class KerasCustomSE(keras.layers.Layer):
            def __init__(self, channels, reduction=16, **kwargs):
                super(KerasCustomSE, self).__init__(**kwargs)
                self.channels = channels
                self.global_avg_pool = keras.layers.GlobalAveragePooling2D()
                self.fc1 = keras.layers.Dense(channels // reduction, activation='relu')
                self.fc2 = keras.layers.Dense(channels, activation='sigmoid')

            def call(self, inputs):
                # Assume inputs shape: (batch, channels, height, width)
                se = self.global_avg_pool(inputs)  # results in shape (batch, channels)
                se = self.fc1(se)
                se = self.fc2(se)
                # Reshape to (batch, channels, 1, 1) for broadcasting.
                se = se.reshape(se.shape[0], self.channels, 1, 1)
                return inputs * se

        # Custom ChannelPad layer rewritten using torch tensor operations.
        class KerasChannelPad(keras.layers.Layer):
            def __init__(self, in_channels, out_channels, **kwargs):
                super(KerasChannelPad, self).__init__(**kwargs)
                self.in_channels = in_channels
                self.out_channels = out_channels
                if in_channels > out_channels:
                    raise ValueError(f"in_channels (={in_channels}) must be inferior out_channels (={out_channels}) to use channel padding")
                elif in_channels == out_channels:
                    self.pad_layer = keras.layers.Identity()
                else:
                    self.pad_layer = keras.layers.ZeroPadding2D(padding=((0, out_channels-in_channels), (0, 0)))
                self.perm =(0,2,1,3)

            def call(self, inputs):
                output = keras.ops.transpose(inputs, axes=self.perm)
                output = self.pad_layer(output)
                output = keras.ops.transpose(output, axes=self.perm)
                return output

        # Custom aggregation layer: sums a list of tensors.
        class KerasAggTensors(keras.layers.Layer):
            def __init__(self, aggregation='sum', **kwargs):
                super(KerasAggTensors, self).__init__(**kwargs)
                if aggregation != 'sum':
                    raise NotImplementedError("Only 'sum' aggregation is implemented.")
                self.aggregator = keras.layers.Add()

            def call(self, inputs):
                # Sum the list of tensors.
                return self.aggregator(inputs)

        # Generate a blueprint from the graph.
        blueprint = self.to_blueprint(input_shape=input_shape, num_classes=num_classes, enforce_max_preds=enforce_max_preds)

        # Topologically sort nodes.
        visited = set()
        execution_order = []
        def visit(node_idx):
            if node_idx in visited:
                return
            visited.add(node_idx)
            for pred in blueprint.predecessors(node_idx):
                visit(pred)
            execution_order.append(node_idx)
        for i in range(blueprint.vcount()):
            visit(i)

        # Create a topology list: each element is (node_index, list of predecessor indices).
        topology = [(node_idx, blueprint.predecessors(node_idx)) for node_idx in execution_order]

        # Define a Keras model that mirrors the blueprint.
        class KerasArcGraphModel(Model):
            def __init__(self, blueprint, topology):
                super(KerasArcGraphModel, self).__init__()
                self.n_params = blueprint.n_params
                self.FLOPs = blueprint.FLOPs
                self.BBGP = blueprint.BBGP

                self.topology = topology
                self.num_nodes = blueprint.vcount()
                self.layers_dict = {}
                # For nodes with multiple predecessors, use the aggregation layer.
                self.agg_layer = KerasAggTensors('sum')

                # Create a layer/block for each node in topological order.
                for node_idx in execution_order:
                    node = blueprint.vs[node_idx]
                    # Determine node type; default to "original" if not set.
                    node_type = node["node_type"] if "node_type" in node.attribute_names() else "original"
                    if node_type == 'original':
                        features = node['features']
                        # Assume node['input_shape'] is in channels-first order.
                        in_channels = node['input_shape'][0]
                        out_channels = features[feature_index('out_channels', blueprint.search_space)]
                        kernel_size = features[feature_index('kernel_size', blueprint.search_space)]
                        stride = features[feature_index('stride', blueprint.search_space)]
                        groups = features[feature_index('groups', blueprint.search_space)]
                        if groups == -1:
                            groups = in_channels
                        block = keras.Sequential()
                        block.add(keras.layers.Conv2D(filters=out_channels,
                                                    kernel_size=kernel_size,
                                                    strides=stride,
                                                    padding='same',
                                                    groups=groups,
                                                    use_bias=True))
                        block.add(keras.layers.BatchNormalization())
                        block.add(keras.layers.ReLU())
                        # Append SE block if defined.
                        if "squeeze_excitation" in blueprint.search_space.node_features.__dict__:
                            se = features[feature_index('squeeze_excitation', blueprint.search_space)]
                            if se == 1:
                                block.add(KerasCustomSE(out_channels))
                        self.layers_dict[node_idx] = block

                    elif node_type == 'breadth_adapter':
                        reduc_factor = node['reduc_factor']
                        self.layers_dict[node_idx] = keras.layers.MaxPooling2D(pool_size=(reduc_factor, reduc_factor),
                                                                            strides=reduc_factor,
                                                                            padding='same')

                    elif node_type == 'channel_adapter':
                        in_channels = node['input_shape'][0]
                        out_channels = node['output_shape'][0]
                        self.layers_dict[node_idx] = KerasChannelPad(in_channels, out_channels)

                    elif node_type == 'global_avg_pool':
                        self.layers_dict[node_idx] = keras.layers.GlobalAveragePooling2D()

                    elif node_type == 'classifier':
                        self.layers_dict[node_idx] = keras.layers.Dense(num_classes)
                    else:
                        raise ValueError(f"Unknown node type: {node_type}")

            def call(self, inputs, training=False):
                outputs = {}
                # Process nodes in topological order.
                for node_idx, preds in self.topology:
                    if not preds:
                        # For input nodes, apply the layer directly.
                        x = self.layers_dict[node_idx](inputs)
                        outputs[node_idx] = x
                    else:
                        # Aggregate outputs from predecessor nodes.
                        pred_outputs = [outputs[p] for p in preds]
                        if len(pred_outputs) > 1:
                            aggregated = self.agg_layer(pred_outputs)
                        else:
                            aggregated = pred_outputs[0]
                        x = self.layers_dict[node_idx](aggregated)
                        outputs[node_idx] = x
                # Return the output from the final node.
                last_node = self.topology[-1][0]
                return outputs[last_node]

        input_tensor = keras.layers.Input(shape=(input_shape[0], input_shape[1], input_shape[1]))
        keras_arc_model = KerasArcGraphModel(blueprint, topology)
        output_tensor = keras_arc_model(input_tensor)
        keras_model = keras.Model(inputs=input_tensor, outputs=output_tensor)
        return keras_model


    def memory_usage_animation(self, output_path=None, delays=50, backbone=True, plot_memory=True):
        """
        Create an animation showing how node outputs are created and freed from memory
        during model forward execution using parallel processing where possible.
        Also tracks and plots memory usage over time.
        
        Args:
            output_path (str): Path to save the animation GIF
            delays (int): Delay between frames in milliseconds
            backbone (bool): Whether to emphasize backbone connections
            plot_memory (bool): Whether to plot memory usage graph
            
        Returns:
            tuple: (animation_path, memory_plot_path) or animation object
        """
        import os
        import tempfile
        import imageio
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import pygraphviz as pgv
        from copy import deepcopy
        import matplotlib.image as mpimg
        import numpy as np
        
        # Create a blueprint if not already one
        if not hasattr(self, 'valid') or not self.valid:
            print("Warning: Creating blueprint from invalid graph for visualization.")
            blueprint = self
        else:
            blueprint = self
        
        # Calculate execution layers for parallel processing
        node_layers, execution_groups = blueprint._layer_blueprint()
        
        # Count how many nodes depend on each node's output
        dependency_count = {i: 0 for i in range(blueprint.vcount())}
        for node in range(blueprint.vcount()):
            for pred in blueprint.predecessors(node):
                dependency_count[pred] += 1
        
        # Track memory state for animation
        frames = []
        current_state = {i: "not_computed" for i in range(blueprint.vcount())}
        memory_usage = []
        current_memory = 0
        
        # Helper function to calculate tensor memory size in bytes (32 bits per element)
        def calc_tensor_memory(node_idx):
            if "output_shape" not in blueprint.vs[node_idx].attribute_names():
                return 0
            
            shape = blueprint.vs[node_idx]["output_shape"]
            # Memory = channels * height * width * 4 bytes (32 bits)
            return shape[0] * (shape[1] ** 2) * 4
        
        # Add initial frame
        frames.append(deepcopy(current_state))
        memory_usage.append(current_memory)
        
        # Process each layer in order
        sorted_layers = sorted(execution_groups.keys())
        
        for layer in sorted_layers:
            nodes_in_layer = execution_groups[layer]
            
            # First compute all nodes in this layer (they run in parallel)
            for node in nodes_in_layer:
                current_state[node] = "in_memory"
                current_memory += calc_tensor_memory(node)
            
            # Add a frame after computing the entire layer
            frames.append(deepcopy(current_state))
            memory_usage.append(current_memory)
            
            # Then check which nodes can be freed after this layer
            freed_memory = 0
            freed_nodes = []
            
            for node_idx in range(blueprint.vcount()):
                if current_state[node_idx] == "in_memory" and node_idx not in nodes_in_layer:
                    # Check if this node is a predecessor to any node in future layers
                    is_still_needed = False
                    
                    for future_layer in [l for l in sorted_layers if l > layer]:
                        for future_node in execution_groups[future_layer]:
                            if node_idx in blueprint.predecessors(future_node):
                                is_still_needed = True
                                break
                        if is_still_needed:
                            break
                    
                    if not is_still_needed:
                        freed_nodes.append(node_idx)
                        freed_memory += calc_tensor_memory(node_idx)
            
            # If any nodes were freed, create a new frame
            if freed_nodes:
                for node in freed_nodes:
                    current_state[node] = "freed"
                current_memory -= freed_memory
                
                frames.append(deepcopy(current_state))
                memory_usage.append(current_memory)
        
        # Create a function to draw the graph with current memory state
        def draw_state(state):
            graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
            cmap = plt.get_cmap('cool')
            
            # Define memory state colors
            memory_colors = {
                "not_computed": "#F5F5DC",  # Beige
                "in_memory": "#FFA500",     # Orange
                "freed": "#A9A9A9"          # Grey
            }
            
            for idx in range(blueprint.vcount()):
                # Use attribute check rather than get method
                node_type = "original"
                if "node_type" in blueprint.vs[idx].attribute_names():
                    node_type = blueprint.vs[idx]["node_type"]
                
                # Get the memory state color
                fillcolor = memory_colors[state[idx]]
                
                # Add layer information to node label
                layer_info = f"Layer {node_layers[idx]}"
                
                if node_type == "original" and "features" in blueprint.vs[idx].attribute_names():
                    features = blueprint.vs[idx]["features"]
                    if blueprint.search_space and hasattr(blueprint.search_space, "aliases"):
                        label_parts = []
                        feature_names = list(blueprint.search_space.node_features.__dict__.keys())              
                        for i, name in enumerate(feature_names):
                            if i < len(features):
                                alias = blueprint.search_space.aliases.get(name, name)
                                label_parts.append(f"{alias}{features[i]}")
                        
                        label = f"({idx}) {layer_info}\n" + " ".join(label_parts)
                    else:
                        label = f"({idx}) {layer_info}\n{features}"
                        
                    fixedsize = False
                    width = None
                    height = None
                    stride = features[feature_index("stride", blueprint.search_space)]
                    shape = 'box'
                    if stride > 1:
                        fixedsize = True
                        width = 2.5
                        height = 0.9
                        shape = 'invtrapezium'
                    if "n_params" in blueprint.vs[idx].attribute_names():
                        n_params = blueprint.vs[idx]["n_params"]
                        label += f"\nparams: {format_number_spaces(n_params)}"
                    if "FLOPs" in blueprint.vs[idx].attribute_names():
                        FLOPs = blueprint.vs[idx]["FLOPs"]
                        label += f"\nFLOPs: {format_number_spaces(FLOPs)}"
                        
                    graph.add_node(
                        idx, 
                        label=label, 
                        color='black', 
                        fillcolor=fillcolor, 
                        shape=shape,
                        style='filled',
                        fixedsize=fixedsize,
                        width=width,
                        height=height,
                        fontsize=12
                    )
                elif node_type == "breadth_adapter":
                    reduc_factor = "?"
                    if "reduc_factor" in blueprint.vs[idx].attribute_names():
                        reduc_factor = blueprint.vs[idx]["reduc_factor"]
                    label = f"({idx}) {layer_info}\nMaxPool {reduc_factor}"
                    graph.add_node(
                        idx, 
                        label=label, 
                        color='black', 
                        fillcolor=fillcolor, 
                        shape='invtrapezium',
                        style='filled',
                        width=1.4, 
                        height=0.5,
                        fixedsize=True,
                        fontsize=12
                    )
                elif node_type == "channel_adapter":
                    padding = "?"
                    if "padding" in blueprint.vs[idx].attribute_names():
                        padding = blueprint.vs[idx]["padding"]
                    
                    label = f"({idx}) {layer_info}\nChannelPad {padding}"
                    graph.add_node(
                        idx, 
                        label=label, 
                        color='black', 
                        fillcolor=fillcolor, 
                        shape='box',
                        style='filled', 
                        fontsize=12
                    )
                elif node_type == "global_avg_pool":
                    label = f"({idx}) {layer_info}\nGlobalAvgPool"
                    if "FLOPs" in blueprint.vs[idx].attribute_names():
                        FLOPs = blueprint.vs[idx]["FLOPs"]
                        label += f"\nFLOPs: {format_number_spaces(FLOPs)}"
                    graph.add_node(
                        idx, 
                        label=label, 
                        color='black', 
                        fillcolor=fillcolor, 
                        shape='invtrapezium',
                        style='filled',
                        width=2, 
                        height=0.7,
                        fixedsize=True,
                        fontsize=12
                    )
                elif node_type == "classifier":
                    label = f"({idx}) {layer_info}\nLinear"
                    if "n_params" in blueprint.vs[idx].attribute_names():
                        n_params = blueprint.vs[idx]["n_params"]
                        label += f"\nparams: {format_number_spaces(n_params)}"
                    if "FLOPs" in blueprint.vs[idx].attribute_names():
                        FLOPs = blueprint.vs[idx]["FLOPs"]
                        label += f"\nFLOPs: {format_number_spaces(FLOPs)}"
                    graph.add_node(
                        idx, 
                        label=label, 
                        color='black', 
                        fillcolor=fillcolor, 
                        shape='box',
                        style='filled', 
                        fontsize=12
                    )
                else:
                    label = f"({idx}) {layer_info}"                
                    graph.add_node(
                        idx, 
                        label=label, 
                        color='black', 
                        fillcolor=fillcolor, 
                        shape='box',
                        style='filled', 
                        fontsize=12
                    )
            
            # Add edges with latent size labels
            for edge in blueprint.get_edgelist():
                source, target = edge
                
                # Add both input and output shapes on edge labels
                input_shape_label = ""
                output_shape_label = ""
                
                # Get the output shape of the source node (edge start)
                if "output_shape" in blueprint.vs[source].attribute_names():
                    source_output = blueprint.vs[source]["output_shape"]
                    output_shape_label = f" {source_output[0]}×{source_output[1]}²"
                
                # Get the input shape of the target node (edge end)
                if "input_shape" in blueprint.vs[target].attribute_names():
                    target_input = blueprint.vs[target]["input_shape"]
                    input_shape_label = f" {target_input[0]}×{target_input[1]}²"
                
                # Combine labels if both exist
                if output_shape_label and input_shape_label:
                    if output_shape_label == input_shape_label:
                        # If they're the same, just use one
                        label = output_shape_label
                    else:
                        # If different, show both
                        label = f" {output_shape_label}\n{input_shape_label}"
                elif output_shape_label:
                    label = output_shape_label
                elif input_shape_label:
                    label = input_shape_label
                else:
                    label = ""
                
                edge_weight = 1
                if backbone and source == target - 1:
                    edge_weight = 3
                
                if label:
                    graph.add_edge(
                        source, 
                        target, 
                        weight=edge_weight,
                        label=label,
                        fontsize=12,
                        labeldistance=1.5,
                        labelangle=0
                    )
                else:
                    graph.add_edge(source, target, weight=edge_weight)
            
            graph.layout(prog='dot')
            
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            graph.draw(temp_path, prog='dot', args='-Gdpi=300')
            return temp_path
        
        # Create a plot of memory usage over time
        def plot_memory_usage():
            plt.figure(figsize=(10, 6))
            
            # Convert to MB for readability
            memory_mb = [m / (1024 * 1024) for m in memory_usage]
            steps = list(range(len(memory_usage)))
            
            plt.plot(steps, memory_mb, 'b-', linewidth=2)
            plt.fill_between(steps, 0, memory_mb, alpha=0.2, color='blue')
            
            plt.title('Memory Usage During Parallel Forward Pass', fontsize=14)
            plt.xlabel('Execution Step', fontsize=12)
            plt.ylabel('Memory Usage (MB)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Mark execution layer boundaries with vertical lines
            layer_boundaries = []
            frame_idx = 0
            for layer in sorted_layers:
                # Add vertical line at the start of each layer's execution
                if frame_idx < len(steps):
                    plt.axvline(x=frame_idx, color='red', linestyle='--', alpha=0.5)
                    plt.text(frame_idx, max(memory_mb)*1.02, f"Layer {layer}", 
                            rotation=90, verticalalignment='bottom')
                layer_boundaries.append(frame_idx)
                frame_idx += 2  # Each layer has two frames (compute, then free)
            
            # Mark peak memory usage
            peak_memory = max(memory_mb)
            peak_step = memory_mb.index(peak_memory)
            plt.scatter(peak_step, peak_memory, color='red', s=100, zorder=5)
            plt.annotate(f'Peak: {peak_memory:.2f} MB', 
                        xy=(peak_step, peak_memory),
                        xytext=(peak_step + 1, peak_memory * 1.1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=12)
            
            plt.tight_layout()
            
            # If saving animation, also save the memory plot
            if output_path:
                memory_plot_path = os.path.splitext(output_path)[0] + "_memory.png"
                plt.savefig(memory_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                return memory_plot_path
            else:
                # Return the figure for display
                memory_fig = plt.gcf()
                plt.close()
                return memory_fig
        
        # Create all frames
        frame_files = []
        print(f"Generating {len(frames)} frames for memory usage animation...")
        for i, state in enumerate(frames):
            if i % 10 == 0:
                print(f"Processing frame {i+1}/{len(frames)}...")
            frame_files.append(draw_state(state))
        
        # Create the memory usage plot
        memory_plot_output = None
        if plot_memory:
            print("Generating memory usage plot...")
            memory_plot_output = plot_memory_usage()
        
        # Create the animation
        if output_path:
            directory = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(directory, exist_ok=True)
            
            print(f"Creating animation and saving to {output_path}...")
            # Use imageio to create the GIF
            with imageio.get_writer(output_path, mode='I', duration=delays/1000) as writer:
                for file in frame_files:
                    image = imageio.imread(file)
                    writer.append_data(image)
                    # Clean up temporary files
                    try:
                        os.remove(file)
                    except:
                        pass
            
            print("Animation created successfully!")
            return (output_path, memory_plot_output)
        else:
            # Display animation in notebook if no output path
            fig, ax = plt.subplots(figsize=(12, 8))
            
            def update(frame):
                ax.clear()
                img = plt.imread(frame_files[frame])
                ax.imshow(img)
                ax.axis('off')
                return ax
            
            print("Creating animation for display...")
            ani = animation.FuncAnimation(fig, update, frames=len(frame_files), interval=delays)
            plt.close()
            
            # Clean up temporary files
            for file in frame_files:
                try:
                    os.remove(file)
                except:
                    pass
            
            print("Animation ready for display!")
            return (ani, memory_plot_output)
    




# X = np.array([[32, 3, 2, 1], [64, 3, 2, -1], [128, 3, 2, 1]], dtype = "object")
# A = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])

# g = ArcGraph(X=X, A=A)
# g.search_space = SearchSpace()
# g.is_valid(input_shape = [3, 32], verbose = True)
# g.add_latent_shapes([3, 32])
# g.add_n_params_and_FlOPs()
# v = g.to_V()

# g2=g.to_blueprint(input_shape=[3, 32])

# g2.plot(display=True)

# model = g.to_torch(input_shape=[3, 32])

# from torchsummary import summary
# from model_profiling import profile_model

# summary(model, input_size=(3, 32, 32))
# model.eval()
# params, flops = profile_model(model, input_shape=(1, 3, 32, 32))
# print(f' n_params(profiler): {params}')
# print(f' FLOPs (profiler): {flops} \n')

# print(f' n_params (blueprint): {g2.n_params}')
# print(f' FLOPs (blueprint): {g2.FLOPs} \n')


# # Compute the output for a random input
# random_input = torch.randn(1, 3, 32, 32)
# output = model(random_input)
