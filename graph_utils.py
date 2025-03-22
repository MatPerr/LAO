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
            X_onehot = V[:-n_adj_comp].reshape(n_nodes, -1)
            X = np.empty((n_nodes, len(search_space.node_features.__dict__)), dtype = 'object')
            for i in range(n_nodes):
                shift = 0
                for j, possible_values in enumerate(search_space.node_features.__dict__.values()):
                    onehot_val = X_onehot[i, shift:shift + len(possible_values)]
                    X[i, j] = possible_values[np.argmax(onehot_val)]
                    shift += len(possible_values)
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
        for i in range(n_nodes):
            for j, possible_values in enumerate(self.search_space.node_features.__dict__.values()):
                val = self.vs[i]["features"][j]
                V += one_hot_encode(val, possible_values).tolist()

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
        input_channels = input_shape[0]
        input_breadth = input_shape[1]
        self.vs[0]["output_shape"] = [out_channels, input_breadth // stride]

        for i in range(1, self.vcount()):
            self.vs[i]["input_shape"] = self._get_input_shape(i)
            self.vs[i]["output_shape"] = self._get_output_shape(i)

        self.shapes_added = True     

    def add_n_params_and_FlOPs(self):
        assert self.search_space is not None, "search_space must be provided to count node params"
        assert self.valid, "the graph must be valid to count node params"
        assert self.shapes_added, "the latent shapes must be added to count node params"
        for node in self.vs:
            input_channels, input_breadth = node['input_shape']
            node_features = node["features"]
            out_channels = node_features[feature_index("out_channels", self.search_space)]
            kernel_size = node_features[feature_index("kernel_size", self.search_space)]
            stride = node_features[feature_index("stride", self.search_space)]
            groups = node_features[feature_index("groups", self.search_space)]
            if groups == -1:
                groups = input_channels
            # Both n_params and FLOPs estimation assume padding = 'same'
            # n_params
            conv_params = out_channels*((kernel_size**2)*(input_channels // groups) + 1)
            bn_params = 2*out_channels
            n_params = conv_params + bn_params
            if "squeeze_excitation" in self.search_space.node_features.__dict__.keys():
                if node_features[feature_index("squeeze_excitation", self.search_space)] == 1:
                    n_params += (out_channels//16)*(out_channels + 1) + (out_channels)*(out_channels//16 + 1)
            node['n_params'] = n_params
            
            # FLOPs
            conv_FLOPs = (2*(input_breadth**2)*(kernel_size**2)*input_channels*out_channels) // ((stride**2)*groups)
            # NOTE: 4* during training and 2* during inference, averaged to 3*
            bn_FLOPs = 3*out_channels*(input_breadth // stride)**2
            act_flops = out_channels*(input_breadth // stride)**2
            FLOPs = conv_FLOPs + bn_FLOPs + act_flops

            if "squeeze_excitation" in self.search_space.node_features.__dict__.keys():
                if node_features[feature_index("squeeze_excitation", self.search_space)] == 1:
                    # Global average pooling
                    FLOPs += out_channels*(input_breadth // stride)**2
                    # First conv
                    FLOPs += 2*out_channels*(out_channels//16)
                    # First activation function
                    FLOPs += out_channels//16
                    # Second conv
                    FLOPs += 2*(out_channels//16)*out_channels
                    # Second activation function
                    FLOPs += out_channels

            if "aggregation" in self.search_space.node_features.__dict__.keys() and node_features[feature_index("aggregation", self.search_space)] != "sum":
                raise NotImplementedError(f"Aggregation type {node_features[feature_index('aggregation', self.search_space)]} not implemented, only 'sum' is supported for now")
            else:
                predecessors = self.predecessors(node.index)
                if len(predecessors) > 1:
                    input_elements = input_channels * (input_breadth**2)
                    agg_flops = (len(predecessors) - 1) * input_elements
                    FLOPs += agg_flops
            node['FLOPs'] = FLOPs

        self.params_and_FLOPs_added = True
        self.n_params = self.count_params()
        self.FLOPs = self.count_FLOPs()
    
    def count_params(self):
        assert self.params_and_FLOPs_added, "n_params must be added to count total params"
        return np.sum([node['n_params'] for node in self.vs if node['n_params'] is not None])
    
    def count_FLOPs(self):
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
    
    def sample_node_features(self, input_shape):
        n_nodes = len(self.vs)
        
        def sample_for_node(node_idx, input_shape):
            self.vs[node_idx]["input_shape"] = input_shape
            out_channels = np.random.choice(self.search_space.node_features.out_channels)
            
            possible_strides = [s for s in self.search_space.node_features.stride if input_shape[1] % s == 0]
            stride = np.random.choice(possible_strides)
            
            possible_groups = [g for g in self.search_space.node_features.groups 
                            if input_shape[0] % g == 0 and out_channels % g == 0]
            groups = np.random.choice(possible_groups)
            
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
                    node_features[index] = np.random.choice(feature_values)
            
            self.vs[node_idx]["features"] = node_features.tolist()
            self.vs[node_idx]["output_shape"] = [out_channels, input_shape[1] // stride]
        
        sample_for_node(0, input_shape)
        for i in range(1, n_nodes):
            current_input_shape = self._get_input_shape(i)
            sample_for_node(i, current_input_shape)


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


    def to_blueprint(self, input_shape=[3, 32], num_classes=10):
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
        
        # Calculate output shape for the first node
        node_features = blueprint.vs[node_mapping[0]]["features"]
        stride = node_features[feature_index("stride", self.search_space)]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        blueprint.vs[node_mapping[0]]["output_shape"] = [out_channels, input_shape[1] // stride]
        
        # Process remaining nodes
        for i in range(1, self.vcount()):
            original_predecessors = self.predecessors(i)
            
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
        blueprint.n_params = blueprint.count_params()
        blueprint.FLOPs = blueprint.count_FLOPs()

        return blueprint


    def to_torch(self, input_shape=[3, 32], num_classes=10):
        import torch
        import torch.nn as nn
        
        blueprint = self.to_blueprint(input_shape=input_shape, num_classes=num_classes)
        
        class ArcGraphModel(nn.Module):
            def __init__(self, blueprint):
                super(ArcGraphModel, self).__init__()
                from custom_layers import CustomSE, ChannelPad, AggTensors
                self.layers = nn.ModuleDict()
                self.topology = []
                
                execution_order = []
                visited = set()
                
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
            
            def forward(self, x):
                outputs = {}
                
                for node_idx, predecessors in self.topology:
                    node_type = "original"
                    if "node_type" in blueprint.vs[node_idx].attribute_names():
                        node_type = blueprint.vs[node_idx]["node_type"]
                    
                    if len(predecessors) == 0:
                        if node_type == 'original':
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
                        else:
                            pred_outputs = [outputs[pred_idx] for pred_idx in predecessors]
                            
                            if node_type == 'original' or node_type == 'global_avg_pool':
                                x = self.layers[f'agg_{node_idx}'](torch.stack(pred_outputs))
                            else:
                                raise ValueError(f'{node_type} nodes may not have >1 predecessors, received {len(predecessors)}')
                        
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
                
                return outputs[self.topology[-1][0]]
        
        model = ArcGraphModel(blueprint)
        return model
    

# TODO: count flops related to aggregations, and check if deepseed can detect them
X = np.array([[32, 3, 2, 1, 0, "sum"], [64, 3, 2, -1, 0, "sum"], [128, 3, 1, 1, 0, "sum"]], dtype = "object")
A = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
# X = np.array([[32, 3, 2, 1, 0, "sum"]], dtype = "object")
# A = np.array([[0]])
g = ArcGraph(X=X, A=A)
g.search_space = SearchSpace()
g.is_valid(input_shape = [3, 32], verbose = True)
g.add_latent_shapes([3, 32])
g.add_n_params_and_FlOPs()

g2=g.to_blueprint(input_shape=[3, 32])
# g.plot(display=True)

model = g.to_torch(input_shape=[3, 32])

from torchsummary import summary
from model_profiling import profile_model

summary(model, input_size=(3, 32, 32))
model.eval()
params, flops = profile_model(model, input_shape=(1, 3, 32, 32))
print(f' n_params(custom deepspeed): {params}')
print(f' FLOPs (custom deepspeed): {flops} \n')

print(f' n_params (blueprint): {g2.n_params}')
print(f' FLOPs (blueprint): {g2.FLOPs} \n')

# print(f' n_params (seed): {g.n_params}')
# print(f' FLOPs (seed): {g.FLOPs} \n')

# print([g2.vs[i]['FLOPs'] for i in range(g2.vcount())])
# # Compute the output for a random input
random_input = torch.randn(1, 3, 32, 32)
output = model(random_input)

g2.plot(display=True)