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

class arc_graph(ig.Graph):
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
    
    def plot(self, output_path=None, backbone=True, show_features=True, display=False):
        graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
        cmap = plt.get_cmap('cool')
        
        for idx in range(self.vcount()):
            if show_features and "features" in self.vs[idx].attribute_names():
                features = self.vs[idx]["features"]
                
                # If we have search_space, we can use aliases for better readability
                if self.search_space and hasattr(self.search_space, "aliases"):
                    label_parts = []
                    
                    # Get feature names and use aliases when available
                    feature_names = list(self.search_space.node_features.__dict__.keys())
                    
                    for i, name in enumerate(feature_names):
                        if i < len(features):
                            alias = self.search_space.aliases.get(name, name)
                            label_parts.append(f"{alias}{features[i]}")
                    
                    label = f"({idx})" + "\n" + " ".join(label_parts)
                else:
                    # Simple display of features
                    label = f"({idx})\n{features}"
            else:
                label = f"({idx})"
            
            color = "goldenrod"
            if "n_params" in self.vs[idx].attribute_names():
                n_params = self.vs[idx]["n_params"]
                norm =  mcolors.LogNorm(vmin=1, vmax=1_000_000) 
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
                shape='box',
                style='filled', 
                fontsize=12
            )
        
        for idx in range(self.vcount()):
            for source in self.neighbors(idx, mode="in"):
                edge_weight = 1
                if backbone and source == idx - 1:
                    edge_weight = 3  
                graph.add_edge(source, idx, weight=edge_weight)        
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
    
    def get_input_shape(self, i):
        all_input_shapes = np.array([self.vs[x]['output_shape'] for x in self.predecessors(i)])
        # The breadth is the smallest breadth among predecessing latents. The larger ones are downsampled
        input_breadth = all_input_shapes[:,0].min()
        input_channels = all_input_shapes[:,1].max()

        return [input_breadth, input_channels]
    
    def get_output_shape(self, i):
        all_input_shapes = np.array([self.vs[x]['output_shape'] for x in self.predecessors(i)])
        # The breadth is the smallest breadth among predecessing latents. The larger ones are downsampled
        input_breadth = all_input_shapes[:,0].min()
        node_features = self.vs[i]["features"]
        stride = node_features[feature_index("stride", self.search_space)]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        return [input_breadth // stride, out_channels] 

    def add_latent_shapes(self, input_shape):
        assert self.search_space is not None, "search_space must be provided to propagate input shape"
        assert self.valid, "the graph must be valid to propagate input shape"
        self.vs[0]["input_shape"] = input_shape

        node_features = self.vs[0]["features"]
        stride = node_features[feature_index("stride", self.search_space)]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        input_breadth = input_shape[0]
        self.vs[0]["output_shape"] = [input_breadth // stride, out_channels]

        for i in range(1, self.vcount()):
            self.vs[i]["input_shape"] = self.get_input_shape(i)
            self.vs[i]["output_shape"] = self.get_output_shape(i)

        self.shapes_added = True     

    def add_n_params_and_FlOPs(self):
        assert self.search_space is not None, "search_space must be provided to count node params"
        assert self.valid, "the graph must be valid to count node params"
        assert self.shapes_added, "the latent shapes must be added to count node params"
        for node in self.vs:
            input_breadth, input_channels = node['input_shape']
            node_features = node["features"]
            out_channels = node_features[feature_index("out_channels", self.search_space)]
            kernel_size = node_features[feature_index("kernel_size", self.search_space)]
            stride = node_features[feature_index("stride", self.search_space)]
            groups = node_features[feature_index("groups", self.search_space)]
            if groups == -1:
                groups = input_channels
            # Assumes padding = 'same'
            node['n_params'] = out_channels*((kernel_size**2)*(input_channels // groups) + 1)
            node['FLOPs'] = (2*(input_breadth**2)*(kernel_size**2)*input_channels*out_channels) // ((stride**2)*groups)
        self.params_and_FLOPs_added = True
        self.n_params = self.count_params()
        self.FLOPs = self.count_FLOPs()
    
    def count_params(self):
        assert self.params_and_FLOPs_added, "n_params must be added to count total params"
        return np.sum([node['n_params'] for node in self.vs if node['n_params'] is not None])
    
    def count_FLOPs(self):
        assert self.params_and_FLOPs_added, "FLOPs must be added to count total FLOPs"
        return np.sum([node['FLOPs'] for node in self.vs if node['FLOPs'] is not None])
        
    def is_node_valid(self, i, verbose):
        node_features = self.vs[i]["features"]
        stride = node_features[feature_index("stride", self.search_space)]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        groups = node_features[feature_index("groups", self.search_space)]
        input_shape = self.vs[i]["input_shape"]
        input_breadth, input_channels = input_shape[0], input_shape[1]

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
        if not self.is_node_valid(0, verbose):
            return False
        node_features = self.vs[0]["features"]
        stride = node_features[feature_index("stride", self.search_space)]
        out_channels = node_features[feature_index("out_channels", self.search_space)]
        input_breadth = input_shape[0]
        self.vs[0]["output_shape"] = [input_breadth // stride, out_channels]

        for i in range(1, self.vcount()):
            self.vs[i]["input_shape"] = self.get_input_shape(i)
            if not self.is_node_valid(i, verbose):
                return False
            self.vs[i]["output_shape"] = self.get_output_shape(i)

        self.valid = True
        return True
    
    def sample_node_features(self, input_shape):
        n_nodes = len(self.vs)
        
        def sample_for_node(node_idx, input_shape):
            self.vs[node_idx]["input_shape"] = input_shape
            out_channels = np.random.choice(self.search_space.node_features.out_channels)
            
            possible_strides = [s for s in self.search_space.node_features.stride if input_shape[0] % s == 0]
            stride = np.random.choice(possible_strides)
            
            possible_groups = [g for g in self.search_space.node_features.groups 
                            if input_shape[1] % g == 0 and out_channels % g == 0]
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
            self.vs[node_idx]["output_shape"] = [input_shape[0] // stride, out_channels]
        
        sample_for_node(0, input_shape)
        for i in range(1, n_nodes):
            current_input_shape = self.get_input_shape(i)
            sample_for_node(i, current_input_shape)

    def make_valid(self):
        pass

    def to_blueprint(self):
        pass

    def to_torch(self):
        pass
    

# X = np.array([[32, 3, 1, 1, 0, "sum"], [64, 3, 1, 1, 0, "sum"], [128, 3, 1, 1, 0, "sum"]], dtype = "object")
# A = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
# g = arc_graph(X=X, A=A)
# g.search_space = SearchSpace()
# print(g.is_valid(input_shape = [32,3], verbose = True))
# g.add_latent_shapes([32, 3])
# g.add_n_params_and_FlOPs()
# print(g.count_params())
# print(g.count_FLOPs())
# g.plot(display=True, show_features=True)