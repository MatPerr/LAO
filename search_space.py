from dataclasses import dataclass, field
from typing import Dict, List, Literal, Union

@dataclass
class NodeFeatures:
    out_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 512, 1024])
    kernel_size: List[int] = field(default_factory=lambda: [1, 3, 5])
    stride: List[int] = field(default_factory=lambda: [1, 2])
    groups: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, -1])
    # squeeze_excitation: List[int] = field(default_factory=lambda: [0, 1])
    # aggregation: List = field(default_factory=lambda: ["sum", "gate"])

@dataclass
class NodeFeatureProbs:
    """Probabilities for sampling node features"""
    out_channels: Union[List[float], str] = "uniform"
    kernel_size: Union[List[float], str] = "uniform"
    stride: Union[List[float], str] = field(default_factory=lambda: [17/20, 3/20])
    groups: Union[List[float], str] = "uniform"
    # squeeze_excitation: Union[List[float], str] = "uniform" 
    # aggregation: Union[List[float], str] = "uniform"

@dataclass
class GraphFeatures:
    n_nodes: List[int] = field(default_factory=lambda: [20])
    max_preds: int = 2
    traceable: bool = True
    node_encoding_type: Literal["categorical", "quantized"] = "quantized"


@dataclass
class SearchSpace:
    node_features: NodeFeatures = field(default_factory=NodeFeatures)
    node_feature_probs: NodeFeatureProbs = field(default_factory=NodeFeatureProbs)
    graph_features: GraphFeatures = field(default_factory=GraphFeatures)
    aliases: Dict[str, str] = field(default_factory=lambda: {
        "out_channels": "OC",
        "kernel_size": "K",
        "stride": "S",
        "groups": "G",
        "squeeze_excitation": "SE",
        "aggregation": "",
    })