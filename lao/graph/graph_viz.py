import os
import tempfile
from copy import deepcopy
from typing import Any

import imageio
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pygraphviz as pgv


def plot_graph(
    graph_obj: Any,
    output_path: Any = None,
    backbone: Any = True,
    display: Any = False,
    feature_index_fn: Any = None,
    format_number_spaces_fn: Any = None,
) -> Any:
    """
    Plot graph.

    Args:
        graph_obj (Any): Input parameter.
        output_path (Any): Input parameter.
        backbone (Any): Input parameter.
        display (Any): Input parameter.
        feature_index_fn (Any): Input parameter.
        format_number_spaces_fn (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    graph = pgv.AGraph(directed=True, strict=True, fontname="Helvetica", arrowtype="open")
    cmap = plt.get_cmap("cool")
    for idx in range(graph_obj.vcount()):
        node_type = "original"
        if "node_type" in graph_obj.vs[idx].attribute_names():
            node_type = graph_obj.vs[idx]["node_type"]
        if node_type == "original" and "features" in graph_obj.vs[idx].attribute_names():
            features = graph_obj.vs[idx]["features"]
            if graph_obj.search_space and hasattr(graph_obj.search_space, "aliases"):
                label_parts = []
                feature_names = list(graph_obj.search_space.node_features.__dict__.keys())
                for i, name in enumerate(feature_names):
                    if i < len(features):
                        alias = graph_obj.search_space.aliases.get(name, name)
                        label_parts.append(f"{alias}{features[i]}")
                label = f"({idx})\n" + " ".join(label_parts)
            else:
                label = f"({idx})\n{features}"
            color = "goldenrod"
            fixedsize = False
            width = None
            height = None
            stride = features[feature_index_fn("stride", graph_obj.search_space)]
            shape = "box"
            if stride > 1:
                fixedsize = True
                width = 2.5
                height = 0.9
                shape = "invtrapezium"
            if "n_params" in graph_obj.vs[idx].attribute_names():
                n_params = graph_obj.vs[idx]["n_params"]
                norm = mcolors.LogNorm(vmin=1, vmax=1000000)
                color = mcolors.to_hex(cmap(norm(n_params)))
                label += f"\nparams: {format_number_spaces_fn(n_params)}"
            if "FLOPs" in graph_obj.vs[idx].attribute_names():
                flops = graph_obj.vs[idx]["FLOPs"]
                label += f"\nFLOPs: {format_number_spaces_fn(flops)}"
            graph.add_node(
                idx,
                label=label,
                color="black",
                fillcolor=color,
                shape=shape,
                style="filled",
                fixedsize=fixedsize,
                width=width,
                height=height,
                fontsize=12,
            )
        elif node_type == "breadth_adapter":
            reduc_factor = "?"
            if "reduc_factor" in graph_obj.vs[idx].attribute_names():
                reduc_factor = graph_obj.vs[idx]["reduc_factor"]
            label = f"({idx})\nMaxPool {reduc_factor}"
            graph.add_node(
                idx,
                label=label,
                color="black",
                fillcolor="grey85",
                shape="invtrapezium",
                style="filled",
                width=1.4,
                height=0.5,
                fixedsize=True,
                fontsize=12,
            )
        elif node_type == "channel_adapter":
            padding = "?"
            if "padding" in graph_obj.vs[idx].attribute_names():
                padding = graph_obj.vs[idx]["padding"]
            label = f"({idx})\nChannelPad {padding}"
            graph.add_node(
                idx, label=label, color="black", fillcolor="grey85", shape="box", style="filled", fontsize=12
            )
        elif node_type == "global_avg_pool":
            label = f"({idx})\nGlobalAvgPool"
            if "FLOPs" in graph_obj.vs[idx].attribute_names():
                flops = graph_obj.vs[idx]["FLOPs"]
                label += f"\nFLOPs: {format_number_spaces_fn(flops)}"
            graph.add_node(
                idx,
                label=label,
                color="black",
                fillcolor="grey85",
                shape="invtrapezium",
                style="filled",
                width=2,
                height=0.7,
                fixedsize=True,
                fontsize=12,
            )
        elif node_type == "classifier":
            label = f"({idx})\nLinear"
            if "n_params" in graph_obj.vs[idx].attribute_names():
                n_params = graph_obj.vs[idx]["n_params"]
                label += f"\nparams: {format_number_spaces_fn(n_params)}"
            if "FLOPs" in graph_obj.vs[idx].attribute_names():
                flops = graph_obj.vs[idx]["FLOPs"]
                label += f"\nFLOPs: {format_number_spaces_fn(flops)}"
            graph.add_node(
                idx, label=label, color="black", fillcolor="grey85", shape="box", style="filled", fontsize=12
            )
        else:
            graph.add_node(
                idx, label=f"({idx})", color="black", fillcolor="white", shape="box", style="filled", fontsize=12
            )
    for source, target in graph_obj.get_edgelist():
        input_shape_label = ""
        output_shape_label = ""
        if "output_shape" in graph_obj.vs[source].attribute_names():
            source_output = graph_obj.vs[source]["output_shape"]
            output_shape_label = f" {source_output[0]}×{source_output[1]}²"
        if "input_shape" in graph_obj.vs[target].attribute_names():
            target_input = graph_obj.vs[target]["input_shape"]
            input_shape_label = f" {target_input[0]}×{target_input[1]}²"
        if output_shape_label and input_shape_label:
            label = (
                output_shape_label
                if output_shape_label == input_shape_label
                else f" {output_shape_label}\n{input_shape_label}"
            )
        elif output_shape_label:
            label = output_shape_label
        elif input_shape_label:
            label = input_shape_label
        else:
            label = ""
        edge_weight = 3 if backbone and source == target - 1 else 1
        if label:
            graph.add_edge(
                source, target, weight=edge_weight, label=label, fontsize=12, labeldistance=1.5, labelangle=0
            )
        else:
            graph.add_edge(source, target, weight=edge_weight)
    graph.layout(prog="dot")
    if display:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            temp_path = tmp_file.name
        graph.draw(temp_path, prog="dot", args="-Gdpi=300")
        img = mpimg.imread(temp_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        graph.draw(output_path, prog="dot", args="-Gdpi=300")
        return output_path
    return graph


def memory_usage_animation(
    graph_obj: Any,
    output_path: Any = None,
    delays: Any = 50,
    backbone: Any = True,
    plot_memory: Any = True,
    feature_index_fn: Any = None,
    format_number_spaces_fn: Any = None,
) -> Any:
    """
    Memory usage animation.

    Args:
        graph_obj (Any): Input parameter.
        output_path (Any): Input parameter.
        delays (Any): Input parameter.
        backbone (Any): Input parameter.
        plot_memory (Any): Input parameter.
        feature_index_fn (Any): Input parameter.
        format_number_spaces_fn (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    if not hasattr(graph_obj, "valid") or not graph_obj.valid:
        print("Warning: Creating blueprint from invalid graph for visualization.")
        blueprint = graph_obj
    else:
        blueprint = graph_obj
    node_layers, execution_groups = blueprint._layer_blueprint()
    frames = []
    current_state = {i: "not_computed" for i in range(blueprint.vcount())}
    memory_usage = []
    current_memory = 0

    def calc_tensor_memory(node_idx: Any) -> Any:
        """
        Calc tensor memory.

        Args:
            node_idx (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        if "output_shape" not in blueprint.vs[node_idx].attribute_names():
            return 0
        shape = blueprint.vs[node_idx]["output_shape"]
        return shape[0] * shape[1] ** 2 * 4

    frames.append(deepcopy(current_state))
    memory_usage.append(current_memory)
    sorted_layers = sorted(execution_groups.keys())
    for layer in sorted_layers:
        nodes_in_layer = execution_groups[layer]
        for node in nodes_in_layer:
            current_state[node] = "in_memory"
            current_memory += calc_tensor_memory(node)
        frames.append(deepcopy(current_state))
        memory_usage.append(current_memory)
        freed_nodes = []
        freed_memory = 0
        for node_idx in range(blueprint.vcount()):
            if current_state[node_idx] == "in_memory" and node_idx not in nodes_in_layer:
                is_still_needed = False
                for future_layer in [
                    future_layer_idx for future_layer_idx in sorted_layers if future_layer_idx > layer
                ]:
                    for future_node in execution_groups[future_layer]:
                        if node_idx in blueprint.predecessors(future_node):
                            is_still_needed = True
                            break
                    if is_still_needed:
                        break
                if not is_still_needed:
                    freed_nodes.append(node_idx)
                    freed_memory += calc_tensor_memory(node_idx)
        if freed_nodes:
            for node in freed_nodes:
                current_state[node] = "freed"
            current_memory -= freed_memory
            frames.append(deepcopy(current_state))
            memory_usage.append(current_memory)

    def draw_state(state: Any) -> Any:
        """
        Draw state.

        Args:
            state (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        graph = pgv.AGraph(directed=True, strict=True, fontname="Helvetica", arrowtype="open")
        memory_colors = {"not_computed": "#F5F5DC", "in_memory": "#FFA500", "freed": "#A9A9A9"}
        for idx in range(blueprint.vcount()):
            node_type = "original"
            if "node_type" in blueprint.vs[idx].attribute_names():
                node_type = blueprint.vs[idx]["node_type"]
            fillcolor = memory_colors[state[idx]]
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
                stride = features[feature_index_fn("stride", blueprint.search_space)]
                shape = "box"
                if stride > 1:
                    fixedsize = True
                    width = 2.5
                    height = 0.9
                    shape = "invtrapezium"
                if "n_params" in blueprint.vs[idx].attribute_names():
                    label += f"\nparams: {format_number_spaces_fn(blueprint.vs[idx]['n_params'])}"
                if "FLOPs" in blueprint.vs[idx].attribute_names():
                    label += f"\nFLOPs: {format_number_spaces_fn(blueprint.vs[idx]['FLOPs'])}"
                graph.add_node(
                    idx,
                    label=label,
                    color="black",
                    fillcolor=fillcolor,
                    shape=shape,
                    style="filled",
                    fixedsize=fixedsize,
                    width=width,
                    height=height,
                    fontsize=12,
                )
            elif node_type == "breadth_adapter":
                reduc_factor = (
                    blueprint.vs[idx]["reduc_factor"] if "reduc_factor" in blueprint.vs[idx].attribute_names() else "?"
                )
                graph.add_node(
                    idx,
                    label=f"({idx}) {layer_info}\nMaxPool {reduc_factor}",
                    color="black",
                    fillcolor=fillcolor,
                    shape="invtrapezium",
                    style="filled",
                    width=1.4,
                    height=0.5,
                    fixedsize=True,
                    fontsize=12,
                )
            elif node_type == "channel_adapter":
                padding = blueprint.vs[idx]["padding"] if "padding" in blueprint.vs[idx].attribute_names() else "?"
                graph.add_node(
                    idx,
                    label=f"({idx}) {layer_info}\nChannelPad {padding}",
                    color="black",
                    fillcolor=fillcolor,
                    shape="box",
                    style="filled",
                    fontsize=12,
                )
            elif node_type == "global_avg_pool":
                label = f"({idx}) {layer_info}\nGlobalAvgPool"
                if "FLOPs" in blueprint.vs[idx].attribute_names():
                    label += f"\nFLOPs: {format_number_spaces_fn(blueprint.vs[idx]['FLOPs'])}"
                graph.add_node(
                    idx,
                    label=label,
                    color="black",
                    fillcolor=fillcolor,
                    shape="invtrapezium",
                    style="filled",
                    width=2,
                    height=0.7,
                    fixedsize=True,
                    fontsize=12,
                )
            elif node_type == "classifier":
                label = f"({idx}) {layer_info}\nLinear"
                if "n_params" in blueprint.vs[idx].attribute_names():
                    label += f"\nparams: {format_number_spaces_fn(blueprint.vs[idx]['n_params'])}"
                if "FLOPs" in blueprint.vs[idx].attribute_names():
                    label += f"\nFLOPs: {format_number_spaces_fn(blueprint.vs[idx]['FLOPs'])}"
                graph.add_node(
                    idx, label=label, color="black", fillcolor=fillcolor, shape="box", style="filled", fontsize=12
                )
            else:
                graph.add_node(
                    idx,
                    label=f"({idx}) {layer_info}",
                    color="black",
                    fillcolor=fillcolor,
                    shape="box",
                    style="filled",
                    fontsize=12,
                )
        for source, target in blueprint.get_edgelist():
            input_shape_label = ""
            output_shape_label = ""
            if "output_shape" in blueprint.vs[source].attribute_names():
                source_output = blueprint.vs[source]["output_shape"]
                output_shape_label = f" {source_output[0]}×{source_output[1]}²"
            if "input_shape" in blueprint.vs[target].attribute_names():
                target_input = blueprint.vs[target]["input_shape"]
                input_shape_label = f" {target_input[0]}×{target_input[1]}²"
            if output_shape_label and input_shape_label:
                label = (
                    output_shape_label
                    if output_shape_label == input_shape_label
                    else f" {output_shape_label}\n{input_shape_label}"
                )
            elif output_shape_label:
                label = output_shape_label
            elif input_shape_label:
                label = input_shape_label
            else:
                label = ""
            edge_weight = 3 if backbone and source == target - 1 else 1
            if label:
                graph.add_edge(
                    source, target, weight=edge_weight, label=label, fontsize=12, labeldistance=1.5, labelangle=0
                )
            else:
                graph.add_edge(source, target, weight=edge_weight)
        graph.layout(prog="dot")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            temp_path = tmp_file.name
        graph.draw(temp_path, prog="dot", args="-Gdpi=300")
        return temp_path

    def plot_memory_usage() -> Any:
        """
        Plot memory usage.

        Args:
            None

        Returns:
            Any: Function output.
        """
        plt.figure(figsize=(10, 6))
        memory_mb = [m / (1024 * 1024) for m in memory_usage]
        steps = list(range(len(memory_usage)))
        plt.plot(steps, memory_mb, "b-", linewidth=2)
        plt.fill_between(steps, 0, memory_mb, alpha=0.2, color="blue")
        plt.title("Memory Usage During Parallel Forward Pass", fontsize=14)
        plt.xlabel("Execution Step", fontsize=12)
        plt.ylabel("Memory Usage (MB)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        frame_idx = 0
        for layer in sorted_layers:
            if frame_idx < len(steps):
                plt.axvline(x=frame_idx, color="red", linestyle="--", alpha=0.5)
                plt.text(frame_idx, max(memory_mb) * 1.02, f"Layer {layer}", rotation=90, verticalalignment="bottom")
            frame_idx += 2
        peak_memory = max(memory_mb)
        peak_step = memory_mb.index(peak_memory)
        plt.scatter(peak_step, peak_memory, color="red", s=100, zorder=5)
        plt.annotate(
            f"Peak: {peak_memory:.2f} MB",
            xy=(peak_step, peak_memory),
            xytext=(peak_step + 1, peak_memory * 1.1),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
            fontsize=12,
        )
        plt.tight_layout()
        if output_path:
            memory_plot_path = os.path.splitext(output_path)[0] + "_memory.png"
            plt.savefig(memory_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            return memory_plot_path
        memory_fig = plt.gcf()
        plt.close()
        return memory_fig

    frame_files = []
    print(f"Generating {len(frames)} frames for memory usage animation...")
    for i, state in enumerate(frames):
        if i % 10 == 0:
            print(f"Processing frame {i + 1}/{len(frames)}...")
        frame_files.append(draw_state(state))
    memory_plot_output = None
    if plot_memory:
        print("Generating memory usage plot...")
        memory_plot_output = plot_memory_usage()
    if output_path:
        directory = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(directory, exist_ok=True)
        print(f"Creating animation and saving to {output_path}...")
        with imageio.get_writer(output_path, mode="I", duration=delays / 1000) as writer:
            for file in frame_files:
                image = imageio.imread(file)
                writer.append_data(image)
                try:
                    os.remove(file)
                except OSError:
                    pass
        print("Animation created successfully!")
        return (output_path, memory_plot_output)
    fig, ax = plt.subplots(figsize=(12, 8))

    def update(frame: Any) -> Any:
        """
        Update.

        Args:
            frame (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        ax.clear()
        img = plt.imread(frame_files[frame])
        ax.imshow(img)
        ax.axis("off")
        return ax

    print("Creating animation for display...")
    ani = animation.FuncAnimation(fig, update, frames=len(frame_files), interval=delays)
    plt.close()
    for file in frame_files:
        try:
            os.remove(file)
        except OSError:
            pass
    print("Animation ready for display!")
    return (ani, memory_plot_output)
