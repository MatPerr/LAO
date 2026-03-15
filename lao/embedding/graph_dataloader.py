import itertools
import multiprocessing as mp
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from lao.graph.graph_utils import ArcGraph, constraints_satisfied


def declare_dir_path(path: str) -> Any:
    """
    Declare dir path.

    Args:
        path (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def sample_A(search_space: Any) -> Any:
    """
    Sample a.

    Args:
        search_space (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    n_nodes = search_space.graph_features.n_nodes[0]
    A = np.zeros((n_nodes, n_nodes), dtype=int)
    for i in range(1, n_nodes):
        possible_preds = list(range(i))
        max_preds = search_space.graph_features.max_preds
        if search_space.graph_features.traceable:
            selected_preds = np.array([i - 1])
            n_extra_preds = np.random.randint(0, min(i, max_preds))
            if n_extra_preds > 0:
                extra_preds = np.random.choice(possible_preds[:-1], size=n_extra_preds, replace=False)
                selected_preds = np.concatenate((extra_preds, selected_preds))
        else:
            n_preds = np.random.randint(0, min(i, max_preds) + 1)
            if n_preds > 0:
                selected_preds = np.random.choice(possible_preds, size=n_preds, replace=False)
            else:
                selected_preds = np.array([], dtype=int)
        A[selected_preds, i] = 1
    return A


def worker_generate_sample(args: Any) -> Any:
    """
    Worker generate sample.

    Args:
        args (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    search_space, input_shape, constraints = args
    import torch

    while True:
        A = sample_A(search_space)
        g = ArcGraph(search_space=search_space, X=None, A=A)
        g.sample_node_features(input_shape, constraints)
        if not g.constraints_met:
            continue
        if not g.is_valid(input_shape):
            print("oooo")
            continue
        g.add_latent_shapes(input_shape)
        g.add_n_params_and_FlOPs()
        blueprint = g.to_blueprint(input_shape=input_shape)
        constraints_met = constraints_satisfied(blueprint, constraints)
        if not constraints_met:
            continue
        v = g.to_V()
        Y = torch.tensor([blueprint.n_params, blueprint.FLOPs, blueprint.BBGP], dtype=torch.float)
        return (v, Y)


class DataGenerator:
    def __init__(self, search_space: Any, input_shape: Any, exp_dir: Any, constraints: Any = None) -> None:
        """
        Init.

        Args:
            search_space (Any): Input parameter.
            input_shape (Any): Input parameter.
            exp_dir (Any): Input parameter.
            constraints (Any): Input parameter.
        """
        self.search_space = search_space
        self.input_shape = input_shape
        self.exp_dir = declare_dir_path(exp_dir)
        self.constraints = constraints

    def generate_dataset(self, n_samples: Any, num_processes: Any = None) -> Any:
        """
        Generate dataset.

        Args:
            n_samples (Any): Input parameter.
            num_processes (Any): Input parameter.

        Returns:
            Any: Function output.
        """
        if num_processes is None:
            num_processes = mp.cpu_count()
        worker_args = (self.search_space, self.input_shape, self.constraints)
        results = []
        with mp.Pool(processes=num_processes) as pool:
            for sample in tqdm(
                pool.imap_unordered(worker_generate_sample, itertools.repeat(worker_args, n_samples)),
                total=n_samples,
                desc="Sampling valid graphs",
            ):
                results.append(sample)
        V_list = [r[0] for r in results]
        Y_list = [r[1] for r in results]
        V = torch.stack(V_list)
        Y = torch.stack(Y_list)
        current_datetime = datetime.now().strftime("%m%d_%H%M")
        save_path = os.path.join(
            declare_dir_path(os.path.join(self.exp_dir, "graph_data")), f"{n_samples}_samples_{current_datetime}.pt"
        )
        torch.save({"V": V, "Y": Y}, save_path)
        print(f"Data saved to: {save_path}")


class GraphDataset(Dataset):
    def __init__(self, V: Any, Y: Any) -> None:
        """
        Init.

        Args:
            V (Any): Input parameter.
            Y (Any): Input parameter.
        """
        self.V = V
        self.Y = Y
        assert len(self.V) == len(self.Y), "V and Y must have the same number of samples"

    def __len__(self) -> Any:
        """Len."""
        return len(self.V)

    def __getitem__(self, idx: Any) -> Any:
        """Getitem."""
        return (self.V[idx], self.Y[idx])


class _RepeatSampler(object):
    def __init__(self, sampler: Any) -> None:
        """Init."""
        self.sampler = sampler

    def __iter__(self) -> Any:
        """
        Iter.

        Args:
            None

        Returns:
            Any: Function output.
        """
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Any,
        batch_size: Any = 32,
        shuffle: Any = True,
        num_workers: Any = 0,
        pin_memory: Any = False,
        **kwargs: Any,
    ) -> None:
        """
        Init.

        Args:
            dataset (Any): Input parameter.
            batch_size (Any): Input parameter.
            shuffle (Any): Input parameter.
            num_workers (Any): Input parameter.
            pin_memory (Any): Input parameter.
            **kwargs (Any): Variable keyword arguments.
        """
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, **kwargs
        )
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self) -> Any:
        """Len."""
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Any:
        """
        Iter.

        Args:
            None

        Returns:
            Any: Function output.
        """
        for _ in range(len(self)):
            yield next(self.iterator)


def get_dataloaders(V: Any, Y: Any, train_split: Any = 0.99, batch_size: Any = 32, num_workers: Any = 0) -> Any:
    """
    Get dataloaders.

    Args:
        V (Any): Input parameter.
        Y (Any): Input parameter.
        train_split (Any): Input parameter.
        batch_size (Any): Input parameter.
        num_workers (Any): Input parameter.

    Returns:
        Any: Function output.
    """
    full_dataset = GraphDataset(V, Y)
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_dataloader = MultiEpochsDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = MultiEpochsDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return (train_dataloader, val_dataloader)
