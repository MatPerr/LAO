import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from graph_utils import ArcGraph
from tqdm import tqdm
import numpy as np
from datetime import datetime
import itertools
import multiprocessing as mp

def declare_dir_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# -------------------------------------------------------
# Module-level helper to sample adjacency matrix A.
def sample_A(search_space):
    n_nodes = search_space.graph_features.n_nodes[0]
    A = np.zeros((n_nodes, n_nodes), dtype=int)
    for i in range(1, n_nodes):
        possible_preds = list(range(i))
        max_preds = search_space.graph_features.max_preds
        if search_space.graph_features.traceable:
            selected_preds = np.array([i - 1])
            n_extra_preds = np.random.randint(0, min(i, max_preds))
            if n_extra_preds > 0:
                # Optionally, you could bias selection toward later nodes.
                extra_preds = np.random.choice(possible_preds[:-1],
                                               size=n_extra_preds,
                                               replace=False)
                selected_preds = np.concatenate((extra_preds, selected_preds))
        else:
            n_preds = np.random.randint(0, min(i, max_preds) + 1)
            if n_preds > 0:
                selected_preds = np.random.choice(possible_preds,
                                                  size=n_preds,
                                                  replace=False)
            else:
                selected_preds = np.array([], dtype=int)
        A[selected_preds, i] = 1
    return A

# -------------------------------------------------------
# Worker function to generate one valid sample.
def worker_generate_sample(args):
    """
    Args is a tuple: (search_space, input_shape, constraints)
    This function generates a single valid sample.
    """
    search_space, input_shape, constraints = args
    # Import locally in worker to ensure clean worker state.
    from graph_utils import ArcGraph  # re-import in each process if needed
    import torch

    while True:
        A = sample_A(search_space)
        g = ArcGraph(search_space=search_space, X=None, A=A)
        g.sample_node_features(input_shape, constraints)
        if not g.constraints_met:
            continue
        if not g.is_valid(input_shape):
            print('oooo')
            continue
        g.add_latent_shapes(input_shape)
        g.add_n_params_and_FlOPs()
        blueprint = g.to_blueprint(input_shape=input_shape)
        constraints_met = all([eval(f"blueprint.{key} <= {value[1]} and blueprint.{key} >= {value[0]}") for key, value in constraints.items()])
        if not constraints_met:
            continue
        v = g.to_V()
        Y = torch.tensor([blueprint.n_params, blueprint.FLOPs, blueprint.BBGP], dtype=torch.float)
        return (v, Y)

# -------------------------------------------------------
# DataGenerator class updated to use multiprocessing.
class DataGenerator():
    def __init__(self, search_space, input_shape, exp_dir, constraints=None):
        self.search_space = search_space
        self.input_shape = input_shape
        self.exp_dir = declare_dir_path(exp_dir)
        self.constraints = constraints
    
    def generate_dataset(self, n_samples, num_processes=None):
        """
        Generates n_samples using multiprocessing.
        num_processes: number of worker processes to use.
        """
        # Set number of processes to CPU count if not specified.
        if num_processes is None:
            num_processes = mp.cpu_count()
            
        # Argument tuple to be passed to each worker.
        worker_args = (self.search_space, self.input_shape, self.constraints)
        
        results = []
        # Create a pool of workers.
        with mp.Pool(processes=num_processes) as pool:
            # itertools.repeat replays the same worker_args n_samples times.
            # imap_unordered will yield valid samples as they are ready.
            for sample in tqdm(pool.imap_unordered(worker_generate_sample, itertools.repeat(worker_args, n_samples)), total=n_samples, desc="Sampling valid graphs"):
                results.append(sample)
        
        # Separate the returned samples.
        V_list = [r[0] for r in results]
        Y_list = [r[1] for r in results]
        
        # Stack into tensors.
        V = torch.stack(V_list)
        Y = torch.stack(Y_list)
        
        current_datetime = datetime.now().strftime("%m%d_%H%M")
        save_path = os.path.join(declare_dir_path(os.path.join(self.exp_dir, "graph_data")), f"{n_samples}_samples_{current_datetime}.pt")
        torch.save({'V': V, 'Y': Y}, save_path)
        print(f"Data saved to: {save_path}")

# -------------------------------------------------------
# Dataset and DataLoader definitions (unchanged).
class GraphDataset(Dataset):
    def __init__(self, V, Y):
        """
        Dataset for graph features and targets.
        """
        self.V = V
        self.Y = Y
        assert len(self.V) == len(self.Y), "V and Y must have the same number of samples"

    def __len__(self):
        return len(self.V)

    def __getitem__(self, idx):
        return self.V[idx], self.Y[idx]


class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False, **kwargs):
        """
        A DataLoader that reuses the same iterator for multiple epochs.
        """
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )
        self._DataLoader__initialized = False  # Hack to let us modify the batch_sampler
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


def get_dataloaders(V, Y, train_split=0.99, batch_size=32, num_workers=0):
    full_dataset = GraphDataset(V, Y)
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataloader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_dataloader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader

# # -------------------------------------------------------
# # Example usage:
# from search_space import SearchSpace

# if __name__ == '__main__':
#     # Initialize data generator with the search space, input shape, and experiment directory.
#     datagen = DataGenerator(SearchSpace(), input_shape=[3, 32], exp_dir="exp1404", constraints={'n_params':[0,1e6], 'FLOPs':[0,1e8]})
#     # Generate dataset with multiprocessing (optionally specifying number of processes)
#     datagen.generate_dataset(1_000_000, num_processes=mp.cpu_count())
