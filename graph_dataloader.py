import torch
from torch.utils.data import Dataset, DataLoader, random_split
from graph_utils import ArcGraph
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np

def declare_dir_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

    return path

class DataGenerator():
    def __init__(self, search_space, input_shape, exp_dir):
        self.search_space = search_space
        self.input_shape = input_shape
        self.exp_dir = declare_dir_path(exp_dir)
    
    def sample_A(self):
        n_nodes = self.search_space.graph_features.n_nodes[0]
        A = np.zeros((n_nodes, n_nodes), dtype=int)
        for i in range(1, n_nodes):
            possible_preds = list(range(i))
            max_preds = self.search_space.graph_features.max_preds
            if self.search_space.graph_features.traceable:
                selected_preds = np.array([i-1])
                n_extra_preds = np.random.randint(0, min(i, max_preds)) # Upper bound is exclusive
                if n_extra_preds > 0:
                    # TODO: prefer later nodes as predecessors
                    extra_preds = np.random.choice(
                        possible_preds[:-1], 
                        size=n_extra_preds, 
                        replace=False
                    )
                    selected_preds = np.concatenate((extra_preds, selected_preds))

            else:
                n_preds = np.random.randint(0, min(i, max_preds)+1) # Upper bound is exclusive
                selected_preds = np.random.choice(
                    possible_preds, 
                    size=n_preds, 
                    replace=False
                )
            
            A[selected_preds, i] = 1
        
        return A

    def generate_dataset(self, n_samples):
        pbar = tqdm(range(n_samples))
        V = []
        Y = []
        for _ in pbar:
            A = self.sample_A()
            g = ArcGraph(search_space=self.search_space, X=None, A=A)
            g.sample_node_features(self.input_shape)
            g.is_valid(self.input_shape)
            g.add_latent_shapes(self.input_shape)
            g.add_n_params_and_FlOPs()
            v = g.to_V()
            n_params = g.count_params()
            FLOPs = g.count_FLOPs()
            V.append(v)
            Y.append(torch.Tensor([n_params, FLOPs]))
            pbar.set_description(f"Sampling valid graphs...")

        V = torch.stack(V)
        Y = torch.stack(Y)
        current_datetime = datetime.now().strftime("%m%d_%H%M")
        path = os.path.join(declare_dir_path(os.path.join(self.exp_dir, "graph_data")), f"{n_samples}_samples_{current_datetime}.pt")
        torch.save({'V': V,'Y': Y}, path)
    
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
        # Replace the batch sampler with a repeat sampler.
        self._DataLoader__initialized = False  # Hack to let us modify the batch_sampler
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        # Initialize the persistent iterator.
        self.iterator = super().__iter__()

    def __len__(self):
        # Number of batches per epoch equals the length of the underlying sampler
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class GraphDataset(Dataset):
    def __init__(self, V, Y):
        """
        Dataset for graph features and targets
        
        Args:
            V (torch.Tensor): Graph feature tensors
            Y (torch.Tensor): Target values (n_params, FLOPs)
        """
        self.V = V
        self.Y = Y
        
        # Verify that V and Y have the same number of samples
        assert len(self.V) == len(self.Y), "V and Y must have the same number of samples"

    def __len__(self):
        return len(self.V)

    def __getitem__(self, idx):
        return self.V[idx], self.Y[idx]


def get_dataloaders(V, Y, train_split=0.99, batch_size=32, num_workers=0):
    """
    Create train and validation dataloaders from V and Y tensors
    
    Args:
        V (torch.Tensor): Graph feature tensors
        Y (torch.Tensor): Target values
        train_split (float): Proportion of data to use for training
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
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

# from search_space import *

# datagen = DataGenerator(SearchSpace(), input_shape = [32, 3], exp_dir="exp1903")
# datagen.generate_dataset(1_000_000)