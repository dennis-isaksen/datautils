import torch
from torch.utils.data import Dataset
import numpy as np

class MixedDataset(Dataset):
    def __init__(self, ground_truth_dataset, synthetic_dataset, gt_ratio_per_synthetic=1.0):
        """
        Initializes the MixedDataset.

        Args:
            ground_truth_dataset (torch.utils.data.Dataset): The ground truth dataset.
            synthetic_dataset (torch.utils.data.Dataset): The synthetic data dataset.
            gt_ratio_per_synthetic (float): The ratio of ground truth samples per synthetic sample.
                                            If synthetic data has N samples, ground truth will have
                                            N * gt_ratio_per_synthetic samples in the mixed dataset.
        """
        if not isinstance(ground_truth_dataset, Dataset) or not isinstance(synthetic_dataset, Dataset):
            raise TypeError("Both ground_truth_dataset and synthetic_dataset must be subclasses of torch.utils.data.Dataset")

        self.ground_truth_dataset = ground_truth_dataset
        self.synthetic_dataset = synthetic_dataset
        self.gt_ratio_per_synthetic = gt_ratio_per_synthetic

        self._calculate_lengths()

    def _calculate_lengths(self):
        """
        Calculates the number of samples to include from each dataset.
        We base the ground truth count on the synthetic dataset's size.
        """
        self.num_synthetic_samples = len(self.synthetic_dataset)
        self.num_ground_truth_samples = int(self.num_synthetic_samples * self.gt_ratio_per_synthetic)

        # Ensure we don't try to draw more ground truth samples than available
        if self.num_ground_truth_samples > len(self.ground_truth_dataset):
            print(f"Warning: Requested {self.num_ground_truth_samples} GT samples, but only "
                  f"{len(self.ground_truth_dataset)} available. Limiting GT samples to available.")
            self.num_ground_truth_samples = len(self.ground_truth_dataset)

        self.total_samples = self.num_synthetic_samples + self.num_ground_truth_samples

        # Create mappings for indexing
        # Synthetic samples will come first, then ground truth samples
        self.synthetic_indices = list(range(self.num_synthetic_samples))
        self.ground_truth_indices = list(range(self.num_ground_truth_samples))

        # Shuffle ground truth indices to pick random samples if num_ground_truth_samples < total_ground_truth_dataset_size
        np.random.shuffle(self.ground_truth_indices)


    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < self.num_synthetic_samples:
            # Return a sample from the synthetic dataset
            return self.synthetic_dataset[self.synthetic_indices[idx]]
        else:
            # Return a sample from the ground truth dataset
            # Adjust index to be relative to the start of ground truth samples
            gt_idx_in_mixed = idx - self.num_synthetic_samples
            original_gt_idx = self.ground_truth_indices[gt_idx_in_mixed]
            return self.ground_truth_dataset[original_gt_idx]