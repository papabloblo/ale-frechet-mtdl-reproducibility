
from __future__ import annotations
from typing import Iterator, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset

from MultiTaskDeepLearning.dataset import MultitaskDataset


class MultitaskDataloader:
    """
    Wrapper around torch.utils.data.DataLoader that permutes batch shape from
    (batch, tasks, ...) to (tasks, batch, ...), convenient for multitask models.

    Parameters
    ----------
    dataset : Dataset
        A dataset that yields (X, Y) where each is shaped (B, T, ...), i.e.,
        batch-first with a task axis in dim=1. `X` may be None for certain datasets.
    permute : bool, default=True
        If True, permutes (B, T, ...) -> (T, B, ...). If False, leaves batches as-is.
    *args, **kwargs :
        Passed through to `torch.utils.data.DataLoader` (e.g., batch_size, shuffle,
        num_workers, pin_memory, collate_fn, etc.)

    Attributes
    ----------
    dataloader : DataLoader
        The underlying PyTorch DataLoader.
    permute : bool
        Whether permutation is applied on iteration.
    """

    def __init__(
            self,
            dataset: MultitaskDataset,
            *args,
            permute: bool = True,
            **kwargs
    ) -> None:
        self._dataset = dataset
        self._dataloader = DataLoader(dataset=dataset, *args, **kwargs)
        self.permute = permute
        self._iter: Optional[Iterator] = None


    # -------- iteration protocol --------
    def __iter__(self) -> "MultitaskDataloader":
        self._iter = iter(self._dataloader)
        return self

    def __next__(self) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if self._iter is None:
            # allow calling next() without explicit iter()
            self._iter = iter(self._dataloader)

        X, Y = next(self._iter)

        if not self.permute:
            return X, Y
        # Only permute the first two dimensions, keeping the rest unchanged.
        X_dims_to_permute = tuple([1, 0] + list(range(2, X.dim())))
        Y_dims_to_permute = tuple([1, 0] + list(range(2, Y.dim())))
        return torch.permute(X, X_dims_to_permute), torch.permute(Y, Y_dims_to_permute)

    # -------- passthroughs / niceties --------
    def __len__(self) -> int:
        return len(self._dataloader)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(permute={self.permute}, dataloader={self._dataloader})"

    # -------- internal helpers --------
    @staticmethod
    def _maybe_permute(t: Optional[torch.Tensor], which: str) -> Optional[torch.Tensor]:
        """
        Permute (B, T, ...) -> (T, B, ...) if t is not None.

        Raises
        ------
        TypeError
            If `t` is not a Tensor (and not None).
        ValueError
            If `t.dim() < 2` (no batch+task axes to swap).
        """
        if t is None:
            return None
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{which} must be a Tensor or None, got {type(t)!r}")
        if t.dim() < 2:
            raise ValueError(
                f"{which} must have at least 2 dims (B,T,...) to permute; got shape {tuple(t.shape)}"
            )
        # swap first two axes, keep the rest unchanged
        dims = (1, 0, *range(2, t.dim()))
        return t.permute(dims)

def create_dataloaders(
        datasets: dict,
        batch_size_train: int,
        batch_size_ale: int,
        batch_size_test: int,
        num_workers: int = 1,
        device: str = "cpu"
) -> dict:
    """
    Creates and returns DataLoaders for training, ALE (Accumulated Local Effects), and testing.

    Args:
        datasets (dict): Dictionary containing datasets with keys 'train', 'ale', and 'test'.
        batch_size_train (int): Batch size for training DataLoader.
        batch_size_ale (int): Batch size for ALE DataLoader.
        batch_size_test (int): Batch size for testing DataLoader.
        num_workers (int, optional): Number of worker processes. Defaults to NUM_WORKERS.
        device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        dict: A dictionary containing DataLoaders for training, ALE, and testing.
    """

    train_dataloader = MultitaskDataloader(
        datasets['train'],
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        device=device
    )

    ale_dataloader = MultitaskDataloader(
        datasets['ale'],
        batch_size=batch_size_ale,
        shuffle=True,
        num_workers=num_workers,
        device=device
    )

    test_dataloader = MultitaskDataloader(
        datasets['test'],
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        device=device
    )

    return dict(train=train_dataloader, test=test_dataloader, ale=ale_dataloader)
