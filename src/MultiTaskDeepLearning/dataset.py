from __future__ import annotations

from typing import List, Tuple, Sequence, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import os
from PIL import Image
from torchvision import transforms


class MultitaskDataset(Dataset):
    """
    A PyTorch Dataset for tabular multi-task learning where each row belongs to a task,
    identified by a task-id column. For each ``__getitem__(i)``, it returns a stacked
    sample across all tasks by cycling modulo each task length (balanced sampling).

    Parameters
    ----------
    data : pd.DataFrame
        Source dataframe. It will not be mutated.
    task_id : str
        Column name that identifies the task.
    target_names : Sequence[str]
        Column names for targets (one or more). Must be non-empty.
    device : str, default="cpu"
        Device for returned tensors ("cpu" or "cuda"). Prefer CPU and move to device in
        the training loop for better DataLoader performance.
    dtype : torch.dtype, default=torch.float32
        Dtype for feature tensors.
    y_dtype : torch.dtype | None, default=None
        Dtype for target tensors. Defaults to ``dtype`` if ``None``. Use ``torch.long`` for
        classification targets.
    presplit_X : bool, default=True
        If True, pre-materialize per-task feature tensors (fast for tabular). If False,
        ``__getitem__`` in subclasses may provide X differently (e.g., image loading).
    idx_column : str | None, default=None
        Optional column whose values (e.g., image filenames) are kept per task and can be
        consumed by subclasses.

    Attributes
    ----------
    task_id : str
        Name of the task-id column.
    target_names : list[str]
        List of target column names used to build ``Y`` tensors.
    idx_column : str | None
        Optional column whose values are preserved per task in ``idx_`` (e.g., filenames).
    presplit_X : bool
        Whether features are precomputed and stored in-memory per task.
    dtype : torch.dtype
        Dtype used for feature tensors ``X``.
    y_dtype : torch.dtype
        Dtype used for target tensors ``Y``.
    label_encoder_ : sklearn.preprocessing.LabelEncoder
        Encoder mapping original task ids to normalized ids ``[0..n_tasks-1]``.
    tasks_ : np.ndarray of shape (n_tasks,)
        Unique normalized task ids.
    task_counts_ : np.ndarray of shape (n_tasks,)
        Number of rows per normalized task id.
    n_tasks : int
        Total number of tasks.
    feature_names_ : list[str]
        Names of feature columns used to build ``X``.
    X : list[torch.Tensor]
        Per-task feature tensors with shapes ``[(n_i, n_features)]`` when ``presplit_X=True``;
        otherwise an empty list.
    Y : list[torch.Tensor]
        Per-task target tensors with shapes ``[(n_i, n_targets)]``.
    idx_ : list[np.ndarray]
        Per-task arrays with values from ``idx_column`` when provided; empty if ``idx_column`` is ``None``.
    task_id_map_ : dict[Any, int]
        Mapping from original task id -> normalized id.
    task_id_inverse_map_ : dict[int, Any]
        Mapping from normalized id -> original task id.
    n_features : int
        Read-only property: number of feature columns.
    n_targets : int
        Read-only property: number of target columns.

    Raises
    ------
    TypeError
        If inputs have incorrect types.
    ValueError
        If dataframe is empty, required columns are missing, or invalid dtypes are found.

    """

    def __init__(
            self,
            data: pd.DataFrame,
            task_id: str,
            target_names: Sequence[str],
            device: str = "cpu",
            dtype: torch.dtype = torch.float32,
            y_dtype: torch.dtype | None = None,
            presplit_X: bool = True,
            idx_column: Optional[str] = None
    ) -> None:

        # ---- basic validation ----
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            # required by tests: test_empty_dataframe_raises_gracefully
            raise ValueError("No tasks found")

        if not isinstance(task_id, str):
            raise TypeError("task_id must be a str column name")
        if task_id not in data.columns:
            raise ValueError(f"task_id='{task_id}' not found in columns.")

        if not target_names:
            raise ValueError("target_names must be a non-empty sequence.")
        missing_targets = [t for t in target_names if t not in data.columns]
        if missing_targets:
            raise ValueError(f"Targets not found in columns: {missing_targets}")

        if data[task_id].isna().any():
            raise ValueError(f"NaN values found in task_id column '{task_id}'.")

        if idx_column is not None and idx_column not in data.columns:
            raise ValueError(f"idx_column '{idx_column}' not found in columns.")

        self.dtype = dtype
        self.y_dtype = dtype if y_dtype is None else y_dtype

        self.task_id = task_id
        self.target_names = list(target_names)
        self.idx_column = idx_column
        self.presplit_X = presplit_X

        # Fit label encoder on original task column (stable mapping)
        self.label_encoder_ = self._create_task_id_encoder(data)

        # Compute normalized task ids once (0..n_tasks-1)
        normalized_task_ids = self.label_encoder_.transform(data[self.task_id].to_numpy())
        self.tasks_, self.task_counts_ = np.unique(normalized_task_ids, return_counts=True)
        self.n_tasks = int(len(self.tasks_))

        # Build feature name list efficiently
        excluded = set(self.target_names) | {self.task_id}
        if self.idx_column is not None:
            excluded.add(self.idx_column)
        self.feature_names_ = [c for c in data.columns if c not in excluded]

        if self.presplit_X and len(self.feature_names_) == 0:
            raise ValueError("No feature columns found after removing targets and task_id.")

        # Pre-split per task into tensors/lists
        self.X, self.Y, self.idx_ = self._create_task_tensors(
            data=data,
            normalized_task_ids=normalized_task_ids,
            dtype=self.dtype,
            y_dtype=self.y_dtype,
            device=device,
        )

        # Useful maps
        original_ids = self.label_encoder_.classes_.tolist()
        normalized_ids = list(range(len(original_ids)))
        self.task_id_map_: Dict[Any, int] = dict(zip(original_ids, normalized_ids))
        self.task_id_inverse_map_: Dict[int, Any] = dict(zip(normalized_ids, original_ids))

    # ------------- helpers -------------

    def _create_task_id_encoder(self, data: pd.DataFrame) -> LabelEncoder:
        """
        Encodes arbitrary task IDs into sequential natural numbers.

        Returns:
            LabelEncoder: A fitted LabelEncoder instance.
        """
        le = LabelEncoder()
        # Ensure 1D array-like is passed
        le.fit(data[self.task_id].to_numpy())
        return le

    def _create_task_tensors(
            self,
            data: pd.DataFrame,
            normalized_task_ids: np.ndarray,
            dtype: torch.dtype,
            y_dtype: torch.dtype | None,
            device: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[np.ndarray]]:

        # Validate dtypes only for columns we actually consume
        if self.feature_names_:
            bad_feats = [c for c in self.feature_names_ if not pd.api.types.is_numeric_dtype(data[c])]
            if bad_feats:
                raise TypeError(
                    f"Non-numeric feature columns: {bad_feats}. Encode/cast before using the dataset."
                )
        bad_targets = [c for c in self.target_names if not pd.api.types.is_numeric_dtype(data[c])]
        if bad_targets:
            raise TypeError(
                f"Non-numeric target columns: {bad_targets}. Encode/cast before using the dataset."
            )

        cols_to_check = list(self.target_names)
        if self.feature_names_:
            cols_to_check = self.feature_names_ + cols_to_check
        if data[cols_to_check].isna().any().any():
            raise ValueError("NaNs found in features/targets. Impute or drop before creating the dataset.")

        # Defensive copy; don't mutate user's df
        df = data.copy()
        df[self.task_id] = normalized_task_ids

        X_list: List[torch.Tensor] = []
        Y_list: List[torch.Tensor] = []
        idx_list: List[np.ndarray] = []

        # Group by normalized task id and convert each to tensors
        for task in self.tasks_:
            df_task = df.loc[df[self.task_id] == task]

            if self.presplit_X and self.feature_names_:
                X_np = df_task[self.feature_names_].to_numpy()
                X_task = torch.tensor(X_np, dtype=dtype, device=device)
                X_list.append(X_task)

            Y_np = df_task[self.target_names].to_numpy()
            if Y_np.ndim == 1:
                Y_np = Y_np.reshape(-1, 1)
            Y_task = torch.tensor(Y_np, dtype=y_dtype, device=device)
            Y_list.append(Y_task)

            if self.idx_column is not None:
                idx_list.append(df_task[self.idx_column].to_numpy())

        return X_list, Y_list, idx_list

    # ------------- public API -------------

    def get_original_task_id(self, normalized_ids: Sequence[int]) -> List[object]:
        """
        Map normalized task ids (0..n_tasks-1) back to original labels.
        """
        return self.label_encoder_.inverse_transform(np.asarray(normalized_ids)).tolist()

    def get_task_counts_by_original(self) -> Dict[Any, int]:
        """Convenience: counts per *original* task id."""
        counts = {}
        for k, v in self.task_id_inverse_map_.items():
            counts[self.task_id_inverse_map_[k]] = int(self.task_counts_[k])
        return counts

    def __len__(self) -> int:
        """
        Length is the maximum task frequency. We cycle modulo each task's own length
        in ``__getitem__`` to produce balanced per-index stacks across tasks.
        """
        return int(self.task_counts_.max())

    def __getitem__(self, idx: int) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Returns a *stacked* sample across all tasks for a given global index.

        Parameters
        ----------
        idx : int
            Global index. Each task is indexed as ``idx % len(task_i)``.

        Returns
        -------
        (X, Y) : tuple[Optional[torch.Tensor], torch.Tensor]
            ``X`` shape: (n_tasks, n_features) when ``presplit_X=True``; otherwise ``None``.
            ``Y`` shape: (n_tasks, n_targets)
        """
        X_stack: Optional[torch.Tensor]
        if self.presplit_X and self.X:
            X_stack = torch.stack([X[idx % n] for X, n in zip(self.X, self.task_counts_)])
        else:
            X_stack = None

        Y_stack = torch.stack([Y[idx % n] for Y, n in zip(self.Y, self.task_counts_)])
        return X_stack, Y_stack

    # ------------- convenience metadata -------------

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)

    @property
    def n_targets(self) -> int:
        return len(self.target_names)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_tasks={self.n_tasks}, n_features={self.n_features}, "
            f"n_targets={self.n_targets}, len={len(self)})"
        )


class MultitaskDatasetDF(MultitaskDataset):
    """
    Thin wrapper around :class:`MultitaskDataset` for tabular data with pre-split X.


    Parameters
    ----------
    data : pd.DataFrame
        Source dataframe. It will not be mutated.
    task_id : str
        Column name that identifies the task.
    target_names : Sequence[str]
        Column names for targets (one or more). Must be non-empty.
    device : str, default="cpu"
        Device for returned tensors ("cpu" or "cuda").
    dtype : torch.dtype, default=torch.float32
        Dtype for feature tensors.
    y_dtype : torch.dtype | None, default=None
        Dtype for target tensors. Defaults to ``dtype`` if ``None``.


    Attributes
    ----------
    All attributes from :class:`MultitaskDataset` apply. In particular:


    presplit_X : bool
        Set to ``True``. Features are precomputed and stored per task in-memory.
    X : list[torch.Tensor]
        Per-task feature tensors with shapes ``[(n_i, n_features)]``.
    Y : list[torch.Tensor]
        Per-task target tensors with shapes ``[(n_i, n_targets)]``.
    feature_names_ : list[str]
        Names of feature columns inferred from ``data`` (excluding ``task_id``, targets, and any ``idx_column``).
    n_tasks : int
        Number of tasks present in ``data``.
    """

    def __init__(
            self,
            data: pd.DataFrame,
            task_id: str,
            target_names: Sequence[str],
            device: str = "cpu",
            dtype: torch.dtype = torch.float32,
            y_dtype: torch.dtype | None = None
    ) -> None:

        super().__init__(
            data=data,
            task_id=task_id,
            target_names=target_names,
            device=device,
            dtype=dtype,
            y_dtype=y_dtype,
            presplit_X=True,
            idx_column=None
        )

class MultitaskDatasetImg(MultitaskDataset):
    """
    A dataset for loading *images* and their corresponding labels for multitask learning.

    Assumes image files are shared across all tasks; each task provides its own labels.

    Parameters
    ----------
    img_dir : str
        Directory containing images.
    img_data : pd.DataFrame
        DataFrame with one row per example, including a filename column and target columns.
    col_img_file : str, default='img_file'
        Column in ``img_data`` with the image filename.
    col_task_id : str, default='task_id'
        Column in ``img_data`` with the task identifier.
    transform : callable | None
        Optional transform applied to the PIL image *before* conversion to tensor.
        If the transform already returns a tensor, it's used as-is; otherwise the
        image is converted via ``transforms.ToTensor()``.

    Attributes
    ----------
    img_dir : str
        Root directory where images are read from.
    transform : callable | None
        Optional transform applied to input images prior to stacking.
    idx_ : list[np.ndarray]
        Per-task arrays of image filenames (from ``col_img_file``).
    presplit_X : bool
        Set to ``False``. Features are **not** precomputed; images are loaded on-the-fly in ``__getitem__``.
    X : list
        Empty list (images are not cached in-memory by default).
    Y : list[torch.Tensor]
        Per-task target tensors with shapes ``[(n_i, n_targets)]``.
    n_tasks : int
        Number of tasks present in ``img_data``.


    Notes
    -----
    When stacking images, all images for a given index must share the same spatial size (C, H, W).
    Ensure your ``transform`` enforces a fixed resize to avoid a shape-mismatch error.
   """

    def __init__(
            self,
            img_dir: str,
            img_data: pd.DataFrame,
            col_img_file: str = 'img_file',
            col_task_id: str = 'task_id',
            transform=None,
            *args,
            **kwargs
    ):
        """
        Initializes the dataset with image directory and annotation file.

        Args:
            img_dir (str): Directory where images are stored.
            annotations_file (str): Path to the CSV file containing image names and labels.
        """

        super().__init__(
            data=img_data,
            target_names=list(set(img_data.columns) - {col_img_file, col_task_id}),
            presplit_X=False,
            task_id=col_task_id,
            idx_column=col_img_file,
            *args,
            **kwargs
        )

        self.img_dir = img_dir
        self.transform = transform

    def _img_import(self, img_file: str) -> torch.Tensor:
        img_path = os.path.join(self.img_dir, img_file)
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        # Apply user transform first (usually includes resize/augmentations)
        if self.transform is not None:
            img = self.transform(img)
        # Ensure output is a tensor in CHW
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)  # float32 in [0,1]
        return img

        return img

    def __getitem__(self, idx):
        """
        Retrieves the image and corresponding labels at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): Loaded image tensor with shape (C, H, W).
                - label (torch.Tensor): Corresponding labels as a tensor.
        """

        # Build stacks by cycling within each task length
        images = [self._img_import(img_files[idx % n]) for img_files, n in zip(self.idx_, self.task_counts_)]
        # Guard: all images must share the same spatial size to stack
        shapes = {tuple(t.shape) for t in images}
        if len(shapes) != 1:
            raise ValueError(
                "All images must have the same shape to stack. Ensure your transform includes a fixed resize."
            )
        X_stack = torch.stack(images)  # (n_tasks, C, H, W)
        Y_stack = torch.stack([Y[idx % n] for Y, n in zip(self.Y, self.task_counts_)])  # (n_tasks, n_targets)
        return X_stack, Y_stack