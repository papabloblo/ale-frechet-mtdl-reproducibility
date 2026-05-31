import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor
import numpy as np
from scipy.interpolate import UnivariateSpline

from MultiTaskDeepLearning.model import MultiTaskModel


class Intervals:
    """
    Feature-wise discretization via quantiles or unique values.

    Builds a **monotone boundary vector** for each input feature and provides:
      - updates of extreme bounds with new data (robust to drift),
      - membership masks,
      - interval indices per sample,
      - per-sample left/right bounds.

    Intended for use by ALE (Accumulated Local Effects) routines.

    Notes
    -----
    • `intervals` is a **list** of length `n_features`. Each element is a 1D
      tensor of boundaries with shape `(n_boundaries_f,)`, `n_boundaries_f >= 2`.
      The number of intervals for feature `f` is `n_boundaries_f - 1`.
    • If a feature has few unique values (≤ `num_intervals`), boundaries are
      derived from its sorted unique values and then the first/last edges are
      expanded by `epsilon`. Otherwise, a regular quantile grid of size
      `num_intervals + 1` is used (deduplicated if needed) and its extreme
      edges are expanded by `epsilon`.
    • Expanding extremes by `epsilon` reduces edge effects so that all values
      fall strictly within some interval: (left, right].

    Attributes
    ----------
    intervals : list[torch.Tensor]
        Per-feature boundary vectors (monotone, length ≥ 2).
    n_features : int
        Number of input features.
    num_intervals : int
        Target number of intervals for the quantile grid. Actual per-feature
        counts can differ after deduplication or for low-cardinality features.
    device : str | torch.device
        Device where tensors are stored ('cpu' or 'cuda').
    epsilon : float
        Small expansion applied to first and last boundary for numerical stability.

    Examples
    --------
    >>> n_features = 3
    >>> n_samples = 8
    >>> X = torch.rand(n_samples, n_features)
    >>> intervals = Intervals(X, num_intervals=5, device='cpu')
    >>> len(intervals.intervals) == intervals.n_features
    True
    >>> # Interval indices for each (sample, feature):
    >>> idx = intervals.compute_index_intervals(X)  # (n_samples, n_features, 1)
    >>> idx.shape
    torch.Size([8, 3, 1])
    >>> # Left/right bounds for each (sample, feature):
    >>> b = intervals.bounds(X)  # (n_samples, n_features, 2)
    >>> b.shape
    torch.Size([8, 3, 2])
    """

    def __init__(self,
                 X: torch.Tensor,
                 num_intervals: int = 30,
                 device: str|torch.device = 'cpu',
                 epsilon: float = 1e-4
                 ):
        """
        Initialize and build per-feature boundary vectors.

        Parameters
        ----------
        X : torch.Tensor
            Input of shape (n_samples, n_features).
        num_intervals : int, default 30
            Target number of intervals (quantile grid has `num_intervals + 1` points).
        device : str | torch.device, default 'cpu'
            Target device ('cpu' or 'cuda').
        epsilon : float, default 1e-4
            Expansion applied to the first/last boundary to ensure coverage.

        Raises
        ------
        ValueError
            If `X` is not 2D or `num_intervals < 1`.
        """
        if X.dim() != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got {tuple(X.shape)}")
        if num_intervals < 1:
            raise ValueError("num_intervals must be >= 1")

        self.num_intervals = int(num_intervals)
        self.device = device
        self.epsilon = float(epsilon)

        self.intervals, self.n_features = self._create(X.to(self.device))

        # in __init__ after building self.intervals
        self._first_cache = torch.stack([b[0] for b in self.intervals]).to(self.device)
        self._last_cache = torch.stack([b[-1] for b in self.intervals]).to(self.device)

    def _create_old(self, X: torch.Tensor) -> torch.Tensor:
        """
        [LEGACY] Build a padded (n_features, num_intervals+1) boundary tensor.

        This earlier approach quantized each feature to a fixed-length boundary vector,
        padding with `-inf` when a feature had fewer valid boundaries. It is kept for
        reference/testing and is **not** used by default.

        Parameters
        ----------
        X : torch.Tensor
            2D tensor (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Padded boundaries of shape (n_features, num_intervals + 1).

        Raises
        ------
        ValueError
            If `X` is not 2D.
        """

        # Check input dimensionality
        if X.dim() != 2:
            raise ValueError(f"Input tensor must be 2D (batch_size, num_features), but got {X.dim()}D")

        # Calculate quantiles based on the desired number of intervals
        # If `self.num_intervals = 5`, then quantiles = torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1]).
        base_quantiles = torch.linspace(0, 1, steps=self.num_intervals - 1, device=self.device)

        intervals_list = []
        n_features = X.size(1)
        for feature in range(n_features):
            x = X[:, feature]

            unique_values = x.unique()

            if len(unique_values) > self.num_intervals:
                # Use quantile-based intervals for features with many unique values
                interval = x.quantile(q=base_quantiles).unique()
            else:
                quantiles = torch.linspace(0, 1, steps=max(len(unique_values) - 1, 2), device=self.device)
                interval = x.quantile(q=quantiles).unique()

            # Prepend and append the extreme boundaries.
            interval = torch.cat((torch.tensor([interval[0]], device=self.device), interval))
            interval = torch.cat((interval, torch.tensor([interval[-1]], device=self.device)))

            # Adjust extreme boundaries with epsilon.
            interval[0]  -= self.epsilon
            interval[-1] += self.epsilon

            # Pad with -inf if needed so that each feature has (num_intervals+1) boundaries.
            if interval.size(0) < self.num_intervals + 1:
                padding = torch.full((self.num_intervals + 1 - interval.size(0),),
                                     float('-inf'),
                                     device=self.device)
                interval = torch.cat([interval, padding])

            intervals_list.append(interval)

        return torch.stack(intervals_list)

    @torch.no_grad()
    def _create(self, X: torch.Tensor) -> tuple[list[torch.Tensor], int]:
        """
        Build per-feature boundary vectors without global padding.

        Strategy
        --------
        • If `#unique(feature) ≤ num_intervals`: use sorted unique values.
        • Else: use a regular quantile grid of size `num_intervals + 1`.
        In both cases, expand the first/last boundary by `epsilon`.

        Parameters
        ----------
        X : torch.Tensor
            2D tensor (n_samples, n_features).

        Returns
        -------
        (list[torch.Tensor], int)
            A list of length `n_features` with 1D boundary tensors, and `n_features`.
        """
        n_features = X.size(1)
        q_full = torch.linspace(0, 1, steps=self.num_intervals + 1, device=self.device)
        interval_list: list[Tensor] = []

        for feature in range(n_features):
            x = X[:, feature]
            uniq_x = x.unique(sorted=True)

            if uniq_x.numel() <= self.num_intervals:
                bounds = torch.cat((uniq_x[:1], uniq_x))
            else:
                bounds = x.quantile(q_full, interpolation='higher')

            # Ensure first/last are slightly open to catch all values
            bounds[0]  = bounds[0]  - self.epsilon
            bounds[-1] = bounds[-1] + self.epsilon

            interval_list.append(bounds)

        return interval_list, n_features

    @torch.no_grad()
    def update(self, X: torch.Tensor) -> None:
        """
        Update extreme boundaries from new data.

        For each feature, sets:
          • first boundary = min(feature) - epsilon
          • last  boundary = max(feature) + epsilon

        Parameters
        ----------
        X : torch.Tensor
            2D tensor (n_samples, n_features).
        """

        self._update_extremes(X)

    @torch.no_grad()
    def _update_extremes(self, X: torch.Tensor) -> None:
        X = X.to(self.device)
        batch_min = X.amin(dim=0) - self.epsilon
        batch_max = X.amax(dim=0) + self.epsilon

        new_first = torch.minimum(self._first_cache, batch_min)
        new_last = torch.maximum(self._last_cache, batch_max)

        upd_left_idx = (new_first != self._first_cache).nonzero(as_tuple=False).flatten()
        upd_right_idx = (new_last != self._last_cache).nonzero(as_tuple=False).flatten()

        for i in upd_left_idx.tolist():
            self.intervals[i][0] = new_first[i]
        for i in upd_right_idx.tolist():
            self.intervals[i][-1] = new_last[i]

        # keep caches in sync
        if upd_left_idx.numel():
            self._first_cache[upd_left_idx] = new_first[upd_left_idx]
        if upd_right_idx.numel():
            self._last_cache[upd_right_idx] = new_last[upd_right_idx]

    def compute_boolean_mask(self, X: torch.Tensor) -> list[Tensor]:
        """
        Compute a boolean membership mask per feature.

        For feature `f` with boundaries `b = intervals[f]` of length `L`,
        returns a mask of shape `(n_samples, L-1)` where entry `(i, j)` is:
            (b[j] < X[i, f] <= b[j+1])

        Parameters
        ----------
        X : torch.Tensor
            2D tensor (n_samples, n_features).

        Returns
        -------
        list[torch.Tensor]
            A list of length `n_features`; each element has shape `(n_samples, n_intervals_f)`.
        """

        return [
            (X[:,feature].unsqueeze(-1) > self.intervals[feature][:-1]) &
            (X[:,feature].unsqueeze(-1) <= self.intervals[feature][1:])
            for feature in range(self.n_features)
        ]

    def compute_index_intervals(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fast left-interval index via torch.bucketize (no boolean masks).

        Returns
        -------
        torch.Tensor
            (n_samples, n_features, 1) int64; for each (i,f), index j s.t. b[j] < x <= b[j+1].
        """
        X = X.to(self.device)
        N, F = X.shape
        # Column slices X[:, f] are strided views; transpose once so each feature
        # row is contiguous and bucketize avoids an internal copy per feature.
        Xt = X.transpose(0, 1).contiguous()
        idx_cols = []
        for f in range(self.n_features):
            boundaries = self.intervals[f]

            left_idx = torch.bucketize(Xt[f], boundaries, right=False) - 1
            left_idx = left_idx.clamp_(0, boundaries.numel() - 1)

            idx_cols.append(left_idx.view(N, 1))
        return torch.cat(idx_cols, dim=1).unsqueeze(-1)    # (N, F, 1)

    @torch.no_grad()
    def bounds(self, X: torch.Tensor, indices_left: torch.Tensor | None = None) -> torch.Tensor:
        """
        Fast left/right bounds using direct indexing on per-feature boundaries.

        Parameters
        ----------
        X : (N, F)
        indices_left : (N, F, 1) or None
            If None, indices are computed (bucketize). If provided, no recompute.

        Returns
        -------
        (N, F, 2) with [left, right]
        """
        X = X.to(self.device)
        #self._update_extremes(X)

        if indices_left is None:
            indices_left = self.compute_index_intervals(X)  # (N, F, 1)
        N, F, _ = indices_left.shape

        out = torch.empty((N, F, 2), device=self.device, dtype=X.dtype)
        for f in range(self.n_features):
            b = self.intervals[f]  # (L,)
            i = indices_left[:, f, 0].clamp(0, b.numel() - 2)  # (N,)
            out[:, f, 0] = b[i]  # left
            out[:, f, 1] = b[i + 1]  # right
        return out

    def cardinality(self, X: torch.Tensor) -> list[torch.Tensor]:
        """
        Count how many samples fall in each interval for each feature.

        Parameters
        ----------
        X : torch.Tensor
            2D tensor (n_samples, n_features).

        Returns
        -------
        list[torch.Tensor]
            List of length `n_features`; each entry has shape `(n_intervals_f,)` (dtype: torch.int64).
        """

        self._update_extremes(X)
        boolean_mask = self.compute_boolean_mask(X)

        return [boolean_mask[feature].sum(dim=0) for feature in range(self.n_features)]

    def n_intervals_per_feature(self) -> torch.Tensor:
        """
        Number of intervals for each feature.

        Returns
        -------
        torch.Tensor
            1D tensor of length `n_features` with `n_intervals_f = len(boundaries_f) - 1`.
        """
        return torch.tensor([interval.numel() - 1 for interval in self.intervals], device=self.device)


class MultiTaskALE:
    """
    Accumulated Local Effects (ALE) for multi-task models.

    Pipeline
    --------
    1) Initialize per-task (or shared) `Intervals` on a sample of data.
    2) For each batch: construct left/right perturbations per feature.
    3) Forward both perturbations, take output differences → local effects.
    4) Accumulate effects and counts per (feature, interval, output).
    5) Read out averaged ALE curves, optionally cumulative/centered and smoothed.

    Expected shapes (contract)
    --------------------------
    • `model.model_similarity_input(X)` → `(T, N, F)`:
        T = number of tasks, N = batch size, F = number of input features.
    • `model.model_by_task(t)['model_similarity'](X_pert)` → `(2N, F, O)`:
        O = number of output features. The first N correspond to left-perturbed,
        the next N to right-perturbed inputs.

    Attributes
    ----------
    model : MultiTaskModel
        Multi-task model wrapper exposing `model_similarity_input` and
        `model_by_task(t)['model_similarity']`.
    dataloader : DataLoader
        Source of batches to compute ALE on.
    n_tasks : int
        Number of tasks T.
    shared_input_data : bool
        If True, all tasks share the same input data; indices/perturbations are computed once per batch.
    n_features_out : int
        Number of outputs O per task used in ALE curves.
    num_intervals : int
        Number of ALE intervals I for accumulation (fixed rectangular grid).
    device : torch.device
        Device where accumulators live.
    epsilon : float
        Stability margin propagated to `Intervals`.
    n_guess : int
        Approximate number of samples to seed interval creation.
    g_ale_per_feature : torch.Tensor
        Accumulator with shape `(T, F, I, O)`.
    cardinality : torch.Tensor
        Counts with shape `(T, F, I)`.
    intervals : list[Intervals] | [Intervals]
        One `Intervals` per task, or a single shared one if `shared_input_data=True`.
    n_features_in : int
        Number of input features F.

    Examples
    --------
    >>> ale = MultiTaskALE(model, dataloader, n_tasks=3, num_intervals=30)
    >>> ale.update(max_batches=5)
    >>> curves = ale(centered=True, cumulative=True, spline=True, spline_smooth=1.0)
    >>> curves.shape  # (T, F, I, 1+O) : x grid + ALE values
    torch.Size([3, F, 30, 1+O])
    """
    def __init__(self,
                 model: MultiTaskModel,
                 dataloader: DataLoader,
                 n_tasks: int,
                 shared_input_data: bool = False,
                 n_features_out: int = 1,
                 num_intervals: int = 30,
                 device: str = 'cpu',
                 epsilon: float = 1e-4,
                 n_guess: int = 100,
                 ):
        """
        Initialize the MultiTaskALE estimator.

       Parameters
        ----------
        model : MultiTaskModel
            Multi-task model with the required interface (see class docstring).
        dataloader : DataLoader
            Iterable of batches consumed to compute ALE.
        n_tasks : int
            Number of tasks T.
        shared_input_data : bool, default False
            If True, a single `Intervals` object is shared across tasks and
            perturbations/indices are computed once per batch.
        n_features_out : int, default 1
            Number of outputs O per task to include in ALE curves.
        num_intervals : int, default 30
            Number of ALE intervals I to accumulate over (fixed grid size).
        device : str, default 'cpu'
            Target device ('cpu' or 'cuda').
        epsilon : float, default 1e-4
            Stability constant for interval edges; forwarded to `Intervals`.
        n_guess : int, default 100
            Approximate sample size used to seed interval construction.

        Raises
        ------
        ValueError
            If `num_intervals < 1`.
        """

        self.model = model
        self.dataloader = dataloader
        self.n_tasks = int(n_tasks)
        self.shared_input_data = bool(shared_input_data)
        self.n_features_out = int(n_features_out)
        self.device = torch.device(device)
        self.epsilon = float(epsilon)
        self.num_intervals = int(num_intervals)
        if self.num_intervals < 1:
            raise ValueError("num_intervals must be >= 1")

        self.n_guess = int(n_guess)


        self.g_ale_per_feature, self.cardinality, self.intervals, self.n_features_in = self._reinit()

    # --------------------------- set-up helpers ---------------------------

    def _reinit(self) -> tuple[Tensor, Tensor, list[Intervals], int]:
        """
        (Re)initialize accumulators and intervals.

        Returns
        -------
        g_ale_per_feature : torch.Tensor
            Zero-initialized accumulator with shape `(T, F, I, O)`.
        cardinality : torch.Tensor
            Zero-initialized counts with shape `(T, F, I)`.
        intervals : list[Intervals]
            List of `Intervals` (one per task) or a single-item list if shared.
        n_features_in : int
            Number of input features F derived from the sampled data.
        """
        intervals, n_features_in = self._initialize_intervals()

        g_ale_per_feature = torch.zeros(
            (self.n_tasks,
             n_features_in,
             self.num_intervals,
             self.n_features_out
             ),
            device=self.device
        )

        cardinality = torch.zeros(
            size=(self.n_tasks,
                  n_features_in,
                  self.num_intervals),
            device=self.device,
            dtype=torch.int
        )

        return g_ale_per_feature, cardinality, intervals, n_features_in

    def _initialize_intervals(self) -> tuple[list[Intervals], int]:
        """
        Build `Intervals` object(s) from a small sample of the dataloader.

        Returns
        -------
        (list[Intervals], int)
            A list of `Intervals` (one per task or one shared), and `n_features_in`.

        Examples
        --------
        >>> ints, nF = ale._initialize_intervals()
        >>> len(ints) in (1, ale.n_tasks)
        True
        """
        X_sample = self._X_sample().to(self.device)
        n_features_in = X_sample.size(2)

        if self.shared_input_data:
            ints = [
                Intervals(
                    X_sample[0],
                    num_intervals=self.num_intervals,
                    device=self.device,
                    epsilon=self.epsilon
                )
            ]
        else:
            ints = [
                Intervals(
                    X_sample[task],
                    num_intervals=self.num_intervals,
                    device=self.device,
                    epsilon=self.epsilon
                )
                for task in range(self.n_tasks)
            ]

        return ints, n_features_in

    @torch.no_grad()
    def _X_sample(self):
        """
        Collect ≈ `n_guess` samples (after `model_similarity_input`) to initialize intervals.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(T, N_s, F)` where `N_s` is the number of collected samples
            (≥ `n_guess`, subject to batch size).
        """
        samples = []
        seen = 0
        for X, _ in self.dataloader:
            X = X.to(self.device)
            Xs = self.model.model_similarity_input(X)
            samples.append(Xs.detach())
            seen += Xs.size(1)
            if seen >= self.n_guess:
                break
        if not samples:
            raise RuntimeError("Dataloader produced no samples to initialize intervals.")
        return torch.cat(samples, dim=1)

    # --------------------------- core update ---------------------------

    @torch.no_grad()
    def update(self, max_batches: int = None) -> None:
        """
        Accumulate local effects into `g_ale_per_feature` and `cardinality`.

        For each batch:
          1) transform inputs with `model_similarity_input`,
          2) build/update intervals and compute per-sample interval indices,
          3) construct left/right perturbations per feature,
          4) forward both perturbations and take differences,
          5) scatter-add differences into `(feature, interval, output)` bins and update counts.

        Parameters
        ----------
        max_batches : int, optional
            Early stop after processing this many batches. Defaults to all batches.

        Examples
        --------
        >>> ale.update(max_batches=5)  # useful to preview curves quickly
        """

        if max_batches is None:
            max_batches = float('inf')

        processed = 0
        I = self.num_intervals
        T = self.n_tasks
        for X, _ in self.dataloader:
            if processed >= max_batches:
                break
            processed += 1

            X = self.model.model_similarity_input(X.to(self.device)).to(self.device)  # (T, N, F)
            t_tasks, N, F = X.shape
            assert t_tasks == self.n_tasks and F == self.n_features_in

            # Precompute feature ids (used for flat indexing)
            # shape (N*F,)
            feat_ids_flat = torch.arange(F, device=self.device).unsqueeze(0).expand(N, F).reshape(-1)

            # Shared-input path: compute once
            if self.shared_input_data:
                ints = self.intervals[0]
                ints.update(X[0])
                idx = ints.compute_index_intervals(X[0])
                X_left, X_right = self._X_perturbation(X[0], ints, idx)  # (N, F, F)

            for t in range(T):
                if not self.shared_input_data:
                    ints = self.intervals[t]
                    ints.update(X[t])
                    idx = ints.compute_index_intervals(X[t])
                    X_left, X_right = self._X_perturbation(X[t], ints, idx)  # (N, F, F)

                # Forward once on concatenated perturbations
                X_cat = torch.cat([X_left, X_right], dim=0)  # (2N, F, F) or model-compatible
                Y_cat = self.model.model_by_task(t)['model_similarity'](X_cat)
                if Y_cat.dim() != 3:
                    raise RuntimeError("model_similarity must return (2N, F, O)")
                O = Y_cat.size(-1)

                Y_left, Y_right = Y_cat[:N], Y_cat[N:]
                Y_diff = (Y_right - Y_left)  # (N, F, O)

                # ---- counts: vectorized via one_hot ----
                # (N, F, I) int -> sum over N -> (F, I)
                counts = torch.nn.functional.one_hot(idx, num_classes=I).sum(dim=0).to(dtype=torch.int64).squeeze()
                if self.shared_input_data:
                    self.cardinality[0] += counts
                else:
                    self.cardinality[t] += counts

                # ---- effects: flat index_add_ into (F, I, O) ----
                # Flatten samples/features
                idx_flat = idx.reshape(-1)  # (N*F,)
                bins_flat = feat_ids_flat * I + idx_flat  # (N*F,)
                y_flat = Y_diff.reshape(-1, O)  # (N*F, O)

                # Accumulate into (F*I, O), then reshape to (F, I, O)
                accum_flat = torch.zeros(F * I, O, device=self.device, dtype=Y_diff.dtype)
                accum_flat.index_add_(0, bins_flat, y_flat)
                accum = accum_flat.view(F, I, O)

                self.g_ale_per_feature[t] += accum

    # --------------------------- perturbation helper ---------------------------

    def _X_perturbation(
            self,
            X: torch.Tensor,
            interval: Intervals,
            indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create per-feature left/right perturbations of `X` for ALE computation.

        For each feature `f`:
          • `X_left[:, f, f]` is replaced with the **left** bound of the interval
            containing `X[:, f]`.
          • `X_right[:, f, f]` is replaced with the **right** bound of the interval
            containing `X[:, f]`.

        Parameters
        ----------
        X : torch.Tensor
            Batch for a single task, shape `(N, F)`.
        interval : Intervals
            Precomputed intervals for this task (or shared across tasks).

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            A pair `(X_left, X_right)`, both of shape `(N, F, F)`.
            Dimension-1 selects the feature being perturbed.
        """

        X = X.to(self.device)
        if indices.device != self.device:
            indices = indices.to(self.device)

        # Vectorized bounds lookup using indices (no boolean masks)
        # bounds[..., 0] = left, bounds[..., 1] = right  -> shape (N, F, 2)
        bounds = interval.bounds(X, indices_left=indices)

        # Make F "views" of X, then clone once to get a writable buffer
        # expand avoids materializing F copies; clone makes it writable/contiguous
        X_expanded = X.unsqueeze(-2).expand(-1, X.size(-1), -1).clone()  # (N, F, F)

        # Overwrite the diagonal feature-by-feature with left/right bounds
        X_left = X_expanded.diagonal_scatter(bounds[:, :, 0], dim1=1, dim2=2)
        X_right = X_expanded.diagonal_scatter(bounds[:, :, 1], dim1=1, dim2=2)

        return X_left, X_right

    # --------------------------- read-out ---------------------------

    def _get_ale(
            self,
            centered: bool = True,
            cumulative: bool = True,
            std: float | None = None
    ) -> torch.Tensor:
        """
        Convert accumulated local effects into ALE curves on the fixed grid.

        Steps
        -----
        • Average local effects by dividing by `cardinality` per (feature, interval).
        • Optionally cumulative along the interval axis.
        • Optionally center per feature/output (zero mean along the interval axis).
        • Optionally scale to a target standard deviation per feature/output.

        Parameters
        ----------
        centered : bool, default True
            Subtract the mean along the interval axis for each (feature, output).
        cumulative : bool, default True
            Cumulative sum along the interval axis (ALE convention).
        std : float, optional
            If provided (>0), rescale each (feature, output) curve to this standard
            deviation along the interval axis.

        Returns
        -------
        torch.Tensor
            ALE values with shape `(T, F, I, O)`.
        """
        ale_curves = []
        for task in range(self.n_tasks):
            # Use shared cardinality if applicable.
            cardinality = self.cardinality[0] if self.shared_input_data else self.cardinality[task]
            # avoid div-by-zero
            denom = cardinality.unsqueeze(-1).clamp(min=1)
            avg = (self.g_ale_per_feature[task] / denom).nan_to_num_(nan=0.0)  # (F, I, O)
            if cumulative:
                avg = avg.cumsum(dim=1)
            if centered:
                avg = avg - avg.mean(dim=1, keepdim=True)
            if std is not None:
                if std <= 0:
                    raise ValueError(f"The standard deviation must be positive, but got {std}")
                # per feature/output std along I
                scale = avg.std(dim=1, keepdim=True).clamp(min=1e-12)
                avg = (avg / scale) * std
            ale_curves.append(avg.nan_to_num(0.0))

        return torch.stack(ale_curves, dim=0)

    def __call__(
            self,
            centered: bool = True,
            cumulative: bool = True,
            std: float = 1.0,
            spline: bool = False,
            spline_smooth: float = 1.0
    ) -> torch.Tensor:
        """
        Generate ALE curves and (optionally) smooth them with a spline.

        Output layout
        -------------
        Returns an array with the x-grid followed by ALE values:
          • `[..., 0]`   = interval right-edge grid `x` of shape `(T, F, I)`,
          • `[..., 1:]`  = ALE values `y`    of shape `(T, F, I, O)`.

        Parameters
        ----------
        centered : bool, default True
            Center curves along the interval axis.
        cumulative : bool, default True
            Use cumulative local effects (ALE convention).
        std : float, optional
            Target standard deviation to rescale curves (per feature/output).
        spline : bool, default False
            If True, smooth each (task, feature, output) curve with `UnivariateSpline`.
        spline_smooth : float, default 1.0
            Smoothing factor passed to the spline; the implementation may internally
            scale it by variance and sample count for stability.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(T, F, I, 1+O)` with `[x, y...]`.
        """
        Y = self._get_ale(centered=centered, cumulative=cumulative, std=std)
        Xg = self._interval_grid_right_edges()         # (T, F, I, O)

        out = torch.cat([Xg.unsqueeze(-1), Y], dim=-1)  # (T, F, I, 1+O)
        if spline:
            out = self.smooth_ale_curves(out, spline_smooth=spline_smooth)
        return out

    # --------------------------- grid helper ---------------------------

    def _interval_grid_right_edges(self) -> Tensor:
        """
        Build a fixed `(T, F, I)` grid of interval **right edges** from the `Intervals` objects.

        Implementation detail
        ---------------------
        Each feature's variable-length boundary vector is padded/truncated to length `I+1`
        (where `I = num_intervals`), then we take `b[1:]` as the right-edge grid.

        Returns
        -------
        torch.Tensor
            Monotone non-decreasing grid of shape `(T, F, I)`. If `shared_input_data=True`,
            the single computed grid is expanded across tasks.
        """
        T = self.n_tasks if not self.shared_input_data else 1
        F, I = self.n_features_in, self.num_intervals
        grids = []

        for t in range(T):
            ints = self.intervals[t]
            feat_edges = []
            for f in range(F):
                b = ints.intervals[f]  # variable length boundaries (>=2)
                # pad/truncate to length I+1
                if b.numel() < I + 1:
                    pad_val = b[-1]
                    pad = pad_val.repeat(I + 1 - b.numel())
                    b_fix = torch.cat([b, pad], dim=0)
                else:
                    b_fix = b[:I + 1]
                # right edges are b_fix[1:]
                feat_edges.append(b_fix[1:].unsqueeze(0))  # (1, I)
            grids.append(torch.cat(feat_edges, dim=0).unsqueeze(0))  # (1, F, I)

        Xg = torch.cat(grids, dim=0).to(self.device)  # (T or 1, F, I)
        if self.shared_input_data:
            Xg = Xg.expand(self.n_tasks, -1, -1)
        return Xg

    @staticmethod
    def smooth_ale_curves(ale_tensor: Tensor, spline_smooth: float = 1.0) -> Tensor:
        """
        Spline-smooth ALE values along the interval axis.

        Smooths each (task, feature, output) series independently using a
        cubic (or lower-order if too few points) `UnivariateSpline`.

        Parameters
        ----------
        ale_tensor : torch.Tensor
            Shape `(T, F, I, 1+O)`. `[..., 0]` is the x-grid, `[..., 1:]` are ALE values.
        spline_smooth : float, default 1.0
            Smoothing factor passed to `UnivariateSpline`. The implementation scales
            it by variance and sample count for stability. Set to 0 for interpolation.

        Returns
        -------
        torch.Tensor
            Same shape as input, with smoothed ALE values in `[..., 1:]`.

        Notes
        -----
        If the x-grid is (near) constant or spline construction fails, the
        original series is returned for that (task, feature, output).
        """
        T, F, I, C = ale_tensor.shape
        device, dtype = ale_tensor.device, ale_tensor.dtype
        out = ale_tensor.clone()

        for t in range(T):
            for f in range(F):
                x = out[t, f, :, 0].detach().cpu().numpy()
                # ensure strictly increasing (sorted unique); if constant, skip
                if np.allclose(x.max(), x.min()):
                    continue
                # enforce monotonic non-decreasing grid
                order = np.argsort(x)
                xs = np.asarray(x)[order]
                for c in range(1, C):
                    y = out[t, f, :, c].detach().cpu().numpy()
                    ys = np.asarray(y)[order]
                    try:
                        s = float(spline_smooth) * (np.var(ys) * max(len(xs), 2))
                        s = max(s, 1e-12)
                        k = min(3, max(1, len(xs) - 1))
                        spline = UnivariateSpline(xs, ys, s=s, k=k)
                        yhat = spline(xs)
                        # invert permutation
                        inv = np.empty_like(order)
                        inv[order] = np.arange(order.size)
                        ysm = yhat[inv]
                    except Exception:
                        ysm = y
                    out[t, f, :, c] = torch.tensor(ysm, device=device, dtype=dtype)
        return out
