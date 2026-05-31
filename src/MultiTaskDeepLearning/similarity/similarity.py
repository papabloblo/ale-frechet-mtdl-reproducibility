import torch
from MultiTaskDeepLearning.similarity.ale import MultiTaskALE


def _precompute_indices(m: int, device: torch.device):
    """Produce index tensors (i_idx, j_idx) for each diagonal with i,j >= 1."""
    diag_indices = []
    for diag in range(2, 2 * m - 1):
        i_start = max(1, diag - (m - 1))
        i_end_exclusive = min(m, diag)
        if i_start < i_end_exclusive:
            i_idx = torch.arange(i_start, i_end_exclusive, device=device)
            j_idx = diag - i_idx
            diag_indices.append((i_idx, j_idx))
    return diag_indices


def frechet_distance_vectorized(
    curve0: torch.Tensor,
    curve1: torch.Tensor,
) -> torch.Tensor:
    """
    Discrete Fréchet distance (vectorized across a batch/features dimension).

    Args:
        curve0: Tensor of shape (m, 2) or (B, m, 2)
        curve1: Tensor of shape (m, 2) or (B, m, 2)
                (batch dims must match if present)

    Returns:
        distances: Tensor of shape () if input was (m,2),
                   or (B,) if input was (B,m,2).
    """
    if curve0.shape != curve1.shape:
        raise ValueError("curve0 and curve1 must have the same shape")
    if curve0.size(-1) != 2:
        raise ValueError("curves must have last dim size 2 (x,y points)")

    # Normalize to (B, m, 2)
    squeeze_back = False
    if curve0.dim() == 2:
        curve0 = curve0.unsqueeze(0)
        curve1 = curve1.unsqueeze(0)
        squeeze_back = True

    B, m, _ = curve0.shape
    device = curve0.device
    dtype = curve0.dtype

    # Pairwise distances per batch/feature: (B, m, m)
    # torch.cdist supports batched inputs
    D = torch.cdist(curve0, curve1, compute_mode="donot_use_mm_for_euclid_dist")  # (B, m, m)

    # DP table: (B, m, m)
    F = torch.empty_like(D)

    # Base cases
    F[:, 0, 0] = D[:, 0, 0]
    # First row: cumulative max along j
    # D[:, 0, 1:] -> (B, m-1), cummax along last dim
    F[:, 0, 1:] = torch.cummax(D[:, 0, 1:], dim=-1).values
    # First column: cumulative max along i
    # D[:, 1:, 0] -> (B, m-1), cummax along dim=1 (the i dimension)
    F[:, 1:, 0] = torch.cummax(D[:, 1:, 0], dim=1).values

    # Fill anti-diagonals for i,j >= 1 (vectorized over B)
    for i_idx, j_idx in _precompute_indices(m, device=device):
        # Gather previous mins (each (B, L) where L=len(diagonal))
        prev_min = torch.minimum(
            torch.minimum(F[:, i_idx - 1, j_idx], F[:, i_idx, j_idx - 1]),
            F[:, i_idx - 1, j_idx - 1],
        )
        # Update current diagonal in one shot
        F[:, i_idx, j_idx] = torch.maximum(D[:, i_idx, j_idx], prev_min)

    distances = F[:, -1, -1]  # (B,)

    # EXPERIMENTAL--------------------------
    distances = torch.exp(-distances)
    #---------------------------------------

    return distances.squeeze(0) if squeeze_back else distances


class MultitaskSimilarity:
    """
    Compute Fréchet distances between ALE curves for multiple tasks.
    Uses vectorized Fréchet across features for speed.
    """

    def __init__(
        self,
        ale_curves: MultiTaskALE,
        similarity_func=frechet_distance_vectorized,
        centered: bool = True,
        cumulative: bool = True,
        std: float = 1.0,
        complete: bool = True,
        spline: bool = False
    ):
        self.device = ale_curves.device
        self.n_tasks = ale_curves.n_tasks
        self.num_intervals = ale_curves.num_intervals
        self.n_features_in = ale_curves.n_features_in
        self.n_features_out = ale_curves.n_features_out
        self.ale_curves = ale_curves

        self.centered = centered
        self.cumulative = cumulative
        self.std = std
        self.complete = complete
        self.spline = spline

        self.similarity_func = similarity_func

        # Preallocate result tensors on the right device/dtype
        #self.distances = torch.full(
        #    (self.n_tasks, self.n_tasks),
        #    fill_value=float("inf"),
        #    device=self.device,
        #    dtype=torch.float32,
        #)

        self.distances = torch.zeros(
            (self.n_tasks, self.n_tasks),
            device=self.device,
            dtype=torch.float32,
        )
        self.similarity_tasks_features = torch.zeros(
            (self.n_tasks, self.n_tasks, self.n_features_in),
            device=self.device,
            dtype=torch.float32,
        )

    def _compute_by_feature(
        self, ale0: torch.Tensor, ale1: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized per-feature Fréchet:
          ale0/ale1: (n_features, m, 2)
        Returns:
          distances per feature: (n_features,)
        """
        # Ensure same dtype/device
        ale0 = ale0.to(self.device)
        ale1 = ale1.to(self.device)
        # Single call computes all features at once
        d = self.similarity_func(ale0, ale1)  # (n_features,)
        # Ensure a consistent dtype
        return d.to(dtype=torch.float32, device=self.device)

    def compute(self) -> None:
        """
        Fills:
          - self.similarity_tasks_features[t0, t1, f] = Fréchet(curve_f^t0, curve_f^t1)
          - self.distances[t0, t1] = sum_f Fréchet(...)
        """
        curves = self.ale_curves(self.centered, self.cumulative, self.std, self.spline)
        # Expecting curves shape: (n_tasks, n_features, m, 2)
        # If your ale_curves returns a different shape, adapt accordingly.

        with torch.inference_mode():
            for t0 in range(self.n_tasks):
                c0 = curves[t0]  # (n_features, m, 2)
                for t1 in range(t0 + 1, self.n_tasks):
                    c1 = curves[t1]  # (n_features, m, 2)
                    feats = self._compute_by_feature(c0, c1)  # (n_features,)
                    self.similarity_tasks_features[t0, t1] = feats
                    self.similarity_tasks_features[t1, t0] = feats
                    dsum = feats.sum()
                    self.distances[t0, t1] = dsum
                    self.distances[t1, t0] = dsum

            # Optional but clearer: zero self-distances
            idx = torch.arange(self.n_tasks, device=self.device)

            # EXPERIMENTAL
            # self.distances[idx, idx] = float('inf')

    def tasks_groups(self):
        tasks = torch.arange(start=0, end=self.n_tasks, device=self.device)
        # EXPERIMENTAL
        similarity, nearest = self.distances.max(dim=1)

        #similarity, nearest = self.distances.min(dim=1)

        pairs = torch.column_stack((tasks, nearest))
        return similarity, pairs