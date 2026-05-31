"""Multi-task learning baseline models.

All models in this module follow the same *I/O contract* used by the project:

* Input ``X`` is a tensor shaped ``(n_tasks, batch, n_features)``.
* Output ``Y_pred`` is a tensor shaped ``(n_tasks, batch, n_outputs)``.

The trainer in this repository expects the model to expose a ``.device``
attribute (a ``torch.device``) and to run entirely on that device.

Implemented baselines
---------------------
* ``SoftSharing``  : independent per-task networks.
* ``SingleTaskMLP``: independent per-task MLPs, reported as a single-task baseline.
* ``HardSharing``  : shared trunk + task-specific heads.
* ``MMoE``         : Multi-gate Mixture-of-Experts (Ma et al., 2018).
* ``CrossStitch``  : Cross-stitch networks (Misra et al., 2016) for feature sharing.
* ``PLE``          : Progressive Layered Extraction / CGC (Tang et al., 2020).
* ``MTAN``         : Multi-Task Attention Network (Liu et al., 2019).

These are widely used baselines for MTL comparisons and should be sufficient
for a first experimental section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch
from torch import nn


def _act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    return nn.ReLU()


def _mlp(sizes: Sequence[int], activation: str = "relu", dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    sizes = list(map(int, sizes))
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(_act(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


class _MTLBase(nn.Module):
    """Small helper to standardize ``.device`` handling."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)

    def to(self, *args, **kwargs):  # type: ignore[override]
        out = super().to(*args, **kwargs)
        # Keep device attribute in sync.
        if args and isinstance(args[0], (torch.device, str)):
            self.device = torch.device(args[0])
        elif "device" in kwargs and kwargs["device"] is not None:
            self.device = torch.device(kwargs["device"])
        return out


class SoftSharing(_MTLBase):
    """Independent models per task (no sharing)."""

    def __init__(
        self,
        n_tasks: int,
        input_dim: int,
        hidden: Sequence[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self.n_tasks = int(n_tasks)
        self.output_dim = int(output_dim)
        sizes = [int(input_dim), *map(int, hidden), int(output_dim)]
        self.nets = nn.ModuleList([
            _mlp(sizes, activation=activation, dropout=dropout) for _ in range(self.n_tasks)
        ])
        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 3:
            raise ValueError(f"Expected X with shape (n_tasks, batch, n_features), got {tuple(X.shape)}")
        if X.size(0) != self.n_tasks:
            raise ValueError(f"Expected X.size(0)=={self.n_tasks}, got {X.size(0)}")
        outs = [self.nets[t](X[t]) for t in range(self.n_tasks)]
        return torch.stack(outs, dim=0)

    def get_task_parameters(self, task: int) -> torch.Tensor:
        parts = [
            p.flatten()
            for name, p in self.nets[int(task)].named_parameters()
            if "weight" in name
        ]
        return torch.cat(parts, dim=0) if parts else torch.empty(0, device=self.device)

    def get_param_groups(self, tasks_groups: list[list[int]]) -> list[torch.Tensor]:
        return [
            torch.stack([self.get_task_parameters(task) for task in group], dim=0)
            for group in tasks_groups
        ]


class SingleTaskMLP(SoftSharing):
    """Single-task baseline implemented as one independent MLP per task.

    The task networks have disjoint parameters and no similarity or sharing
    penalty. Training them in the multitask batch loop is equivalent to
    optimizing independent task-specific MLPs with a shared optimizer schedule.
    """


class HardSharing(_MTLBase):
    """Shared trunk + task-specific heads (classic hard parameter sharing)."""

    def __init__(
        self,
        n_tasks: int,
        input_dim: int,
        trunk: Sequence[int],
        head: Sequence[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self.n_tasks = int(n_tasks)
        self.output_dim = int(output_dim)

        trunk_sizes = [int(input_dim), *map(int, trunk)]
        if len(trunk_sizes) < 2:
            trunk_sizes = [int(input_dim), int(input_dim)]
        self.trunk = _mlp(trunk_sizes, activation=activation, dropout=dropout)

        head_sizes = [trunk_sizes[-1], *map(int, head), int(output_dim)]
        self.heads = nn.ModuleList([
            _mlp(head_sizes, activation=activation, dropout=dropout) for _ in range(self.n_tasks)
        ])
        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 3:
            raise ValueError(f"Expected X with shape (n_tasks, batch, n_features), got {tuple(X.shape)}")
        if X.size(0) != self.n_tasks:
            raise ValueError(f"Expected X.size(0)=={self.n_tasks}, got {X.size(0)}")
        feats = torch.stack([self.trunk(X[t]) for t in range(self.n_tasks)], dim=0)
        outs = [self.heads[t](feats[t]) for t in range(self.n_tasks)]
        return torch.stack(outs, dim=0)


class MMoE(_MTLBase):
    """Multi-gate Mixture-of-Experts (MMoE).

    Reference: Ma et al., "Modeling Task Relationships in Multi-task Learning
    with Multi-gate Mixture-of-Experts", KDD 2018.
    """

    def __init__(
        self,
        n_tasks: int,
        input_dim: int,
        n_experts: int,
        expert_hidden: Sequence[int],
        tower_hidden: Sequence[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self.n_tasks = int(n_tasks)
        self.n_experts = int(n_experts)
        self.output_dim = int(output_dim)

        expert_sizes = [int(input_dim), *map(int, expert_hidden)]
        if len(expert_sizes) < 2:
            expert_sizes = [int(input_dim), 64]
        self.experts = nn.ModuleList([
            _mlp(expert_sizes, activation=activation, dropout=dropout) for _ in range(self.n_experts)
        ])
        expert_out_dim = expert_sizes[-1]

        # Per-task gates: produce weights for experts.
        self.gates = nn.ModuleList([
            nn.Linear(int(input_dim), self.n_experts) for _ in range(self.n_tasks)
        ])

        # Per-task towers/heads.
        tower_sizes = [expert_out_dim, *map(int, tower_hidden), int(output_dim)]
        self.towers = nn.ModuleList([
            _mlp(tower_sizes, activation=activation, dropout=dropout) for _ in range(self.n_tasks)
        ])

        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 3:
            raise ValueError(f"Expected X with shape (n_tasks, batch, n_features), got {tuple(X.shape)}")
        if X.size(0) != self.n_tasks:
            raise ValueError(f"Expected X.size(0)=={self.n_tasks}, got {X.size(0)}")

        # Compute expert outputs per task input: (n_tasks, n_experts, batch, expert_dim)
        expert_outs = []
        for e in self.experts:
            expert_outs.append(torch.stack([e(X[t]) for t in range(self.n_tasks)], dim=0))
        expert_outs = torch.stack(expert_outs, dim=1)

        outs = []
        for t in range(self.n_tasks):
            gate_logits = self.gates[t](X[t])  # (batch, n_experts)
            gate_w = torch.softmax(gate_logits, dim=-1)  # (batch, n_experts)
            # Weighted sum of experts: (batch, expert_dim)
            mix = torch.einsum("be,ebd->bd", gate_w, expert_outs[t])
            outs.append(self.towers[t](mix))
        return torch.stack(outs, dim=0)


class CrossStitch(_MTLBase):
    """Cross-stitch network for MTL.

    This is a simple MLP-based adaptation of cross-stitch units:
    each layer produces task-specific representations, then a learnable
    cross-stitch matrix mixes those representations across tasks.

    Reference: Misra et al., "Cross-stitch Networks for Multi-task Learning",
    CVPR 2016.
    """

    def __init__(
        self,
        n_tasks: int,
        input_dim: int,
        shared_dims: Sequence[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        init_identity: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self.n_tasks = int(n_tasks)
        self.output_dim = int(output_dim)

        dims = [int(input_dim), *map(int, shared_dims)]
        if len(dims) < 2:
            dims = [int(input_dim), 64]

        # Per-layer per-task linear transforms.
        self.task_linears = nn.ModuleList()
        self.task_acts = nn.ModuleList()
        self.task_drop = nn.ModuleList()
        self.cross = nn.ParameterList()

        for li in range(len(dims) - 1):
            in_d, out_d = dims[li], dims[li + 1]
            self.task_linears.append(nn.ModuleList([
                nn.Linear(in_d, out_d) for _ in range(self.n_tasks)
            ]))
            self.task_acts.append(nn.ModuleList([
                _act(activation) for _ in range(self.n_tasks)
            ]))
            self.task_drop.append(nn.ModuleList([
                nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
                for _ in range(self.n_tasks)
            ]))

            # Cross-stitch matrix A: mixes tasks at this layer.
            A = torch.eye(self.n_tasks)
            if not init_identity:
                A = A + 0.01 * torch.randn_like(A)
            self.cross.append(nn.Parameter(A))

        # Final heads per task
        self.heads = nn.ModuleList([
            nn.Linear(dims[-1], int(output_dim)) for _ in range(self.n_tasks)
        ])

        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 3:
            raise ValueError(f"Expected X with shape (n_tasks, batch, n_features), got {tuple(X.shape)}")
        if X.size(0) != self.n_tasks:
            raise ValueError(f"Expected X.size(0)=={self.n_tasks}, got {X.size(0)}")

        h = [X[t] for t in range(self.n_tasks)]  # list of (batch, dim)
        for li in range(len(self.task_linears)):
            # independent transform
            z = []
            for t in range(self.n_tasks):
                out = self.task_linears[li][t](h[t])
                out = self.task_acts[li][t](out)
                out = self.task_drop[li][t](out)
                z.append(out)

            # mix across tasks: h_t = sum_j A[t,j] * z_j
            A = self.cross[li]
            mixed = []
            for t in range(self.n_tasks):
                acc = 0.0
                for j in range(self.n_tasks):
                    acc = acc + A[t, j] * z[j]
                mixed.append(acc)
            h = mixed

        outs = [self.heads[t](h[t]) for t in range(self.n_tasks)]
        return torch.stack(outs, dim=0)


class PLE(_MTLBase):
    """Progressive Layered Extraction (PLE) / CGC blocks for MTL.

    This is a *tabular/MLP-friendly* implementation of the PLE idea:
    stacked CGC layers with shared experts + task-specific experts, each
    combined by learnable gates.

    Reference: Tang et al., "Progressive Layered Extraction (PLE): A Novel
    Multi-Task Learning (MTL) Model for Personalized Recommendations", 2020.
    """

    def __init__(
        self,
        n_tasks: int,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        n_shared_experts: int = 4,
        n_task_experts: int = 2,
        expert_hidden: Sequence[int] = (128,),
        tower_hidden: Sequence[int] = (64,),
        activation: str = "relu",
        dropout: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self.n_tasks = int(n_tasks)
        self.output_dim = int(output_dim)
        self.n_layers = int(n_layers)
        self.n_shared_experts = int(n_shared_experts)
        self.n_task_experts = int(n_task_experts)

        # CGC layers
        self.cgc_layers = nn.ModuleList()
        cur_dim = int(input_dim)
        for _ in range(self.n_layers):
            layer = _CGCLayer(
                n_tasks=self.n_tasks,
                input_dim=cur_dim,
                n_shared_experts=self.n_shared_experts,
                n_task_experts=self.n_task_experts,
                expert_hidden=expert_hidden,
                activation=activation,
                dropout=dropout,
            )
            self.cgc_layers.append(layer)
            cur_dim = layer.out_dim

        # Task towers/heads
        tower_sizes = [cur_dim, *map(int, tower_hidden), int(output_dim)]
        self.towers = nn.ModuleList([
            _mlp(tower_sizes, activation=activation, dropout=dropout) for _ in range(self.n_tasks)
        ])

        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 3:
            raise ValueError(f"Expected X with shape (n_tasks, batch, n_features), got {tuple(X.shape)}")
        if X.size(0) != self.n_tasks:
            raise ValueError(f"Expected X.size(0)=={self.n_tasks}, got {X.size(0)}")

        # Representations for each task + shared stream
        reps_t = [X[t] for t in range(self.n_tasks)]
        rep_s = X.mean(dim=0)  # (batch, dim)

        for layer in self.cgc_layers:
            reps_t, rep_s = layer(reps_t, rep_s)

        outs = [self.towers[t](reps_t[t]) for t in range(self.n_tasks)]
        return torch.stack(outs, dim=0)


class MTAN(_MTLBase):
    """Multi-Task Attention Network (MTAN) for tabular/MLP backbones.

    MTAN uses a shared backbone and applies *task-specific attention masks*
    to shared representations.

    Reference: Liu et al., "End-to-End Multi-Task Learning with Attention", 2019.
    """

    def __init__(
        self,
        n_tasks: int,
        input_dim: int,
        shared_dims: Sequence[int],
        output_dim: int,
        tower_hidden: Sequence[int] = (64,),
        activation: str = "relu",
        dropout: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self.n_tasks = int(n_tasks)
        self.output_dim = int(output_dim)

        dims = [int(input_dim), *map(int, shared_dims)]
        if len(dims) < 2:
            dims = [int(input_dim), 128]

        self.shared_linears = nn.ModuleList()
        self.shared_acts = nn.ModuleList()
        self.shared_drop = nn.ModuleList()
        self.attn = nn.ModuleList()  # per layer: per-task attention mask generator

        for li in range(len(dims) - 1):
            in_d, out_d = dims[li], dims[li + 1]
            self.shared_linears.append(nn.Linear(in_d, out_d))
            self.shared_acts.append(_act(activation))
            self.shared_drop.append(nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity())
            self.attn.append(nn.ModuleList([
                nn.Sequential(nn.Linear(out_d, out_d), nn.Sigmoid()) for _ in range(self.n_tasks)
            ]))

        final_dim = dims[-1]
        tower_sizes = [final_dim, *map(int, tower_hidden), int(output_dim)]
        self.towers = nn.ModuleList([
            _mlp(tower_sizes, activation=activation, dropout=dropout) for _ in range(self.n_tasks)
        ])

        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 3:
            raise ValueError(f"Expected X with shape (n_tasks, batch, n_features), got {tuple(X.shape)}")
        if X.size(0) != self.n_tasks:
            raise ValueError(f"Expected X.size(0)=={self.n_tasks}, got {X.size(0)}")

        # Shared input: average across tasks (common in tabular MTL when tasks share covariates)
        h = X.mean(dim=0)  # (batch, dim)

        # Track last attentive representation per task
        h_task = [h for _ in range(self.n_tasks)]

        for li in range(len(self.shared_linears)):
            h = self.shared_linears[li](h)
            h = self.shared_acts[li](h)
            h = self.shared_drop[li](h)

            # Attention masks per task over the shared representation
            for t in range(self.n_tasks):
                mask = self.attn[li][t](h)
                h_task[t] = h * mask

        outs = [self.towers[t](h_task[t]) for t in range(self.n_tasks)]
        return torch.stack(outs, dim=0)


class _CGCLayer(nn.Module):
    """A single CGC layer used by PLE."""

    def __init__(
        self,
        n_tasks: int,
        input_dim: int,
        n_shared_experts: int,
        n_task_experts: int,
        expert_hidden: Sequence[int],
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_tasks = int(n_tasks)
        self.n_shared_experts = int(n_shared_experts)
        self.n_task_experts = int(n_task_experts)

        # Experts are MLPs that map input_dim -> expert_out_dim
        sizes = [int(input_dim), *map(int, expert_hidden)]
        if len(sizes) < 2:
            sizes = [int(input_dim), 128]
        self.out_dim = int(sizes[-1])

        self.shared_experts = nn.ModuleList([
            _mlp(sizes, activation=activation, dropout=dropout) for _ in range(self.n_shared_experts)
        ])

        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                _mlp(sizes, activation=activation, dropout=dropout) for _ in range(self.n_task_experts)
            ])
            for _ in range(self.n_tasks)
        ])

        # Gates
        # - Task gate sees (shared + its task experts)
        self.task_gates = nn.ModuleList([
            nn.Linear(int(input_dim), self.n_shared_experts + self.n_task_experts) for _ in range(self.n_tasks)
        ])
        # - Shared gate sees (shared + all task experts)
        self.shared_gate = nn.Linear(int(input_dim), self.n_shared_experts + self.n_tasks * self.n_task_experts)

    def forward(self, reps_t: list[torch.Tensor], rep_s: torch.Tensor):
        # Expert outputs
        # Shared experts: use shared rep
        shared_out = [e(rep_s) for e in self.shared_experts]  # list[(batch,out_dim)]

        # Task experts: use task rep
        task_out: list[list[torch.Tensor]] = []
        for t in range(self.n_tasks):
            task_out.append([e(reps_t[t]) for e in self.task_experts[t]])

        # Combine for each task
        new_reps_t: list[torch.Tensor] = []
        for t in range(self.n_tasks):
            experts = shared_out + task_out[t]
            stack = torch.stack(experts, dim=1)  # (batch, n_exp, out_dim)
            w = torch.softmax(self.task_gates[t](reps_t[t]), dim=-1)  # (batch, n_exp)
            mixed = torch.einsum('be,bed->bd', w, stack)
            new_reps_t.append(mixed)

        # Combine for shared stream
        all_task_out = [x for t in range(self.n_tasks) for x in task_out[t]]
        experts_s = shared_out + all_task_out
        stack_s = torch.stack(experts_s, dim=1)
        w_s = torch.softmax(self.shared_gate(rep_s), dim=-1)
        new_rep_s = torch.einsum('be,bed->bd', w_s, stack_s)

        return new_reps_t, new_rep_s
