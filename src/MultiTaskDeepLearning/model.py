import math
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, Union, TypedDict

import torch
from torch import nn

TensorOrTensors = Union[torch.Tensor, Sequence[torch.Tensor]]
ModuleFactory = Callable[[], nn.Module]

# ---------- Utils ----------

def random_state_dict_like(
    state_dict: MutableMapping[str, torch.Tensor],
    weight_init: str = "kaiming_uniform",
    bias_range: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Create a dtype/device-safe randomized copy of a state_dict.

    - Float / bfloat16 / half: kaiming/xavier/normal as chosen
    - Integer / bool / others: zeros_like (non-trainable buffers)
    """
    new_sd: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not v.is_floating_point():
            # Keep shapes/devices, but avoid randn_like errors.
            new_sd[k] = torch.zeros_like(v)
            continue

        out = torch.empty_like(v)

        if "bias" in k:
            nn.init.uniform_(out, -bias_range, bias_range)
        else:
            if weight_init == "kaiming_uniform":
                # a=√5 is PyTorch's Linear default gain; tune as you like
                nn.init.kaiming_uniform_(out, a=math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu")
            elif weight_init == "xavier_uniform":
                nn.init.xavier_uniform_(out)
            elif weight_init == "normal_0_02":
                nn.init.normal_(out, mean=0.0, std=0.02)
            else:
                nn.init.kaiming_uniform_(out, a=math.sqrt(5))

        new_sd[k] = out
    return new_sd

# ---------- Hard sharing (one module reused across tasks) ----------

class SharedModule(nn.Module):
    """
    Wrap a single module that is shared (hard sharing) across tasks.

    Forward expects X of shape (n_tasks, *input_shape) and applies the same module
    per task, stacking along dim 0.
    """
    def __init__(
            self,
            module: nn.Module,
            n_tasks: int,
            shared_input_data: bool = False
    ) -> None:
        super().__init__()
        self.module = module
        self.shared_input_data = shared_input_data
        self.n_tasks = n_tasks

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.shared_input_data:
            # Normalize to a single (batch, ...) tensor to run once.
            if X.dim() >= 1 and X.size(0) == self.n_tasks:
                x0 = X[0]  # assume shared across tasks; optionally assert equality in debug
            elif X.dim() >= 1 and X.size(0) == 1:
                x0 = X[0]
            else:
                x0 = X  # already (batch, ...)
            y = self.module(x0)  # run once
            return y.unsqueeze(0).expand(self.n_tasks, *y.shape)  # zero-copy broadcast
        else:
            # Standard behavior: per-task forward
            if X.size(0) != self.n_tasks:
                raise ValueError(f"Expected X.size(0) == n_tasks ({self.n_tasks}), got {X.size(0)}.")
            return torch.stack([self.module(x) for x in X], dim=0)


# ---------- Soft sharing (task-specific copies of a module) ----------

class SoftSharedModule(nn.Module):
    """
    Keep task-specific copies of a base module (soft sharing).
    If same_parameters=True, all task modules start with the *same values*.
    """

    def __init__(
            self,
            n_tasks: int,
            module_factory: ModuleFactory,
            *,
            same_parameters: bool = False,
            weight_init: str = "kaiming_uniform",
            bias_range: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_tasks = int(n_tasks)
        self.task_nets = nn.ModuleList([module_factory() for _ in range(self.n_tasks)])

        if same_parameters:
            # Create one randomized state_dict and load it into all task nets
            template_sd = random_state_dict_like(
                self.task_nets[0].state_dict(),
                weight_init=weight_init,
                bias_range=bias_range,
            )
            for net in self.task_nets:
                net.load_state_dict(template_sd)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X shape: (n_tasks, *...)
        if X.size(0) != self.n_tasks:
            raise ValueError(f"Expected X.size(0) == n_tasks ({self.n_tasks}), but got {X.size(0)}.")
        return torch.stack([self.task_nets[i](X[i]) for i in range(self.n_tasks)], dim=0)

# ---------- Multi-task model ----------

class ModuleSpec(TypedDict):
    shared: Optional[str]           # 'hard', 'soft', or None
    module: ModuleFactory           # factory that builds a fresh nn.Module

Layer = Tuple[str, ModuleSpec]

class MultiTaskModel(nn.Module):
    """
    A multi-task learning model with shared and task-specific layers.

    This class builds a model composed of:
      - Hard-shared layers: one module reused across all tasks.
      - Soft-shared layers: separate module copies for each task.

    Parameters
    ----------
    n_tasks : int
        Number of tasks.
    modules_layout : dict
        Ordered mapping from layer name to a ``ModuleSpec``:
        ``{"shared": "hard"|"soft"|None, "module": ModuleFactory}``.
        The order of insertion defines the model execution order.
    similarity_layers : dict
        Dictionary with keys ``"in"`` and ``"out"`` that specify
        the layer names delimiting the similarity sub-model.
        ``"in"`` can be ``None`` (start from the first layer).
    device : str or torch.device, default="cpu"
        Device where the model should be created.
    same_parameters : bool, default=False
        If True, initialize task-specific modules with identical parameters.
    shared_input_data : bool, default=False
        If True, a single input batch will be broadcasted to all tasks.
    weight_init : {"kaiming_uniform", "xavier_uniform", "normal_0_02"}, default="kaiming_uniform"
        Weight initialization method for randomized state dict copies.
    bias_range : float, default=0.1
        Range for uniform bias initialization.

    Attributes
    ----------
    model : nn.Sequential
        The complete multi-task model (hard + soft shared layers).
    model_similarity : nn.Sequential
        Submodel containing only the similarity layers.
    model_similarity_input : nn.Sequential
        Submodel containing the input part before the similarity layers.

    Examples
    --------
    Basic usage with one hard-shared trunk and two soft heads::

        >>> import torch
        >>> from torch import nn
        >>> modules_layout = {
        ...     "trunk": {"shared": "hard", "module": lambda: nn.Sequential(nn.Linear(10, 32), nn.ReLU())},
        ...     "head":  {"shared": "soft", "module":  lambda: nn.Linear(32, 1)},
        ... }
        >>> similarity_layers = {"in": "trunk", "out": "head"}
        >>> model = MultiTaskModel(n_tasks=2, modules_layout=modules_layout, similarity_layers=similarity_layers)
        >>> X = torch.randn(2, 8, 10)   # (tasks, batch, features)
        >>> Y = model(X)
        >>> Y.shape
        torch.Size([2, 8, 1])

    Using shared input broadcasting::

        >>> model = MultiTaskModel(
        ...     n_tasks=3,
        ...     modules_layout=modules_layout,
        ...     similarity_layers=similarity_layers,
        ...     shared_input_data=True,
        ... )
        >>> X = torch.randn(8, 10)  # single batch, no task dimension
        >>> Y = model(X)
        >>> Y.shape
        torch.Size([3, 8, 1])
    """

    def __init__(
            self,
            n_tasks: int,
            modules_layout: List[Layer],
            similarity_layers: Dict[str, Optional[str]],
            *,
            device: Union[str, torch.device] = "cpu",
            same_parameters: bool = False,
            shared_input_data: bool = False,
            weight_init: str = "kaiming_uniform",
            bias_range: float = 0.1,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.n_tasks = int(n_tasks)
        self.shared_input_data = bool(shared_input_data)

        self.modules_layout = self._build_modules_layout(modules_layout)

        if similarity_layers.get("out") is None:
            raise ValueError("similarity_layers['out'] must be provided.")
        self.similarity_layers = similarity_layers

        self.model = self._create_model(
            same_parameters=same_parameters,
            weight_init=weight_init,
            bias_range=bias_range,
        ).to(self.device)

        self.model_similarity = self.create_submodel(self.get_name_similarity_layers())
        self.model_similarity_input = self.create_submodel(self.get_name_input_similarity_layers())

    # ---- builders ----

    @staticmethod
    def _build_modules_layout(layers: List[Layer]) -> Dict[str, ModuleSpec]:
        # Validate: no duplicate names
        names = [name for name, _ in layers]
        assert len(names) == len(set(names)), f"Duplicate layer names: {names}"
        return dict(layers)

    def _create_model(
        self,
        *,
        same_parameters: bool,
        weight_init: str,
        bias_range: float,
    ) -> nn.Sequential:
        model = nn.Sequential()
        for module_name, spec in self.modules_layout.items():
            shared_type = spec.get("shared")
            factory = spec["module"]

            if shared_type == "hard":
                module_instance = SharedModule(
                    module=factory(),
                    n_tasks=self.n_tasks,
                    shared_input_data=self.shared_input_data
                )
            elif shared_type in ("soft", None):
                module_instance = SoftSharedModule(
                    n_tasks=self.n_tasks,
                    module_factory=factory,
                    same_parameters=same_parameters,
                    weight_init=weight_init,
                    bias_range=bias_range,
                )
            else:
                raise ValueError(f"Unknown shared type: {shared_type!r} for module {module_name!r}")

            model.add_module(name=module_name, module=module_instance)
        return model

    def create_submodel(self, layer_names: Optional[List[str]], model_input: Optional[nn.Sequential] = None) -> nn.Sequential:
        """
        Construct a shallow sequential that references submodules by name.
        """
        if model_input is None:
            model_input = self.model

        if layer_names is None:
            # Use Identity for "no-op" submodel, but keep as Sequential for uniformity
            return nn.Sequential(nn.Identity())

        sub = nn.Sequential()
        for name in layer_names:
            sub.add_module(name, model_input.get_submodule(name))
        return sub

    # ---- layer name helpers ----

    def _all_layer_names(self) -> List[str]:
        return list(self.modules_layout.keys())

    def get_name_task_specific_layers(self) -> List[str]:
        """
        Names of layers with soft sharing (task-specific copies).
        (Renamed from 'shared' to avoid confusion.)
        """
        return [name for name, spec in self.modules_layout.items() if spec["shared"] != "hard"]

    def get_name_soft_shared_layers(self) -> List[str]:
        """
        Names of layers with soft sharing (task-specific copies).
        (Renamed from 'shared' to avoid confusion.)
        """
        return [name for name, spec in self.modules_layout.items() if spec["shared"] == "soft"]

    def get_name_similarity_layers(self) -> List[str]:
        """
        Names of layers used for similarity computation (inclusive slice from 'in' to 'out').
        """
        layers = self._all_layer_names()

        out_idx = layers.index(self.similarity_layers["out"])
        in_name = self.similarity_layers.get("in")
        in_idx = layers.index(in_name) if in_name is not None else 0

        if not (0 <= in_idx <= out_idx < len(layers)):
            raise ValueError(f"Invalid similarity layer range: in={in_name}, out={self.similarity_layers['out']}")
        return layers[in_idx: out_idx + 1]

    def get_name_input_similarity_layers(self) -> Optional[List[str]]:
        """
        Names of layers *before* the 'in' similarity layer, or None if no 'in' provided.
        """
        in_name = self.similarity_layers.get("in")
        if in_name is None:
            return None
        layers = self._all_layer_names()
        in_idx = layers.index(in_name)
        return layers[:in_idx] if in_idx > 0 else None

    # ---- parameter utilities ----

    @staticmethod
    def _flatten_weights_from(module: nn.Module) -> torch.Tensor:
        parts = [p.flatten() for name, p in module.named_parameters() if "weight" in name]
        if not parts:
            # Return an empty tensor on the right device/dtype to keep shapes consistent
            return torch.empty(0, device=next(module.parameters(), torch.empty(0)).device)
        return torch.cat(parts, dim=0)

    def get_soft_shared_parameters_by_task(self, task: int) -> torch.Tensor:
        """
        Concatenate weights from all task-specific (soft) layers for a given task.
        """
        tensors: List[torch.Tensor] = []
        for layer_name in self.get_name_soft_shared_layers():
            layer = self.model.get_submodule(layer_name)
            assert isinstance(layer, SoftSharedModule)
            tensors.append(self._flatten_weights_from(layer.task_nets[task]))
        return torch.cat(tensors, dim=0) if tensors else torch.empty(0, device=self.device)

    def get_param_groups(self, tasks_groups: List[List[int]]) -> List[torch.Tensor]:
        """
        Stack parameter vectors for groups of tasks:
            shape -> (n_groups, group_size, n_params)
        """
        return [
            torch.stack([self.get_soft_shared_parameters_by_task(t) for t in group], dim=0)
            for group in tasks_groups
        ]

    # ---------- forwards ----------

    def _forward(self, X, model):
        """
        Forward pass for main model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the main model.
        """
        return model(X)

    def forward(self, X):
        """
        Forward pass for main model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the main model.
        """
        return self._forward(X, self.model)

    def forward_input_similarity(self, X):
        """
        Forward pass for input similarity sub-model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of input similarity sub-model.
        """
        return self._forward(X, self.model_similarity_input)

    def forward_similarity(self, X):
        """
        Forward pass for similarity sub-model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of similarity sub-model.
        """
        return self.model_similarity(X)

    # ---- per-task extractor ----

    def model_by_task(self, task: int) -> nn.ModuleDict:
        """
        Build per-task views of the model by picking the appropriate submodules.
        """
        model = nn.Sequential()
        for module_name, spec in self.modules_layout.items():
            sub = self.model.get_submodule(module_name)
            shared_type = spec["shared"]
            if isinstance(sub, SharedModule) or shared_type == "hard":
                model.add_module(module_name, sub.module)
            else:
                # SoftSharedModule
                assert isinstance(sub, SoftSharedModule)
                model.add_module(module_name, sub.task_nets[task])

        model_input_similarity = self.create_submodel(self.get_name_input_similarity_layers(), model)
        model_similarity = self.create_submodel(self.get_name_similarity_layers(), model)

        return nn.ModuleDict(
            {
                "model": model,
                "model_similarity": model_similarity,
                "model_input_similarity": model_input_similarity,
            }
        )

        # ---- optional: shape helper (fixed) ----

    @staticmethod
    def guess_output_shape(module: nn.Module, X: torch.Tensor) -> Tuple[torch.Size, torch.Tensor]:
        """
        Do a single forward to get output shape and the output itself.
        """
        y = module(X)
        return y.shape, y

