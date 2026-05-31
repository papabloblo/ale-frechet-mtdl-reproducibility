import time
import torch
import os
from typing import Optional


class ElapsedTime:
    def __init__(self, id):
        self.id = id
        self._start_time = None
        self._end_time = None
        self.elapsed_times = []

    def start(self):
        self._start_time = time.time()

    def end(self):
        if self._start_time is None:
            return
        self._end_time = time.time()
        self.elapsed_times.append(self._end_time - self._start_time)
        self._start_time = None
        self._end_time = None

    def total_time(self, minutes: bool = False, avg: bool = False):
        if not self.elapsed_times:
            return 0.0
        val = (sum(self.elapsed_times) / len(self.elapsed_times)) if avg else sum(self.elapsed_times)
        return val / 60 if minutes else val

    def last_epoch(self, minutes: bool = False):
        last_time = self.elapsed_times[-1] if len(self.elapsed_times) > 0 else 0
        return last_time / 60 if minutes else last_time

    def print_epoch(self):
        print(f"[{self.id}] Last epoch: {self.last_epoch(minutes=True):.2f} minutes | "
              f"All epochs: {self.total_time(minutes=True):.2f} minutes")

class TrackALE_Similarity:
    def __init__(self, device: str = 'cpu', keep_epochs: Optional[int] = None):
        self.device = device
        self.keep_epochs = keep_epochs
        self.track: dict[int, torch.Tensor] = {}

    def update(self, epoch: int, result: torch.Tensor) -> None:
        self.track[epoch] = result.detach().to(self.device).clone()
        if self.keep_epochs is not None and self.keep_epochs > 0:
            while len(self.track) > self.keep_epochs:
                oldest_epoch = min(self.track.keys())
                del self.track[oldest_epoch]

    def for_save(self):
        return {epoch: res.cpu() for epoch, res in self.track.items()}

class TrackMetrics:
    def __init__(
            self,
            errors_names: list[str],
            device: str = 'cpu',
            keep_epochs: Optional[int] = None
    ) -> None:

        self.device = device
        self.keep_epochs = keep_epochs

        self.loss_per_task: list[Optional[torch.Tensor]] = []
        self.loss_penalty: list[Optional[torch.Tensor]] = []
        self.l2penalty: list[Optional[torch.Tensor]] = []
        self.errors: dict[str, list[Optional[torch.Tensor]]] = {k: [] for k in errors_names}
        self.batch_counts: list[int] = []

    def new_epoch(self) -> None:
        self.loss_per_task.append(None)
        self.loss_penalty.append(None)
        self.l2penalty.append(None)
        self.batch_counts.append(0)
        for key in self.errors.keys():
            self.errors[key].append(None)

        self._prune(self.loss_per_task)
        self._prune(self.loss_penalty)
        self._prune(self.l2penalty)
        self._prune(self.batch_counts)
        for k in self.errors:
            self._prune(self.errors[k])

    def _prune(self, lst):
        if self.keep_epochs is not None and 0 < self.keep_epochs < len(lst):
            del lst[0:len(lst) - self.keep_epochs]

    def _accumulate(
            self,
            store: list[Optional[torch.Tensor]],
            value: Optional[torch.Tensor | float]
    ) -> None:
        if value is None:
            return
        if torch.is_tensor(value):
            tensor = value.detach().to(self.device).clone()
        else:
            tensor = torch.as_tensor(value, device=self.device).clone()
        if store[-1] is None:
            store[-1] = tensor
        else:
            store[-1] += tensor

    def _epoch_mean(self, values: list[Optional[torch.Tensor]], epoch: int) -> torch.Tensor:
        value = values[epoch]
        count = self.batch_counts[epoch]
        if value is None or count <= 0:
            return torch.tensor(0.0, device=self.device)
        return value / count

    def update(
            self,
            loss_per_task: torch.Tensor,
            errors_per_task: dict[str, torch.Tensor],
            loss_penalty: torch.Tensor,
            l2_penalty: torch.Tensor | float | None = None
    ) -> None:

        self._accumulate(self.loss_per_task, loss_per_task.squeeze())
        self._accumulate(self.loss_penalty, loss_penalty)
        self._accumulate(self.l2penalty, l2_penalty)
        self.batch_counts[-1] += 1

        for key, error in self.errors.items():
            self._accumulate(self.errors[key], errors_per_task[key])

    def loss_epoch(self, epoch: int):
        ep = self._epoch_mean(self.loss_per_task, epoch)
        return ep.mean() if ep.ndim > 0 else ep

    def last_loss(self):
        return self.loss_epoch(-1)

    def for_save(self):
        return {
            'loss_per_task': self.to_cpu_tensor(self.loss_per_task, self.batch_counts),
            'loss_penalty': self.to_cpu_tensor(self.loss_penalty, self.batch_counts),
            'l2penalty': self.to_cpu_tensor(self.l2penalty, self.batch_counts),
            'errors': {error: self.to_cpu_tensor(v, self.batch_counts) for error, v in self.errors.items()}
        }

    def info(self, epoch: int) -> dict:
        errors = {key: self._epoch_mean(value, epoch).mean() for key, value in self.errors.items()}
        errors['LOSS'] = self.loss_epoch(epoch)
        return errors

    # helper----
    @staticmethod
    def to_cpu_tensor(
            lst: list[Optional[torch.Tensor]],
            batch_counts: list[int]
    ) -> torch.Tensor:
        if not lst:
            return torch.empty(0, dtype=torch.float32)

        first = next((tensor for tensor in lst if tensor is not None), None)
        if first is None:
            return torch.zeros(len(lst), dtype=torch.float32)

        means = []
        zero = torch.zeros_like(first, device=first.device)
        for tensor, count in zip(lst, batch_counts):
            if tensor is None or count <= 0:
                means.append(zero)
            else:
                means.append(tensor / count)
        return torch.stack(means).cpu()


class Tracker:
    def __init__(
            self,
            errors_names: list[str],
            device: str = 'cpu',
            keep_epochs: Optional[int] = None,
            path: Optional[str] = None,
            config_info: Optional[dict] = None
    ):

        self.device = device
        self.keep_epochs = keep_epochs
        self.current_epoch = 0
        self.path = path

        self.config_info = config_info

        self.track = {
            'train': {
                'timing': ElapsedTime('train'),
                'metrics': TrackMetrics(errors_names, device=self.device),
            },
            'validation': {
                'timing': ElapsedTime('test'),
                'metrics': TrackMetrics(errors_names, device=self.device),
            },
            'test': {
                'timing': ElapsedTime('test'),
                'metrics': TrackMetrics(errors_names, device=self.device),
            },
            'ale': {
                'timing': ElapsedTime('ale'),
                'metrics': TrackALE_Similarity(device=self.device, keep_epochs=self.keep_epochs)
            },
            'similarity': {
                'timing': ElapsedTime('similarity'),
                'metrics': TrackALE_Similarity(device=self.device, keep_epochs=self.keep_epochs)
            }
        }

        self.best_epoch: int = -1
        self.best_model_parameters: Optional[dict] = None
        self.best_val_loss_value: Optional[float] = None  # robust to pruning

    # Start and end methods ------------------------------------
    def start_epoch(self):
        self.current_epoch += 1

    def start_train(self):
        self.track['train']['metrics'].new_epoch()
        self.track['train']['timing'].start()

    def end_train(self):
        self.track['train']['timing'].end()

    def start_validation(self):
        self.track['validation']['metrics'].new_epoch()
        self.track['validation']['timing'].start()

    def end_validation(self):
        self.track['validation']['timing'].end()

    def start_test(self):
        self.track['test']['metrics'].new_epoch()
        self.track['test']['timing'].start()

    def end_test(self):
        self.track['test']['timing'].end()

    def start_ale(self):
        self.track['ale']['timing'].start()

    def end_ale(self, epoch: int = None, ale: torch.Tensor = None) -> None:
        if epoch is not None:
            self.track['ale']['metrics'].update(epoch, ale)
        self.track['ale']['timing'].end()

    def start_similarity(self):
        self.track['similarity']['timing'].start()

    def end_similarity(self, epoch: int = None, similarity: torch.Tensor = None) -> None:
        if epoch is not None:
            self.track['similarity']['metrics'].update(epoch, similarity)
        self.track['similarity']['timing'].end()

    def update_metrics(
            self,
            phase: str,
            loss_per_task: torch.Tensor,
            errors_per_task: dict[str, torch.Tensor],
            penalty: torch.Tensor,
            l2_penalty: Optional[torch.Tensor | float] = None
    ) -> None:
        if phase not in self.track:
            raise KeyError(f"Unknown phase '{phase}'. Valid: {['train', 'test']}")
        self.track[phase]['metrics'].update(loss_per_task, errors_per_task, penalty, l2_penalty)

    # Early stopping methods ------------------------------------

    def last_val_loss(self) -> torch.Tensor:
        return self.track['validation']['metrics'].last_loss()

    def best_val_loss(self) -> Optional[float]:
        return self.best_val_loss_value

    def update_best_model(self, params: dict, epoch: int, best_loss_value: float) -> None:
        self.best_epoch = epoch
        self.best_val_loss_value = float(best_loss_value)
        self.best_model_parameters = {k: v.detach().cpu().clone() for k, v in params.items()}

    def best_model(self) -> dict:
        return dict(
            epoch= self.best_epoch,
            train_metrics=self.track['train']['metrics'].info(self.best_epoch),
            val_metrics=self.track['validation']['metrics'].info(self.best_epoch),
            test_metrics=self.track['test']['metrics'].info(epoch=0),
            state_dict= self.best_model_parameters
        )

    # Save methods ----------------------------------------------
    def _ensure_path(self):
        if not self.path:
            raise ValueError("Tracker.path is not set. Provide a directory before saving.")
        os.makedirs(self.path, exist_ok=True)

    def save_times(self) -> None:
        self._ensure_path()
        times = {mode: item['timing'].elapsed_times for mode, item in self.track.items()}
        torch.save(times, os.path.join(self.path, "times.pth"))

    def save_metrics(self) -> None:
        self._ensure_path()
        metrics = {mode: item['metrics'].for_save() for mode, item in self.track.items()}
        torch.save(metrics, os.path.join(self.path, "metrics.pth"))

    def save_best_model(self) -> None:
        self._ensure_path()
        torch.save(self.best_model(),
                   os.path.join(self.path, "best_model.pth")
                   )

    def save_config(self) -> None:
        self._ensure_path()
        if self.config_info is not None:
            torch.save(self.config_info, os.path.join(self.path, "config.pth"))

    def save(self) -> None:
        self.save_times()
        self.save_metrics()
        self.save_best_model()
        self.save_config()

    # Load  methods ----------------------------------------------
