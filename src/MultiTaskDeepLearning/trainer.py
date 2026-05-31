"""
Utilities for training, evaluating, and monitoring a multi-task PyTorch model.

This module provides:
- A user-friendly printer/banner helper (``_MultiTaskTrainerPrint``) to
  display execution settings, per-epoch metric tables, and timing summaries.
- The main trainer class (``MultiTaskTrainer``) that encapsulates a standard
  train/test loop with optional ALE curve computation and task-similarity
  tracking, early stopping, checkpointing via an external ``Tracker`` object,
  and optional LR scheduling.

Typical usage example:
    trainer = MultiTaskTrainer(
        model, train_loader, test_loader, optimizer, loss,
        ale=ale_helper, multitask_similarity=sim_helper, scheduler=sched,
        early_stopping_epochs=20, print_each_epochs=1, ...
    )
    trainer.train(epochs=100)
"""

import time, math
import torch
import copy
from contextlib import nullcontext

from MultiTaskDeepLearning.model import MultiTaskModel
from MultiTaskDeepLearning.dataloader import MultitaskDataloader
from MultiTaskDeepLearning.similarity.ale import MultiTaskALE
from MultiTaskDeepLearning.tracking import Tracker
from MultiTaskDeepLearning.similarity.similarity import MultitaskSimilarity
from MultiTaskDeepLearning.loss import MultiTaskLoss


class _MultiTaskTrainerPrint:
    """Pretty-printer and banners for training sessions.

       This helper formats and prints run configuration, per-epoch metric tables,
       and timing summaries for the different tracked phases (train, test, ALE,
       similarity, etc.).

       Args:
           learning_type: Short description of the training regime (e.g. "MTL").
           dataset_name: Human-readable dataset identifier.
           total_epochs: Total number of epochs to train (shown in headers).
           n_tasks: Number of prediction tasks in the model.
           logging_dir: Path where external tracking artifacts are stored.
           seed: Random seed used for the run.
           early_stopping_rounds: Patience used for early stopping.
           n_intervals_ale: Number of intervals used for ALE computation.
           learning_rate: Optimizer learning rate (for display only).
           similarity_each_epochs: Frequency (epochs) to compute similarity.
           train_batch_size: Train DataLoader batch size.
           test_batch_size: Test DataLoader batch size.
           ale_batch_size: Batch size used during ALE computation.
           l2penalty: L2 regularization coefficient used in the loss.
           architecture: Text label for the model architecture.
       """

    def __init__(
            self,
            learning_type: str,
            dataset_name: str,
            total_epochs: int,
            n_tasks: int,
            logging_dir: str,
            seed: int,
            early_stopping_rounds: int,
            n_intervals_ale: int,
            learning_rate: float,
            similarity_each_epochs: int | None,
            train_batch_size: int,
            test_batch_size: int,
            ale_batch_size: int,
            l2penalty: float,
            architecture: str
    ):

        self.current_epoch = 0
        self.dataset_name = dataset_name
        self.learning_type = learning_type
        self.total_epochs = total_epochs
        self.logging_dir = logging_dir
        self.n_tasks = n_tasks
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        self.n_intervals_ale = n_intervals_ale
        self.learning_rate = learning_rate
        self.similarity_each_epochs = similarity_each_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.ale_batch_size = ale_batch_size
        self.l2penalty = l2penalty
        self.architecture = architecture

        self.info, self.max_char, self.separator, self.blank_line = self._create_info()

    def _create_info(self):
        """Builds the formatted info dictionary and line helpers.

        Returns:
           tuple[dict[str, str], int, str, str]: A tuple containing:
               - info: Mapping of info keys to padded printable strings.
               - max_char: The computed maximum line width.
               - separator: Horizontal line string of width ``max_char``.
               - blank_line: A bordered blank line with width ``max_char``.
        """
        info = dict(
            execution_information="EXECUTION INFORMATION",
            learning_type=f"Learning type: {self.learning_type}",
            architecture=f"Architecture: {self.architecture}",
            dataset=f"Dataset: {self.dataset_name}",
            num_epochs=f"Number of epochs: {self.total_epochs}",
            batch_size_train=f"Train batch size: {self.train_batch_size}",
            batch_size_test=f"Test batch size: {self.test_batch_size}",
            batch_size_ale=f"ALE batch size: {self.ale_batch_size}",
            similarity_each_epochs=f"Compute similarity each epochs: {self.similarity_each_epochs}",
            learning_rate=f"Learning rate: {self.learning_rate}",
            l2penalty=f"L2 penalty: {self.l2penalty}",
            n_intervals_ale=f"Number of ALE intervals: {self.n_intervals_ale}",
            early_stopping_rounds=f"Early stopping rounds: {self.early_stopping_rounds}",
            seed=f"Seed: {self.seed}",
            n_tasks=f"Number of tasks: {self.n_tasks}",
            log_dir=f"Logging directory: {self.logging_dir}",
            execution_date=f"Execution datetime: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        max_char = max(len(x) for x in info.values()) + 4
        separator = "-" * max_char
        blank_line = "|" + " " * (max_char - 2) + "|"

        info = {key: "| " + value + " " * (max_char - 4 - len(value)) + " |" for key, value in
                info.items()}

        return info, max_char, separator, blank_line

    def info_model(self):
        """
        Prints a banner with the execution configuration.
        """
        self.info, self.max_char, self.separator, self.blank_line = self._create_info()
        print()
        print(self.separator)
        print(self.blank_line)

        print(self.info['execution_information'])

        print(self.blank_line)
        print(self.separator)

        print(self.info['learning_type'])

        print(self.blank_line)

        print(self.info['architecture'])
        print(self.info['dataset'])

        print(self.blank_line)

        print(self.info['n_tasks'])
        print(self.info['num_epochs'])
        print(self.info['batch_size_train'])
        print(self.info['batch_size_ale'])
        print(self.info['batch_size_test'])
        print(self.info['similarity_each_epochs'])

        print(self.blank_line)

        print(self.info['learning_rate'])
        print(self.info['l2penalty'])

        print(self.blank_line)

        print(self.info['n_intervals_ale'])
        print(self.info['early_stopping_rounds'])

        print(self.blank_line)

        print(self.info['seed'])

        print(self.separator)

        print(self.info['log_dir'])
        print(self.info['execution_date'])

        print(self.separator)
        print()

    def start_epoch_banner(self, epoch: int, total_epochs: int) -> None:
        """Prints a centered banner announcing the start of an epoch.

        Args:
            epoch: Zero-based epoch index to show.
            total_epochs: Total number of epochs for the session.
        """
        print_info = f"EPOCH {epoch + 1} OF {total_epochs}"
        spaces = self.max_char - len(print_info) - 2
        print()
        print(self.separator)
        print('|' + " " * (spaces // 2) + print_info + " " * (spaces - spaces // 2) + '|')
        print(self.separator)

    def best_model_banner(self, best_model) -> None:
        """Prints a banner announcing the best model found during training.

        Args:
        """

        print(f"BEST MODEL FOUND AT EPOCH {best_model['epoch'] + 1}")

        metric_names = list(best_model['train_metrics'].keys())
        for metric in metric_names:
            print(f'{metric}:')
            print(f'\tTrain: {best_model["train_metrics"][metric]:.4f}')
            print(f'\tValidation: {best_model["val_metrics"][metric]:.4f}')
            print(f'\tTest: {best_model["test_metrics"][metric]:.4f}')

    @staticmethod
    def epoch_table(
            track,
            metric_names: list[str],
            from_epoch: int,
            to_epoch: int,
            similarity_each_epochs: int | None = None
    ):
        """Prints a compact table with train/test metrics for a range of epochs.

        Args:
            track: Tracker internal state (``Tracker.track``).
            metric_names: Ordered list of metric names to display. Typically
                ``['LOSS', <error_1>, <error_2>, ...]``.
            from_epoch: Inclusive starting epoch index to print.
            to_epoch: Inclusive ending epoch index to print.
        """
        header = f"|  {' ':^8}  |" + "".join(f"  {m:^19}  |" for m in metric_names)
        separator = "-" * len(header)
        print("+-----------+")
        print("|  METRICS  |")
        print(separator)
        print(header)
        print(separator)
        header2 = f"|  {'EPOCH':^8}  |" + f"  {'Train':^8} | {'Validation':^8}  |" * len(metric_names)
        print(header2)
        print(separator)
        for ep in range(from_epoch, to_epoch + 1):
            if similarity_each_epochs is not None and ep % similarity_each_epochs == 0:
                print(separator)
            print(f"|  {ep + 1:^8}  |", end="")
            mtr_train = track['train']['metrics'].info(ep)
            mtr_val = track['validation']['metrics'].info(ep)
            for m in metric_names:
                print(f"  {mtr_train[m]:^8.4f} | {mtr_val[m]:^8.4f}  |", end="")
            print()
        print(separator)
        print()

    @staticmethod
    def times(track, epoch, total_epochs: int):
        """Prints elapsed, average, and estimated times for tracked phases.

        Args:
            track: Tracker internal state (``Tracker.track``).
            epoch: Current zero-based epoch index.
            total_epochs: Total number of epochs planned.
        """
        header = f"|  {' ':^25}  |  CURRENT  EPOCH  |  {'ALL EPOCHS':^14}  |"
        separator = "-" * len(header)
        print("+-----------------+");
        print("|  ELAPSED TIMES  |")
        print(separator);
        print(header);
        print(separator)
        steps = [track[k]['timing'] for k in track.keys()]
        for st in steps:
            print(f"|  {st.id:^25}  |  {st.last_epoch(minutes=True):^6.2f} minutes  "
                  f"|  {st.total_time(minutes=True):^6.2f} minutes  |")
        print(separator)
        total_current = sum(st.last_epoch(minutes=True) for st in steps)
        total_all = sum(st.total_time(minutes=True) for st in steps)
        print(f'|  {"TOTAL TIME":^25}  |  {total_current:^6.2f} minutes  |  {total_all:^6.2f} minutes  |')
        avg_time = total_all / (epoch + 1)
        est_time = total_all * (total_epochs / (epoch + 1))
        print(f'|  {"AVERAGE TIME":^25}  |  {" ":14}  |  {avg_time:^6.2f} minutes  |')
        print(f'|  {"ESTIMATED TIME":^25}  |  {" ":14}  |  {est_time:^6.2f} minutes  |')
        print(separator)


class MultiTaskTrainer:
    """End-to-end trainer for multi-task PyTorch models.

    This class runs the canonical train/test loop, tracks metrics and times via
    ``Tracker``, optionally computes ALE curves and task-feature similarity, and
    supports early stopping, checkpoints (via ``Tracker.save()``), and a step
    LR scheduler.

    Args:
        model: The multi-task model to train/evaluate.
        train_dataloader: Dataloader used for training.
        test_dataloader: Dataloader used for evaluation.
        optimizer: Optimizer used during training.
        loss: Multi-task loss object that also exposes per-task errors
            and regularization penalty.
        ale: Optional ALE computer. If provided and scheduled, ALE curves
            will be updated and recorded.
        multitask_similarity: Optional similarity helper to compute task
            similarity and produce task groups for the loss.
        scheduler: Optional learning rate scheduler stepped once per epoch.
        maximize_loss: If True, "improvement" means a larger loss; otherwise
            smaller is better (default).
        early_stopping_epochs: Patience (epochs) with no improvement before
            stopping early. Use ``float('inf')`` to disable.
        print_each_epochs: Frequency to print metrics tables and times.
        ale_each_epochs: Frequency (epochs) to compute ALE. If ``None``, skip.
        similarity_each_epochs: Frequency (epochs) to compute similarity.
        keep_similarity_epochs: Number of epochs after a similarity computation
            to keep the inferred task groups in the loss; after that, groups
            are cleared (if supported by the loss).
        track_device: Device for the ``Tracker`` tensors/aggregations.
        track_epochs: Optional limit of epochs kept in memory by ``Tracker``.
        save_results_each: Frequency (epochs) to checkpoint via ``Tracker.save()``.
        logging_dir: Path displayed in the printer banner (for context only).
        learning_type: Display label for training regime (printer banner).
        dataset_name: Display dataset name (printer banner).
        n_intervals_ale: Number of intervals for ALE plots (printer banner).
        learning_rate: Learning rate to display (printer banner).
        train_batch_size: Train batch size to display (printer banner).
        test_batch_size: Test batch size to display (printer banner).
        ale_batch_size: ALE batch size to display (printer banner).
        l2penalty: L2 penalty coefficient to display (printer banner).
        architecture: Model architecture label to display (printer banner).
        seed: Random seed to display (printer banner).
    """

    def __init__(
            self,
            model: MultiTaskModel,
            train_dataloader: MultitaskDataloader,
            validation_dataloader: MultitaskDataloader,
            test_dataloader: MultitaskDataloader,
            optimizer: torch.optim.Optimizer,
            loss: MultiTaskLoss,
            ale: MultiTaskALE | None = None,
            multitask_similarity: MultitaskSimilarity | None = None,
            scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
            maximize_loss: bool = False,
            early_stopping_epochs: int = float('inf'),
            print_each_epochs: int = 1,
            ale_each_epochs: int | None = None,
            similarity_each_epochs: int | None = None,
            keep_similarity_epochs: int = 0,
            track_device: str = 'cpu',
            track_epochs: int | None = None,
            save_results_each: int | None = None,
            logging_dir: str = "",
            learning_type: str = "",
            dataset_name: str = "",
            n_intervals_ale: int = 0,
            learning_rate: float = 0.0,
            train_batch_size: int = 0,
            test_batch_size: int = 0,
            ale_batch_size: int = 0,
            l2penalty: float = 0.0,
            architecture: str = "",
            seed: int = 0,
            print_limit_epochs: int = 5,
            config_info: dict | None = None,
            amp: bool = False,
            amp_dtype: str = 'float16'
    ):

        self.early_stopping_epochs = early_stopping_epochs
        self.model = model

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.ale = ale
        self.multitask_similarity = multitask_similarity

        self.tracking = Tracker(
            errors_names=list(self.loss.errors_dict.keys()),
            device=track_device,
            keep_epochs=track_epochs,
            path=logging_dir,
            config_info=config_info
        )

        self.maximize_loss = maximize_loss

        # Compute ale/similarity every n epochs
        self.ale_each_epochs = ale_each_epochs
        self.similarity_each_epochs = similarity_each_epochs
        self.keep_similarity_epochs = keep_similarity_epochs
        self.print_each_epochs = print_each_epochs
        self.print_limit_epochs = print_limit_epochs

        self.printer = _MultiTaskTrainerPrint(
            learning_type=learning_type,
            dataset_name=dataset_name,
            total_epochs=0,  # will set in train()
            n_tasks=self.model.n_tasks,
            logging_dir=logging_dir,
            seed=seed,
            early_stopping_rounds=early_stopping_epochs,
            n_intervals_ale=n_intervals_ale,
            learning_rate=learning_rate,
            similarity_each_epochs=similarity_each_epochs,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            ale_batch_size=ale_batch_size,
            l2penalty=l2penalty,
            architecture=architecture
        )

        self.save_results_each = save_results_each
        self._best_seen: float | None = None  # for early stop compare
        self.amp = bool(amp and torch.cuda.is_available() and str(self.model.device).startswith('cuda'))
        self.amp_dtype = str(amp_dtype).lower()

    def train(self, epochs: int = None, max_batches: int | None = None) -> None:
        """Runs the training loop with evaluation, ALE, and similarity steps.

        Passes the model through ``train_step()`` and ``test_step()`` each epoch,
        optionally computing ALE curves and task similarity. Metrics and times
        are accumulated in the internal ``Tracker`` and periodically printed.

        Early stopping is triggered if no improvement is observed on test loss
        for ``early_stopping_epochs`` epochs.

        Args:
            epochs: Number of epochs to train. Must be a positive integer.
            max_batches: Optional cap on the number of batches per phase per
                epoch (useful for quick runs or debugging).

        Raises:
            AssertionError: If ``epochs`` is not a positive integer.
        """
        assert isinstance(epochs, int) and epochs > 0, "epochs must be a positive integer"
        self.printer.total_epochs = epochs  # set banner total
        self.printer.info_model()

        for epoch in range(epochs):
            self.tracking.start_epoch()

            self.train_step(max_batches)
            self.validation_step(max_batches)
            self.ale_step(epoch)
            self.similarity_step(epoch)

            # progress print
            if ((epoch + 1) % self.print_each_epochs) == 0:
                self.printer.start_epoch_banner(epoch, epochs)
                # derive metric names from the loss dict
                metric_names = ['LOSS'] + list(self.loss.errors_dict.keys())
                from_epoch = max(0, epoch - (self.print_limit_epochs - 1))
                self.printer.epoch_table(
                    self.tracking.track,
                    metric_names,
                    from_epoch,
                    epoch,
                    self.similarity_each_epochs
                )
                self.printer.times(self.tracking.track, epoch, epochs)

            # track best & early stop
            current_loss_t = self.tracking.last_val_loss()
            current_loss = float(current_loss_t.item()) if torch.is_tensor(current_loss_t) else float(
                current_loss_t)
            improved = (self._best_seen is None) or \
                       (current_loss > self._best_seen if self.maximize_loss else current_loss < self._best_seen)

            if improved:
                self._best_seen = current_loss
                self.tracking.update_best_model(self.model.state_dict(), epoch, best_loss_value=current_loss)
            elif (epoch - self.tracking.best_epoch) >= self.early_stopping_epochs:
                print(f"Early stopping at epoch {epoch + 1}. "
                      f"No improvement for {self.early_stopping_epochs} epochs.")

                break

            # checkpoint
            if self.save_results_each is not None and ((epoch + 1) % self.save_results_each == 0):
                self.tracking.save()

            if self.scheduler is not None:
                self.scheduler.step(current_loss)

        self.test_step()
        self.tracking.save()
        self.printer.best_model_banner(self.tracking.best_model())




    # Steps ----------------------------------------------
    def train_step(self, max_batches: int | None = None) -> None:
        """Performs one training phase for the current epoch.

        Accumulates loss/errors/penalties and updates the optimizer at each
        batch. Metrics are recorded via ``Tracker``.

        Args:
            max_batches: Optional cap on the number of batches to process.
        """
        self.tracking.start_train()

        self.model.train()
        self._step(phase='train', model=self.model, dataloader=self.train_dataloader, max_batches=max_batches)

        self.tracking.end_train()

    def test_val_step(self, phase: str, model, dataloader, max_batches: int | None = None) -> None:
        """Performs one evaluation phase for the current epoch.

        Runs the forward pass without gradient tracking and records metrics
        via ``Tracker``.

        Args:
            max_batches: Optional cap on the number of batches to process.
        """
        with torch.no_grad():
            model.eval()
            self._step(phase=phase, model=model, dataloader=dataloader, max_batches=max_batches)

    def validation_step(self, max_batches: int | None = None) -> None:
        self.tracking.start_validation()
        self.test_val_step(phase='validation', model=self.model, dataloader=self.validation_dataloader, max_batches=max_batches)
        self.tracking.end_validation()

    def test_step(self, max_batches: int | None = None) -> None:
        self.tracking.start_test()
        if self.tracking.best_model_parameters is not None:
            self.model.load_state_dict(self.tracking.best_model_parameters)
        self.test_val_step(phase='test', model=self.model, dataloader=self.test_dataloader, max_batches=max_batches)
        self.tracking.end_test()

    def _step(
            self,
            phase: str,
            dataloader: MultitaskDataloader,
            model: MultiTaskModel,
            max_batches: int | None = None
    ) -> None:
        """Runs a single phase (train or test) over the dataloader.

        Handles forward pass, loss/penalty/error computation, backpropagation
        and optimizer step (only during training), and updates the tracker.

        Args:
            phase: Either ``'train'`` or ``'test'``.
            max_batches: Optional cap on number of batches to iterate.

        Raises:
            ValueError: If ``phase`` is not ``'train'`` or ``'test'``.
        """

        max_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))

        # Loop through data loader data batches
        amp_dtype = torch.bfloat16 if self.amp_dtype == 'bfloat16' else torch.float16

        for batch, (X, y) in enumerate(dataloader):
            if phase == 'train':
                self.optimizer.zero_grad()

            X = X.to(model.device, non_blocking=True)
            y = y.to(model.device, non_blocking=True)

            with (torch.autocast(device_type='cuda', dtype=amp_dtype) if self.amp else nullcontext()):
                y_pred = model(X)

                loss_per_task = self.loss.loss_per_task(y, y_pred)
                penalty = self.loss.penalty()

                errors_per_task = self.loss.errors_per_task(y, y_pred)
                l2_penalty = self.loss.l2_penalty

            self.tracking.update_metrics(phase, loss_per_task, errors_per_task, penalty, l2_penalty)

            if phase == 'train':
                # make sure both parts are scalars
                mean_loss = loss_per_task.mean()
                l2 = l2_penalty.item() if torch.is_tensor(l2_penalty) else float(l2_penalty)

                # Similarity-weighted penalty (only for methods that provide it)
                if self.multitask_similarity is not None:
                    similarity, _ = self.multitask_similarity.tasks_groups()
                    # keep on same device as penalty
                    if torch.is_tensor(similarity):
                        similarity = similarity.to(penalty.device)
                    penalty = penalty * similarity

                reg = l2 * (penalty.sum() if torch.is_tensor(penalty) else float(penalty))
                total = mean_loss + reg
                total.backward()
                self.optimizer.step()

            if (batch + 1) >= max_batches:
                break

    @torch.no_grad()
    def ale_step(self, epoch: int) -> None:
        """Optionally computes and records ALE curves at a scheduled frequency.

        If ``ale_each_epochs`` is set and ``ale`` is provided, this updates and
        stores ALE curves in the ``Tracker`` for the current epoch.

        Args:
            epoch: Zero-based current epoch index.
        """
        self.tracking.start_ale()
        if self.ale_each_epochs and self.ale and ((epoch + 1) % self.ale_each_epochs == 0):
            # Check that ALE curves are initialized here
            self.ale.update()
            self.tracking.end_ale(epoch, self.ale())
        else:
            self.tracking.end_ale()

    @torch.no_grad()
    def similarity_step(self, epoch: int) -> None:
        """Optionally computes and records task-feature similarity.

        Computes similarity at the configured frequency; if the loss exposes
        ``update_tasks_groups(groups)``, it will receive the new grouping and
        subsequently be cleared according to ``keep_similarity_epochs``.

        Args:
            epoch: Zero-based current epoch index.
        """
        self.tracking.start_similarity()
        if self.similarity_each_epochs and self.multitask_similarity and ((epoch + 1) % self.similarity_each_epochs == 0):
            self.multitask_similarity.compute()
            # update grouping in the loss if available
            if hasattr(self.loss, "update_tasks_groups"):
                _, groups = self.multitask_similarity.tasks_groups()
                self.loss.update_tasks_groups(groups)
            # store similarity
            self.tracking.end_similarity(epoch, self.multitask_similarity.similarity_tasks_features)
        elif self.similarity_each_epochs and self.keep_similarity_epochs is not None \
                and ((epoch + 1) % self.similarity_each_epochs) >= self.keep_similarity_epochs:
            if hasattr(self.loss, "update_tasks_groups"):
                self.loss.update_tasks_groups(None)
            self.tracking.end_similarity()
        else:
            self.tracking.end_similarity()

    # Helpers ----------------------------------------------
    def _start_epoch(self, epoch: int, total_epochs: int) -> None:
        """Starts timing/trackers for an epoch and prints the epoch banner.

        Args:
            epoch: Zero-based current epoch index.
            total_epochs: Total number of epochs planned.
        """
        self.tracking.start_epoch()
        self.printer.start_epoch_banner(epoch, total_epochs)

    def get_best_model(self) -> dict:
        """Returns the best model metrics and parameters recorded by the tracker.

        Returns:
            dict: State dictionary of the best model seen during training.
        """
        return self.tracking.best_model()
