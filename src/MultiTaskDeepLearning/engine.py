"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from MultiTaskDeepLearning.model import MultiTaskModel
from MultiTaskDeepLearning.dataloader import MultitaskDataloader
from MultiTaskDeepLearning.tracking import Tracking
from MultiTaskDeepLearning.similarity.similarity import MultitaskSimilarity

def train_step(model: MultiTaskModel,
               dataloader: MultitaskDataloader,
               optimizer: torch.optim.Optimizer,
               max_batches: int = None,
               print_each_batch: int = None,
               track: Tracking = None) -> None:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.F
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    max_batches = len(dataloader) if max_batches is None else max_batches
    print_each_batch = 0 if print_each_batch is None else print_each_batch

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        y_pred = model(X)

        loss_mean = track.update_metrics(y, y_pred)
        loss_mean.backward()
        optimizer.step()

        if print_each_batch and (batch + 1) % print_each_batch  == 0:
            total_batches = len(dataloader)
            print(
                f"\nBatch {batch} of {total_batches} ({batch / total_batches * 100:.2f}%)"
            )

        if (batch + 1) == max_batches:
            break


def test_step(model: torch.nn.Module,
              dataloader: MultitaskDataloader,
              track: Tracking = None) -> None:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """

    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            # 1. Forward pass
            y_pred = model(X)

            track.update_metrics(y, y_pred)


def train(model: MultiTaskModel,
          train_dataloader: MultitaskDataloader,
          test_dataloader: MultitaskDataloader,
          multitask_similarity: MultitaskSimilarity,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          epochs: int,
          similarity_epochs: int,
          max_batches: int = None,
          print_each_batch: int = 1,
          std: float = None,
          track: Tracking = None,
          epochs_keep_similarity: float = 0
          ) -> None:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """

    similarity_epoch = 0
    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        track.start_epoch()

        track.train()
        with track:
            train_step(model=model,
                       dataloader=train_dataloader,
                       optimizer=optimizer,
                       max_batches=max_batches,
                       print_each_batch=print_each_batch,
                       track=track
                       )

        track.test()
        with track:
            test_step(model=model,
                      dataloader=test_dataloader,
                      track=track
                      )
            # scheduler.step(track.loss())

        with torch.inference_mode():
            calculate_similarity: bool = multitask_similarity and (epoch + 1) % similarity_epochs == 0

            track.ale()
            with track:
                if calculate_similarity:
                    if epoch > 0:
                        multitask_similarity.ale_curves._reinit()
                    multitask_similarity.ale_curves.update()

            track.similarity()
            with track:
                if calculate_similarity:
                    multitask_similarity.compute(std=std)
                    track.lossClass.update_tasks_groups(multitask_similarity.tasks_groups()[1])
                    similarity_epoch = 0
                else:
                    similarity_epoch += 1
                    if similarity_epoch >= epochs_keep_similarity:
                        track.lossClass.update_tasks_groups(None)

        track.print_info_epoch(calculate_similarity)

        track.save()
        if track.early_stopping(model):
            break
        #     multitask_similarity.ale_curves.step()
        #
        #     multitask_similarity.compute()
        #
        #     dist, indices = multitask_similarity.tasks_groups()
        #     track.update_similarityNew(epoch, dist, indices, multitask_similarity.ale_curves())
        #     track.save()
        #     break
        # else:
        #     track.save()


def train_model(config: dict) -> None:
    trainNew(
        model=config['model']['model'],
        train_dataloader=config['train']['dataloader'],
        test_dataloader=config['test']['dataloader'],
        multitask_similarity=config['model']['multitask_similarity'],
        loss_fn=config['loss_fn'],
        optimizer=config['optimizer']['optimizer'],
        scheduler=config['optimizer']['scheduler'],
        epochs=config['train']['epochs'],
        similarity_epochs=config['similarity']['each_epochs'],
        device=config['model']['device'],
        track=config['track'],
        config=config,
        data_transform=config['data']['data_transform']
    )

    # train(
    #     model=config['model']['model'],
    #     train_dataloader=config['train']['dataloader'],
    #     test_dataloader=config['test']['dataloader'],
    #     multitask_similarity=config['model']['multitask_similarity'],
    #     loss_fn=config['loss_fn'],
    #     optimizer=config['optimizer']['optimizer'],
    #     scheduler=config['optimizer']['scheduler'],
    #     epochs=config['train']['epochs'],
    #     device=config['model']['device'],
    #     similarity_each_batches=config['similarity']['each_batches'],
    #     track=config['track'],
    #     config=config
    # )

