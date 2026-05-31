import torch
import torch.nn as nn
from torch import linalg as LA
from MultiTaskDeepLearning.model import MultiTaskModel
from typing import Callable

class MultiTaskLoss:
    def __init__(self,
                 model: MultiTaskModel,
                 loss_fn: torch.nn,
                 errors_fn: dict[str, Callable],
                 l2_penalty: float = 1e-3
                 ):
        """
        :type model: object
        """
        self.tasks_groups = None
        self.l2_penalty = l2_penalty
        self.loss_fn = loss_fn
        self.model = model
        self.errors_dict = errors_fn

    def update_tasks_groups(self, tasks_groups):
        self.tasks_groups = tasks_groups

    def penalty(self):
        if self.tasks_groups is not None:
            parameters = torch.stack(self.model.get_param_groups(self.tasks_groups))
            return LA.vector_norm(parameters[:, 0] - parameters[:, 1], dim=1)
        else:
            return torch.Tensor([0.0]*self.model.n_tasks).to(self.model.device)

    def loss_per_task(self, real: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(real, pred).mean(dim=1)

    @torch.no_grad()
    def errors_per_task(self, real: torch.Tensor, pred: torch.Tensor) -> dict[str, torch.Tensor]:
        return {key: error(real, pred) for key, error in self.errors_dict.items()}


def accuracy2(y_pred, y, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred_threshold = torch.greater_equal(y_pred, threshold)
    acc = torch.eq(y_pred_threshold, y).to(torch.float)

    return acc.mean(dim=0)


def accuracy(y_pred, y, threshold=0.5):
    # Apply sigmoid activation and threshold in a single step
    y_pred = (torch.sigmoid(y_pred) >= threshold)

    # Calculate accuracy by comparing predictions with ground truth
    acc = (y_pred == y).float().mean(dim=0)

    return acc


def confusion_matrix2(y_pred, y, threshold=0.5):
    conf_matrix = {'true_positives': torch.zeros(y.shape[1]),
                   'true_negatives': torch.zeros(y.shape[1]),
                   'false_positives': torch.zeros(y.shape[1]),
                   'false_negatives': torch.zeros(y.shape[1])
                   }

    y_pred = torch.sigmoid(y_pred)
    y_pred_threshold = torch.greater_equal(y_pred, threshold)
    for attr in range(y.shape[1]):
        conf_matrix['true_positives'][attr] = y_pred_threshold[:, attr][y[:, attr] == 1.].sum().item()
        conf_matrix['true_negatives'][attr] = torch.logical_not(y_pred_threshold[:, attr][y[:, attr] == 1.]).sum().item()

        conf_matrix['false_positives'][attr] = y_pred_threshold[:, attr][y[:, attr] == 0.].sum().item()
        conf_matrix['false_negatives'][attr] = torch.logical_not(y_pred_threshold[:, attr][y[:, attr] == 0.]).sum().item()

    return conf_matrix


def confusion_matrix(y_pred, y, threshold=0.5):
    # Initialize confusion matrix tensors for each metric across all attributes
    conf_matrix = {
        'true_positives': torch.zeros(y.shape[1]),
        'true_negatives': torch.zeros(y.shape[1]),
        'false_positives': torch.zeros(y.shape[1]),
        'false_negatives': torch.zeros(y.shape[1])
    }

    # Apply sigmoid activation and thresholding
    y_pred = torch.sigmoid(y_pred) >= threshold

    # Compute confusion matrix components in a vectorized manner
    conf_matrix['true_positives'] = (y_pred & y.bool()).sum(dim=0).float()
    conf_matrix['true_negatives'] = ((~y_pred) & (y.bool())).sum(dim=0).float()
    conf_matrix['false_positives'] = (y_pred & (~y.bool())).sum(dim=0).float()
    conf_matrix['false_negatives'] = ((~y_pred) & (~y.bool())).sum(dim=0).float()

    return conf_matrix


def f1_score(y_pred, y, threshold=0.5):

    conf_matrix = confusion_matrix(y_pred, y, threshold)
    # precision = conf_matrix['true_positives'] / (conf_matrix['true_positives'] + conf_matrix['false_positives'])
    # recall = conf_matrix['true_positives'] / (conf_matrix['true_positives'] + conf_matrix['false_negatives'])
    #
    # f1 = 2*precision*recall / (precision + recall)

    f1 = 2*conf_matrix['true_positives'] / (2*conf_matrix['true_positives'] + conf_matrix['false_positives'] + conf_matrix['false_negatives'])

    return f1


def mape_loss(real, pred, eps=1e-8):
    """
    Computes the Mean Absolute Percentage Error (MAPE).

    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): Ground truth values.
        eps (float): A small constant to prevent division by zero.

    Returns:
        torch.Tensor: The MAPE value in percentage.
    """
    # Avoid division by zero by adding a small epsilon
    return torch.abs((real - pred) / (real + eps)).mean(dim=1)

def mae_loss(real, pred):
    """Compute the Mean Absolute Error between output and target tensors."""
    pred = torch.maximum(pred, torch.zeros_like(pred))
    return torch.abs(pred - real).mean(dim=1)

def rmse_loss(real, pred):
    """Compute the Mean Absolute Error between output and target tensors."""
    pred = torch.maximum(pred, torch.zeros_like(pred))
    return torch.sqrt(((pred - real)**2).mean(dim=1))