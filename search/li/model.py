from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as nnf
import time
import numpy as np

class Model(nn.Module):
    def __init__(self, input_dim=768, output_dim=1000):
        super().__init__()
        self.layers = torch.nn.Sequential(
          torch.nn.Linear(input_dim, 128),
          torch.nn.ReLU(),
          torch.nn.Linear(128, output_dim)
        )
        self.n_output_neurons = output_dim

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        outputs = self.layers(x)
        return outputs

def get_device() -> torch.device:
    """ Gets the `device` to be used by torch.
    This arugment is needed to operate with the PyTorch model instance.

    Returns
    ------
    torch.device
        Device
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True
    return device


def data_X_to_torch(data) -> torch.FloatTensor:
    """ Creates torch training data."""
    data_X = torch.from_numpy(np.array(data).astype(np.float32))
    return data_X


def data_to_torch(data, labels) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """ Creates torch training data and labels."""
    data_X = data_X_to_torch(data)
    data_y = torch.as_tensor(torch.from_numpy(labels), dtype=torch.long)
    return data_X, data_y


def train(
    data_X: torch.FloatTensor,
    data_y: torch.LongTensor,
    model,
    optimizer,
    device,
    loss,
    epochs=500,
    logger=None
):
    step = epochs // 10
    losses = []
    if logger:
        logger.info(f'Epochs: {epochs}, step: {step}')
    for ep in range(epochs):
        if ep % step == 0 and ep != 0:
            if logger:
                logger.info(f'Epoch {ep} | Loss {curr_loss.item()}')
        pred_y = model(data_X.to(device))
        curr_loss = loss(pred_y, data_y.to(device))
        losses.append(curr_loss.item())

        model.zero_grad()
        curr_loss.backward()

        optimizer.step()
    return losses, model


def predict(model, device, data_X: torch.FloatTensor):
    """ Collects predictions for multiple data points (used in structure building)."""
    model = model.to(device)
    model.eval()

    all_outputs = torch.tensor([], device=device)
    with torch.no_grad():
        outputs = model(data_X.to(device))
        all_outputs = torch.cat((all_outputs, outputs), 0)

    _, y_pred = torch.max(all_outputs, 1)
    return y_pred.cpu().numpy()


def predict_proba(model, device, data_X: torch.FloatTensor):
    """ Collects predictions for a single data point (used in query predictions)."""
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(data_X.to(device))

    print(outputs.shape)
    prob = nnf.softmax(outputs, dim=0)
    return prob.cpu().numpy()
