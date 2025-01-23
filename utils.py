import numpy as np
from itertools import islice
import torch
from datetime import datetime
import torch.nn as nn


loss_function_dict = {
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss
}

def get_cur_time():
    return datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def get_float_wn(parameters):
    """
    :param parameters:
        Example: parameters = model_clf.linear_1.parameters()
    :return: float
    """
    with torch.no_grad():
        out = sum(torch.pow(p, 2).sum() for p in parameters)
        out = float(np.sqrt(out.item()))
    return out


# def compute_accuracy(network, dataset, device, N=2000, batch_size=50):
#     """Computes accuracy of `network` on `dataset`."""
#     with torch.no_grad():
#         N = min(len(dataset), N)
#         batch_size = min(batch_size, N)
#         dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         correct = 0
#         total_acc = 0
#         for x, labels in islice(dataset_loader, N // batch_size):
#             logits = network(x.to(device))
#             predicted_labels = torch.argmax(logits, dim=1)
#             correct += torch.sum(predicted_labels == labels.to(device))
#             total_acc += x.size(0)
#         return (correct / total_acc).item()

@torch.no_grad()
def compute_loss_accuracy(network, dataset, loss_function, device, N=2000, batch_size=128):
    """Computes mean loss and accuracy of `network` on `dataset`."""
    N = min(len(dataset), N)
    batch_size = min(batch_size, N)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = loss_function_dict[loss_function](reduction='sum')
    one_hots = torch.eye(10, 10).to(device)
    total = 0
    points = 0
    # accuracy
    correct = 0
    total_acc = 0
    for x, labels in islice(dataset_loader, N // batch_size):
        y = network(x.to(device))
        if loss_function == 'CrossEntropy':
            total += loss_fn(y, labels.to(device)).item()
        elif loss_function == 'MSE':
            total += loss_fn(y, one_hots[labels]).item()
        points += len(labels)
        # accuracy
        predicted_labels = torch.argmax(y, dim=1)
        correct += torch.sum(predicted_labels == labels.to(device))
        total_acc += x.size(0)  
    return total / points, (correct / total_acc).item()

def log_gradients_in_model_tb(model, logger, step):
    with torch.no_grad():
        for tag, value in model.named_parameters():
            if value.grad is not None:
                logger.add_scalar(f"grad_mean/{tag.split('.')[1]}/{tag.split('.')[0]}",
                                  torch.mean(value.grad.cpu()), step)
                logger.add_scalar(f"grad_var/{tag.split('.')[1]}/{tag.split('.')[0]}",
                                  torch.var(value.grad.cpu()), step)

def log_gradients_in_model_wandb(model, run, step):
    with torch.no_grad():
        for tag, value in model.named_parameters():
            if value.grad is not None:
                run.log({f"grad_mean_{tag.split('.')[1]}/{tag.split('.')[0]}": torch.mean(value.grad.cpu())},
                        step=step)
                run.log({f"grad_var_{tag.split('.')[1]}/{tag.split('.')[0]}": torch.var(value.grad.cpu())},
                        step=step)


# For Sliced MI     
def sample_spherical(n_projections, dim):
    sampled_vectors = np.array([]).reshape(0,dim)
    while len(sampled_vectors) < n_projections:
        vec = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), size=dim) # (num_vec, dim)
        vec = np.linalg.qr(vec).Q
        sampled_vectors = np.vstack((sampled_vectors, vec))
    return sampled_vectors[:n_projections] # (num_vec, dim)
    
class smi_compressor():
    def __init__(self, dim, n_projections):
        self.theta = sample_spherical(n_projections=n_projections, dim=dim) # (n_projections, dim)
        
    def __call__(self, X):
        # getting projections
        X_compressed = np.dot(self.theta, X.T)
        return X_compressed # m x n

def measure_smi_projection(mi_estimator, x, y):
    mi_estimator.fit(x, y, verbose=0)
    return mi_estimator.estimate(x, y, verbose=0)