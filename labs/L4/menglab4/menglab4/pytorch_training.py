def set_pytorch_seed(seed):
    # Set the seed
  torch.manual_seed(seed)

  # If you're using CUDA (GPU)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)  # For multi-GPU setups


# Helper function
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_model_predictions_pytorch(model, X_train, y_train, X_test=None, y_test=None, n_points=400):
    """
    Plot a trained PyTorch model's predictions against the data.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model (e.g., nn.Sequential).
    X_train, y_train : torch.Tensor
        Training inputs and targets. Can be 1D or 2D, on CPU or GPU.
    X_test, y_test : torch.Tensor, optional
        Optional test inputs and targets to visualize.
    n_points : int
        Number of evenly spaced points for the smooth prediction curve.
    """

    def to_numpy(t):
        """Detach, move to CPU, and convert to NumPy."""
        if torch.is_tensor(t):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def ensure_2d_col(t):
        """Ensure shape is (N, 1) for nn.Linear layers."""
        if t.ndim == 1:
            return t.unsqueeze(1)
        return t

    # Move to NumPy for plotting
    X_train_np = to_numpy(X_train).reshape(-1)
    y_train_np = to_numpy(y_train).reshape(-1)
    if X_test is not None and y_test is not None:
        X_test_np = to_numpy(X_test).reshape(-1)
        y_test_np = to_numpy(y_test).reshape(-1)
    else:
        X_test_np = y_test_np = None

    # Smooth x-grid for model prediction
    X_all = X_train_np if X_test_np is None else np.concatenate([X_train_np, X_test_np])
    xs_np = np.linspace(X_all.min(), X_all.max(), n_points)

    # Detect model device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    xs_t = torch.from_numpy(xs_np.astype(np.float32)).to(device)
    xs_t = ensure_2d_col(xs_t)

    # Model prediction (no gradient tracking)
    with torch.no_grad():
        ys_t = model(xs_t)

    ys_np = to_numpy(ys_t).reshape(-1)

    # --- Plot ---
    plt.figure(figsize=(7, 5))
    plt.scatter(X_train_np, y_train_np, alpha=0.6, label='Train data')
    if X_test_np is not None:
        plt.scatter(X_test_np, y_test_np, alpha=0.6, label='Test data')
    plt.plot(xs_np, ys_np, 'r-', linewidth=2, label='Model prediction')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Model Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Helper function (execute to define helper plotting function)
def plot_pytorch_training(model, X_train, X_test, y_train, y_test, losses):  
    """
    Visualize PyTorch model training results.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model
    X_train, X_test, y_train, y_test : torch.Tensor
        Training and test data as PyTorch tensors
    losses : list of float
        Training loss history
    """
    # Get the device the model is on
    device = next(model.parameters()).device
    
    # Convert tensors to numpy for plotting
    X_train_np = X_train.detach().cpu().numpy()
    X_test_np = X_test.detach().cpu().numpy()
    y_train_np = y_train.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: model fit on both splits
    ax1.scatter(X_train_np, y_train_np, alpha=0.6, label='Train data')
    ax1.scatter(X_test_np, y_test_np, alpha=0.6, label='Test data')

    # Smooth prediction curve across the full range
    X_all_np = np.concatenate([X_train_np, X_test_np])
    xs = np.linspace(X_all_np.min(), X_all_np.max(), 400)
    
    # Convert to tensor on the same device as model, predict, convert back to numpy
    model.eval()
    with torch.no_grad():
        xs_tensor = torch.FloatTensor(xs).reshape(-1, 1).to(device)  # Move to same device!
        ys_tensor = model(xs_tensor)
        ys = ys_tensor.cpu().numpy()
    
    ax1.plot(xs, ys, 'r-', linewidth=2, label='Model prediction')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.set_title('Model Fit (Train/Test)')

    # Right: training loss curve
    ax2.plot(losses)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.show()
