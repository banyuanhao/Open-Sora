import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

# Define a function that computes the loss (difference between attn_ratio and target)
def compute_loss(W, lambda_reg):
    # Use input_q and input_k as ones for simplicity
    input_q = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')
    input_k = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')

    W_q_matrix = torch.stack([torch.block_diag(*torch.chunk(W[i], head_dim // group_size, dim=0)).float() for i in range(seq_length)])
    
    W_k_matrix = W_q_matrix.transpose(1, 2)

    rotated_q = input_q @ W_q_matrix
    rotated_k = input_k @ W_k_matrix

    rotated_q = rotated_q.transpose(1, 2)
    rotated_k = rotated_k.transpose(1, 2)
    attn = rotated_q @ rotated_k.transpose(-2, -1)

    attn_ratio = torch.mean(attn, dim=1).squeeze(0)
    attn_ratio = attn_ratio / torch.max(attn_ratio)

    # Compute the original loss
    main_loss = torch.nn.functional.mse_loss(attn_ratio, target)
    
    # Compute the orthogonality regularization loss
    ortho_loss = orthogonality_regularization(W_q_matrix)
    
    # Combine the main loss and the regularization term
    loss = main_loss + lambda_reg * ortho_loss
    return loss

def orthogonality_regularization(W):
    I = torch.eye(W.size(-1), device='cuda')  # Identity matrix
    WTW = W.transpose(-2, -1) @ W  # Compute W^T W
    ortho_loss = torch.norm(WTW - I, p='fro')  # Frobenius norm of the difference
    return ortho_loss

# Wrapper function for scipy
def scipy_objective(flat_W_numpy):
    # Convert the flat NumPy array back to a PyTorch tensor
    W_tensor = torch.tensor(flat_W_numpy, dtype=torch.float64, device='cuda').view(seq_length, head_dim, group_size)
    W_tensor.requires_grad = True

    # Compute the loss
    loss = compute_loss(W_tensor, lambda_reg)
    
    # Compute gradients
    loss.backward()
    gradients = W_tensor.grad.cpu().numpy().astype(np.float64).flatten()  # Ensure float64 for scipy compatibility
    
    return loss.item(), gradients

# Plot function remains unchanged
def plot_attn(W):
    input_q = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')
    input_k = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')
    
    W_q_matrix = torch.stack([torch.block_diag(*torch.chunk(W[i], head_dim // group_size, dim=0)).float() for i in range(seq_length)])
    
    W_k_matrix = W_q_matrix.transpose(1, 2)
    
    rotated_q = input_q @ W_q_matrix
    rotated_k = input_k @ W_k_matrix

    rotated_q = rotated_q.transpose(1, 2)
    rotated_k = rotated_k.transpose(1, 2)
    attn = rotated_q @ rotated_k.transpose(-2, -1)

    attn_ratio = torch.mean(attn, dim=1).squeeze(0)
    attn_ratio = attn_ratio / torch.max(attn_ratio)

    plt.figure(figsize=(10, 10))
    im = plt.imshow(attn_ratio.cpu(), cmap='jet', aspect='auto')
    cbar = plt.colorbar()
    im.set_clim(0, 1) 
    plt.savefig('average_t.png')
    plt.close()
    
    plt.figure(figsize=(10, 10))
    im = plt.imshow(target.cpu(), cmap='jet', aspect='auto')
    cbar = plt.colorbar()
    im.set_clim(0, 1) 
    plt.savefig('average_t_target.png')
    plt.close()
    
def initizalize_params():
    W = torch.randn(seq_length, head_dim, group_size, device='cuda') * 0.1
    return W

# Load target and set other parameters
target = torch.load('rope/correlation_maps_t.pth').to('cuda')
target = target / torch.max(target)

# Parameters
batch_size = 1
head_num = 16
head_dim = 64
seq_length = 24
group_size = 2
lambda_reg = 1e-6

# Initialize the parameters
W = initizalize_params()

# Convert W to a flat NumPy array for scipy.optimize with float64
flat_W_numpy = W.cpu().detach().numpy().astype(np.float64).flatten()

# Run scipy.optimize.minimize with BFGS method
result = minimize(
    fun=lambda w: scipy_objective(w)[0],  # Loss function only for function call
    x0=flat_W_numpy,
    jac=lambda w: scipy_objective(w)[1],  # Gradient function
    method='L-BFGS-B',  # Using L-BFGS-B for box constraints (if needed)
    options={'maxiter': 1000, 'disp': True}
)

# Convert the result back to PyTorch tensor for further analysis or plotting
final_W = torch.tensor(result.x, dtype=torch.float32, device='cuda').view(seq_length, head_dim, group_size)
plot_attn(final_W)
