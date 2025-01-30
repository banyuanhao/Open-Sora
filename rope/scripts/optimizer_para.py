import torch
from torch.autograd.functional import hessian, jacobian
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define a function that computes the loss (difference between attn_ratio and target)
def compute_loss(params):
    
    # input_q = torch.randn(batch_size, seq_length, head_num, head_dim, device='cuda')
    # input_k = torch.randn(batch_size, seq_length, head_num, head_dim, device='cuda')
    input_q = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')
    input_k = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')

    W_q_data = params[:W_q.numel()].reshape(W_q.shape).to('cuda')
    W_k_data = params[W_q.numel():].reshape(W_k.shape).to('cuda')
    
    W_q_matrix = torch.block_diag(*torch.chunk(W_q_data, head_dim // group_size, dim=0)).float()
    W_k_matrix = torch.block_diag(*torch.chunk(W_k_data, head_dim // group_size, dim=0)).float()

    rotated_q = input_q @ W_q_matrix
    rotated_k = input_k @ W_k_matrix

    rotated_q = rotated_q.transpose(1, 2)
    rotated_k = rotated_k.transpose(1, 2)
    attn_new = rotated_q @ rotated_k.transpose(-2, -1)
    
    input_q = input_q.transpose(1, 2)
    input_k = input_k.transpose(1, 2)
    attn_ori = input_q @ input_k.transpose(-2, -1)


    attn_ratio = attn_new / attn_ori
    attn_ratio = torch.mean(attn_ratio, dim=1).squeeze(0)
    attn_ratio = attn_ratio / torch.max(attn_ratio)

    loss = torch.nn.functional.mse_loss(attn_ratio, target)
    print("Loss:", loss.item())
    return loss

def plot_attn(params):
    
    if not torch.is_tensor(params):
        params = torch.tensor(params, device='cuda')
    
    # input_q = torch.randn(batch_size, seq_length, head_num, head_dim, device='cuda')
    # input_k = torch.randn(batch_size, seq_length, head_num, head_dim, device='cuda')
    input_q = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')
    input_k = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')

    W_q_data = params[:W_q.numel()].reshape(W_q.shape).to('cuda')
    W_k_data = params[W_q.numel():].reshape(W_k.shape).to('cuda')
    
    W_q_matrix = torch.block_diag(*torch.chunk(W_q_data, head_dim // group_size, dim=0)).float()
    W_k_matrix = torch.block_diag(*torch.chunk(W_k_data, head_dim // group_size, dim=0)).float()

    rotated_q = input_q @ W_q_matrix
    rotated_k = input_k @ W_k_matrix

    rotated_q = rotated_q.transpose(1, 2)
    rotated_k = rotated_k.transpose(1, 2)
    attn_new = rotated_q @ rotated_k.transpose(-2, -1)
    
    input_q = input_q.transpose(1, 2)
    input_k = input_k.transpose(1, 2)
    attn_ori = input_q @ input_k.transpose(-2, -1)


    attn_ratio = attn_new / attn_ori
    attn_ratio = torch.mean(attn_ratio, dim=1).squeeze(0)
    attn_ratio = attn_ratio / torch.max(attn_ratio)

    plt.figure(figsize=(10, 10))
    im = plt.imshow(attn_ratio.cpu(), cmap='jet', aspect='auto')
    cbar = plt.colorbar()
    im.set_clim(0, 1) 
    plt.savefig('average_t.png')
    
    plt.figure(figsize=(10, 10))
    im = plt.imshow(target.cpu(), cmap='jet', aspect='auto')
    cbar = plt.colorbar()
    im.set_clim(0, 1) 
    plt.savefig('average_t_target.png')

# Define functions to compute the gradient (Jacobian) and Hessian matrix for scipy optimization
def loss_and_grad(params):
    params = torch.tensor(params, requires_grad=True, device='cuda')
    loss_value = compute_loss(params)
    loss_value.backward()
    return loss_value.item(), params.grad.cpu().numpy()

def hessian_matrix(params):
    params_tensor = torch.tensor(params, requires_grad=True, device='cuda')
    hess = hessian(compute_loss, params_tensor)
    return hess.cpu().numpy()


# target = torch.load('rope/correlation_maps_t.pth').to('cuda')
target = torch.load('rope/attn_t.pt').to('cuda')
target = target/torch.max(target)

batch_size = 1
head_num = 16
head_dim = 64
seq_length = 24

# Initialize the parameters for optimization
group_size = 4
W_q = torch.randn(head_dim,group_size, device='cuda') * 0.01
W_k = torch.randn(head_dim,group_size, device='cuda') * 0.01

initial_params = torch.cat([W_q.flatten(), W_k.flatten()]).cpu().numpy()
    
# Run the optimization using scipy's minimize with BFGS for second-order optimization
result = minimize(loss_and_grad, initial_params, method='L-BFGS-B', jac=True, hess=hessian_matrix,options={'maxfun': 1000, 'ftol': 1e-8})

# Extract the optimized parameters
optimized_params = result.x
optimized_W_q = torch.tensor(optimized_params[:W_q.numel()]).reshape(W_q.shape).to('cuda')
optimized_W_k = torch.tensor(optimized_params[W_q.numel():]).reshape(W_k.shape).to('cuda')

print("Optimization successful:", result.success)
# print("Optimized W_q:", optimized_W_q)
# print("Optimized W_k:", optimized_W_k)
plot_attn(optimized_params)
