import torch
import matplotlib.pyplot as plt

# Define a function that computes the loss (difference between attn_ratio and target)
def compute_loss(W, lambda_reg):
    # Use input_q and input_k as ones for simplicity
    input_q = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')
    input_k = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')

    W_q_matrix = torch.stack([torch.block_diag(*torch.chunk(W[i], head_dim // group_size, dim=0)).float() for i in range(seq_length)])
    
    # W_k_matrix = torch.stack([torch.block_diag(*torch.chunk(W[i], head_dim // group_size, dim=0)).float() for i in range(seq_length)])
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

def plot_attn(W):
    input_q = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')
    input_k = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')
    

    W_q_matrix = torch.stack([torch.block_diag(*torch.chunk(W[i], head_dim // group_size, dim=0)).float() for i in range(seq_length)])
    
    # W_k_matrix = torch.stack([torch.block_diag(*torch.chunk(W_k[i], head_dim // group_size, dim=0)).float() for i in range(seq_length)])
    W_k_matrix = W_q_matrix.transpose(1, 2)
    
    # compute the biggest signular value of W_q
    print(torch.svd(W_q_matrix[0])[1][0])
    print(torch.svd(W_q_matrix[0])[1][0])

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
    
def initizalize_params(method = None):

    W = torch.randn(seq_length, head_dim, group_size, device='cuda').detach().requires_grad_(True) * 0.1
    
    return W

def orthogonality_regularization(W):
    I = torch.eye(W.size(-1), device='cuda')  # Identity matrix
    WTW = W.transpose(-2, -1) @ W  # Compute W^T W
    ortho_loss = torch.norm(WTW - I, p='fro')  # Frobenius norm of the difference
    return ortho_loss

# Load target and set other parameters
# target = torch.load('rope/attn_t.pt').to('cuda')
target = torch.load('rope/correlation_maps_y.pth').to('cuda')
target = target / torch.max(target)

batch_size = 1
head_num = 16
head_dim = 64
seq_length = 16
group_size = 2

# Initialize the parameters for optimization as leaf tensors
W = initizalize_params('cossin')

W = torch.nn.Parameter(W)

# Set up the Adam optimizer
optimizer = torch.optim.Adam([W], lr=1)

# Run the optimization loop
num_iterations = 1000
for i in range(num_iterations):
    optimizer.zero_grad()
    loss = compute_loss(W, lambda_reg=1e-6)
    loss.backward()
    optimizer.step()
    if (i + 1) % 5 == 0:
        print(f"Iteration {i + 1}, Loss: {loss.item()}")
    if (i + 1) % 20 == 0:
# Plot the attention maps after optimization
        plot_attn(W.detach())