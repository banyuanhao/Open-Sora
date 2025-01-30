import torch
import matplotlib.pyplot as plt

# Define a function that computes the loss (difference between attn_ratio and target)
def compute_loss(W_q, W_k):
    # Use input_q and input_k as ones for simplicity
    input_q = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')
    input_k = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')

    W_q_matrix = torch.stack([torch.block_diag(*torch.chunk(W_q[i], head_dim // group_size, dim=0)).float() for i in range(seq_length)])
    
    W_k_matrix = torch.stack([torch.block_diag(*torch.chunk(W_k[i], head_dim // group_size, dim=0)).float() for i in range(seq_length)])

    rotated_q = input_q @ W_q_matrix
    rotated_k = input_k @ W_k_matrix

    rotated_q = rotated_q.transpose(1, 2)
    rotated_k = rotated_k.transpose(1, 2)
    attn = rotated_q @ rotated_k.transpose(-2, -1)

    attn_ratio = torch.mean(attn, dim=1).squeeze(0)
    attn_ratio = attn_ratio / torch.max(attn_ratio)

    loss = torch.nn.functional.mse_loss(attn_ratio, target)
    return loss

def plot_attn(W_q, W_k):
    input_q = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')
    input_k = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')

    W_q_matrix = torch.stack([torch.block_diag(*torch.chunk(W_q[i], head_dim // group_size, dim=0)).float() for i in range(seq_length)])
    
    W_k_matrix = torch.stack([torch.block_diag(*torch.chunk(W_k[i], head_dim // group_size, dim=0)).float() for i in range(seq_length)])
    
    # # compute the biggest signular value of W_q
    # print(torch.svd(W_q_matrix[0])[1])
    # print(W_q_matrix[0].shape)
    # print(torch.svd(W_q_matrix[0])[1])

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
    
    plt.figure(figsize=(10, 10))
    im = plt.imshow(target.cpu(), cmap='jet', aspect='auto')
    cbar = plt.colorbar()
    im.set_clim(0, 1) 
    plt.savefig('average_t_target.png')
    
def initizalize_params(method = None):

    W_q = torch.randn(seq_length, head_dim, group_size, device='cuda').detach().requires_grad_(True) * 0.1
    W_k = torch.randn(seq_length, head_dim, group_size, device='cuda').detach().requires_grad_(True) * 0.1
    
    return W_q, W_k

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
W_q, W_k = initizalize_params('cossin')

W_q = torch.nn.Parameter(W_q)
W_k = torch.nn.Parameter(W_k)

# Set up the Adam optimizer
optimizer = torch.optim.Adam([W_q, W_k], lr=1)

# Run the optimization loop
num_iterations = 10000
for i in range(num_iterations):
    # print(W_q)
    optimizer.zero_grad()
    loss = compute_loss(W_q, W_k)
    loss.backward()
    # print(W_q.grad)
    optimizer.step()
    if (i + 1) % 100 == 0:
        print(f"Iteration {i + 1}, Loss: {loss.item()}")
    if (i + 1) % 1000 == 0:
# Plot the attention maps after optimization
        plot_attn(W_q.detach(), W_k.detach())   

