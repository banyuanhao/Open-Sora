import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define a function that computes the loss (difference between attn_ratio and target)
def compute_loss(W_q, W_k):
    # Use input_q and input_k as ones for simplicity
    input_q = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')
    input_k = torch.ones(batch_size, seq_length, head_num, head_dim, device='cuda')

    W_q_matrix = torch.block_diag(*torch.chunk(W_q, head_dim // group_size, dim=0)).float()
    W_k_matrix = torch.block_diag(*torch.chunk(W_k, head_dim // group_size, dim=0)).float()

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

    W_q_matrix = torch.block_diag(*torch.chunk(W_q, head_dim // group_size, dim=0)).float()
    W_k_matrix = torch.block_diag(*torch.chunk(W_k, head_dim // group_size, dim=0)).float()

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
    if method == 'cossin':
        print("Using cossin")
        theta = torch.tensor(0.1)
        thetas = [theta*i for i in range(head_dim // 2)]
        
        W_q = torch.concatenate([torch.tensor([[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]]) for theta in thetas], dim=0)
        W_k = torch.concatenate([torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]) for theta in thetas], dim=0)
        print(W_q.shape)
        
    else:
        W_q = torch.randn(head_dim, group_size, device='cuda').detach().requires_grad_(True) * 0.1
        W_k = torch.randn(head_dim, group_size, device='cuda').detach().requires_grad_(True) * 0.1
    return W_q, W_k

# Load target and set other parameters
# target = torch.load('rope/attn_t.pt').to('cuda')
target = torch.load('rope/correlation_maps_t.pth').to('cuda')
target = target / torch.max(target)

batch_size = 1
head_num = 16
head_dim = 64
seq_length = 24
group_size = 4

# Initialize the parameters for optimization as leaf tensors
W_q, W_k = initizalize_params('cossin')
print(W_q)
print(W_k)
plot_attn(W_q.detach().to('cuda'), W_k.detach().to('cuda'))   
exit()
W_q = torch.nn.Parameter(W_q)
W_k = torch.nn.Parameter(W_k)

# Set up the Adam optimizer
optimizer = torch.optim.Adam([W_q, W_k], lr=1)

# Run the optimization loop
num_iterations = 10000
for i in tqdm(range(num_iterations)):
    W_q_pre = W_q.clone().detach()
    # print(W_q)
    optimizer.zero_grad()
    loss = compute_loss(W_q, W_k)
    loss.backward()
    # print(W_q.grad)
    optimizer.step()
    W_q_post = W_q.clone().detach()
    if (i + 1) % 100 == 0:
        print(f"Iteration {i + 1}, Loss: {loss.item()}")
        print(torch.norm(W_q_pre - W_q_post))

# Plot the attention maps after optimization
plot_attn(W_q.detach(), W_k.detach())   

