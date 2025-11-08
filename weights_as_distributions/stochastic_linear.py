import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class StochasticLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))

        # Initialize
        nn.init.kaiming_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_log_sigma, -3.0)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_log_sigma, -3.0)

    def forward(self, x, return_weight=False):
        weight_sigma = torch.exp(self.weight_log_sigma)
        epsilon_weight = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_sigma * epsilon_weight

        bias_sigma = torch.exp(self.bias_log_sigma)
        epsilon_bias = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_sigma * epsilon_bias

        output = nn.functional.linear(x, weight, bias)

        if return_weight:
            return output, weight

        return output


class StochasticMLP(nn.Module):
    def __init__(self, input_dim, layer_dim, output_dim):
        super().__init__()

        self.fc1 = StochasticLinear(input_dim, layer_dim)
        self.fc2 = StochasticLinear(layer_dim, layer_dim)
        self.fc3 = StochasticLinear(layer_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, weight_matrix = self.fc1(x, return_weight=True)
        x = self.relu(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        weight = weight_matrix[0, 0].clone()

        return x, weight


# Training loop
model = StochasticMLP(5, 128, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X_train = torch.randn(1000, 5)  # 1000 samples, 5 features
y_train = torch.randn(1000, 1)  # 1000 targets

# Create DataLoader
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        output, weight = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    # Print average loss for the epoch
    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

# input = torch.randn(1, 5)
# weight_distribution = []

# # sanity check
# for i in range(10000):
#     output, weight = model(input)
#     weight_distribution.append(weight.item())  # Convert to Python float
#
# # Convert to numpy
# weight_distribution = np.array(weight_distribution)
#
# # Create the plot
# plt.figure(figsize=(10, 6))
#
# # Histogram
# plt.hist(weight_distribution, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
#
# # Add statistics
# mean_weight = weight_distribution.mean()
# std_weight = weight_distribution.std()
# plt.axvline(mean_weight, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_weight:.3f}')
# plt.axvline(mean_weight + std_weight, color='orange', linestyle=':', linewidth=2,
#             label=f'+1 Std: {mean_weight + std_weight:.3f}')
# plt.axvline(mean_weight - std_weight, color='orange', linestyle=':', linewidth=2,
#             label=f'-1 Std: {mean_weight - std_weight:.3f}')
#
# # Overlay theoretical normal distribution
# from scipy import stats
#
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = stats.norm.pdf(x, mean_weight, std_weight)
# plt.plot(x, p, 'r-', linewidth=2, label=f'Normal fit (μ={mean_weight:.3f}, σ={std_weight:.3f})')
#
# plt.xlabel('Weight Value', fontsize=12)
# plt.ylabel('Density', fontsize=12)
# plt.title('Distribution of Sampled Weight [0, 0] in fc1 over 1000 Forward Passes', fontsize=14)
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()
