# training_script.py

import torch
import torch.optim as optim
import numpy as np
from safety_value_network import SafetyValueNetwork

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Example environment state dimension (modify as per env)
STATE_DIM = 4

# Hyperparameters
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 64
GAMMA = 0.99

# Generate toy data (Replace with real data from rollouts)
def generate_data(num_samples=1000):
    states = np.random.uniform(-1, 1, (num_samples, STATE_DIM)).astype(np.float32)
    safety_values = np.random.uniform(0, 1, (num_samples, 1)).astype(np.float32)
    return states, safety_values

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize network
    model = SafetyValueNetwork(STATE_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    # Load data
    states, values = generate_data()
    dataset = torch.utils.data.TensorDataset(torch.tensor(states), torch.tensor(values))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_states, batch_targets in dataloader:
            batch_states = batch_states.to(device)
            batch_targets = batch_targets.to(device)

            preds = model(batch_states)
            loss = loss_fn(preds, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "safety_value_network.pth")
    print("Training complete. Model saved to 'safety_value_network.pth'")

if __name__ == "__main__":
    train()
