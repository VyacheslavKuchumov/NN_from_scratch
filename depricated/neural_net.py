import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np


class HomerBartDataset(Dataset):
    def __init__(self, csv_file):
        data = np.loadtxt(csv_file, delimiter=',')
        self.features = data[:, :-2]
        self.labels = np.argmax(data[:, -2:], axis=1)  # Convert to class indices

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    

# Hyperparameters
input_size = 64 * 64  # 4096
hidden_size = 512
num_classes = 2
batch_size = 32
lr = 0.001
epochs = 10

# Initialize
dataset = HomerBartDataset('homer_bart_images_pandas.csv')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = Net(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

print("Training complete!")