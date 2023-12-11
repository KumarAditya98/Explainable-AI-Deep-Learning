#%%

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import streamlit as st
from mpl_toolkits.axes_grid1 import make_axes_locatable

st.header("XAI MLP")

os.chdir("/home/ubuntu/attempt2")
torch.manual_seed(42)

if "phase1" not in st.session_state:
    st.session_state["phase1"] = False
    
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        act1 = []
        act2 = []
        x = self.fc1(x)
        act1.append(x)
        x = self.relu(x)
        x = self.fc2(x)
        act2.append(x)
        return x, act1, act2

with st.form("my_form"):
    
    st.write("Batch Size:")
    BATCH = st.text_input(label=" ", key = "batch")
    st.write("Hidden Layer Size:")
    HLS = st.text_input(label=" ", key = "hls")
    st.write("Learning Rate:")
    LR = st.text_input(label=" ", key = "lr")
    st.write("Epochs:")
    EPOCHS = st.text_input(label=" ", key = "epochs")
    
    submit = st.form_submit_button("Submit")

if submit == False:
    st.stop()

BATCH = int(BATCH)
HLS = int(HLS)
LR = float(LR)
EPOCHS = int(EPOCHS)

# Define hyperparameters
input_size = 28 * 28  # MNIST image size
# Number of neurons in the hidden layer
output_size = 10  # Number of classes in MNIST

# Instantiate the model, loss function, and optimizer
model = MLPClassifier(input_size, HLS, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Download and prepare the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH, shuffle=False)

# Training loop
act1_ = []
act2_ = []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.view(inputs.size(0), -1)  # Flatten the input images
        optimizer.zero_grad()
        outputs, act1, act2 = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    act1_.append(act1)
    act2_.append(act2)

    st.write(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader)}')

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.view(inputs.size(0), -1)
        outputs, act1, act2 = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
st.write(f'Test Accuracy: {accuracy * 100:.2f}%')

act1 = [x[0].detach().numpy() for x in act1_]
act2 = [x[0].detach().numpy() for x in act2_]

act1 = np.array(act1)
act2 = np.array(act2)

fig, ax = plt.subplots(figsize=(8,8))
plt.axis("off")

def update(frame):
    ax.imshow(act1[frame], animated =True, cmap='hot', interpolation='bicubic')
    ax.set_title(f'MLP Layer 1 ({input_size}, {HLS}): Epoch {frame + 1}')
    return ax,

# Create the animation
animation = FuncAnimation(fig, 
                        update, 
                        repeat = True, 
                        frames=len(act1))

writer = PillowWriter(fps = 2)

animation.save('ACT1.gif', writer=writer)

st.image("ACT1.gif")

fig, ax = plt.subplots(figsize=(8,8))
plt.axis("off")

def update(frame):
    ax.imshow(act2[frame], animated =True, cmap='hot', interpolation='bicubic')
    ax.set_title(f'MLP Layer 2 ({HLS}, {output_size}): Epoch {frame + 1}')
    return ax,

# Create the animation
animation = FuncAnimation(fig, 
                        update, 
                        repeat = True, 
                        frames=len(act2))

writer = PillowWriter(fps = 2)

animation.save('ACT2.gif', writer=writer)

st.image("ACT2.gif")

