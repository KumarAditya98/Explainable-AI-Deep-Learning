#%%

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import random_split
from torch.utils.data import TensorDataset, DataLoader
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
import streamlit as st
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.chdir("/home/ubuntu/attempt2")

torch.manual_seed(42)

st.header("XAI LSTM")

if "phase1" not in st.session_state:
    st.session_state["phase1"] = False

with st.form("my_form"):
    
    st.write("Ticker Symbol:")
    TICKER = st.text_input(label=" ", key = "ticker")
    START = st.date_input("Start Date", datetime.date(2022, 1, 1), key= "startdate")
    END = st.date_input("End Date", datetime.date(2023, 1, 1), key= "enddate")
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
        
    stock_data = yf.download(TICKER, start=START, end=END)

    # Extract closing prices
    price_data = stock_data["Close"].values.reshape(-1, 1)
    price_data = torch.tensor(price_data, dtype=torch.float32)

    # Create input sequences and corresponding targets (e.g., predicting the next day's closing price)
    seq_length = 20
    X, y = [], []

    for i in range(len(price_data) - seq_length):
        X.append(price_data[i:i+seq_length])
        y.append(price_data[i+seq_length])

    X = torch.stack(X)
    y = torch.stack(y)

    # Split the data into training and validation sets
    train_size = int(0.8 * len(X))
    val_size = len(X) - train_size
    train_dataset, val_dataset = random_split(TensorDataset(X, y), [train_size, val_size])

    # DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

    # Define the LSTM model
    class MyLSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MyLSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])  # Assuming you want to predict the last time step
            return output

        def get_forget_gate_output(self, x):
            lstm_out, _ = self.lstm(x)
            forget_gate_output = torch.sigmoid(lstm_out[:, -1, 2 * self.lstm.hidden_size // 4:3 * self.lstm.hidden_size // 4])  # Extracting forget gate output
            return forget_gate_output

    # Initialize the model, loss function, and optimizer
    model = MyLSTMModel(input_size=1, hidden_size=HLS, output_size=1)  # input_size=1 because we're using the closing price
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    forget_cell = []
    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

                # Get forget gate output
                forget_gate_output = model.get_forget_gate_output(inputs)

        val_loss /= len(val_loader)
        st.write(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

        # Extract forget gate output for the last batch in the validation set
        forget_cell.append(forget_gate_output)

    act = np.array([x[0].detach().numpy().reshape(-1,1) for x in forget_cell])

    
    fig, ax = plt.subplots(figsize=(8,8))
    plt.axis("off")
    im = ax.imshow(act[0], animated =True, cmap='hot', interpolation='bicubic')
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '100%', '700%')
    fig.colorbar(im, cax)

    def update(frame):
        ax.imshow(act[frame], animated =True, cmap='hot', interpolation='bicubic')
        ax.set_title(f'Forget Cell LSTM: Epoch {frame + 1}')
        return ax,

    # Create the animation
    animation = FuncAnimation(fig, 
                            update, 
                            repeat = True, 
                            frames=len(act))

    writer = PillowWriter(fps = 10)

    animation.save('ACT_LSTM.gif', writer=writer)
    
    st.image('ACT_LSTM.gif')

# # %%
