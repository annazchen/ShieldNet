import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

start_time = time.time()


# Example: Load your dataset
data = pd.read_csv('Monday.csv')  # replace with your dataset path

# Print the number of rows in the original dataset
print("Original number of rows:", len(data))

# Remove leading/trailing spaces in column names
#data.columns = data.columns.str.strip()

# Specify the columns you want to keep
columns_to_keep = [' Destination Port', ' Flow Duration', ' Total Fwd Packets', 
                   ' Total Backward Packets', ' Label']  # Add or remove as needed

# Filter the DataFrame to only keep the specified columns
filtered_data_columns = data[columns_to_keep]

# Sample 10% of the rows randomly
filtered_data_rows = filtered_data_columns.sample(frac=0.001, random_state=1)

# Print the number of rows in the sampled data and the first few rows for verification
print("Sampled number of rows:", len(filtered_data_rows))
print("Sampled random 10% of rows:")
print(filtered_data_rows.head()) 


# Preprocess the data (example: scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(filtered_data_rows.drop(' Label', axis=1))  # Scale only the features
print(filtered_data_rows.head())

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10  # Set your sequence length
X, y = create_sequences(data_scaled, seq_length)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Optional: print shapes of the tensors to verify
print(f"X_tensor shape: {X_tensor.shape}")
print(f"y_tensor shape: {y_tensor.shape}")


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out


# Hyperparameters
input_size = 4  # Number of features in your dataset
hidden_size = 50  # Number of hidden units
output_size = 1  # Predicting one value
num_epochs = 100
learning_rate = 0.001

# Model, loss function, optimizer
model = SimpleRNN(input_size, hidden_size, output_size).to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#convert tensors
X_tensor = torch.FloatTensor(X).to('cuda')
Y_tensor = torch.FloatTensor(y).to('cuda')

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_tensor)  # Add input dimension
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:  # Print every 10 epochs
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Example of visualizing predictions
model.eval()
with torch.no_grad():
    train_predict = model(X_tensor)
    train_predict = train_predict.detach().cpu().numpy()

# Inverse transform to get original values
train_predict = scaler.inverse_transform(train_predict)
y_actual = scaler.inverse_transform(y_tensor.detach().numpy().reshape(-1, 1))

# Plotting
plt.plot(y_actual, label='Actual')
plt.plot(train_predict, label='Predicted')
plt.legend()
plt.show()


torch.save(model.state_dict(), 'rnn_model.pth')


end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")
