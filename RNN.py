import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

start_time = time.time()

# Device Set
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load dataset
data = pd.read_csv('Monday.csv')  # replace with your dataset path
print("Original number of rows:", len(data))

# Specify columns to keep
columns_to_keep = [' Destination Port', ' Flow Duration', ' Total Fwd Packets', 
                   ' Total Backward Packets', ' Label']  # Add/remove columns as needed

# Filter and sample data
filtered_data_columns = data[columns_to_keep]
filtered_data_rows = filtered_data_columns.sample(frac=0.001, random_state=1)
print("Sampled number of rows:", len(filtered_data_rows))
print("Sampled rows preview:")
print(filtered_data_rows.head()) 

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(filtered_data_rows.drop(' Label', axis=1))
print("Scaled data preview:")
print(data_scaled[:5])

# Sequence creation function
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(data_scaled, seq_length)

# Convert to tensors
X_tensor = torch.FloatTensor(X).to(device)
y_tensor = torch.FloatTensor(y).to(device)

print(f"X_tensor shape: {X_tensor.shape}")
print(f"y_tensor shape: {y_tensor.shape}")

# Define RNN model
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
input_size = 4
hidden_size = 50
output_size = 4
num_epochs = 100000
learning_rate = 0.001

# Model, loss function, optimizer
model = SimpleRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions for visualization
model.eval()
with torch.no_grad():
    train_predict = model(X_tensor)
    train_predict = train_predict.detach().cpu().numpy()

# Prepare for inverse scaling
train_predict_expanded = np.zeros((train_predict.shape[0], data_scaled.shape[1]))
train_predict_expanded[:, 0] = train_predict[:, 0]
train_predict_original = scaler.inverse_transform(train_predict_expanded)[:, 0]

# Inverse transform actual y values
y_actual_expanded = np.zeros((y_tensor.shape[0], data_scaled.shape[1]))  # Ensure this matches
y_actual_expanded[:, :4] = y_tensor.detach().cpu().numpy()  # Fill in the correct shape
y_actual = scaler.inverse_transform(y_actual_expanded)[:, 0]  # Adjust based on the desired output

'''
# Inverse transform actual y values
y_actual_expanded = np.zeros((y_tensor.shape[0], data_scaled.shape[1]))
y_actual_expanded[:, 0] = y_tensor.detach().cpu().numpy()
y_actual = scaler.inverse_transform(y_actual_expanded)[:, 0]'''

print(f"X_tensor shape: {X_tensor.shape}")
print(f"y_tensor shape: {y_tensor.shape} \n ------- ")

print(f"X_tensor shape: {y_actual_expanded.shape} \n ------- ")
print(f"y_tensor shape: {y_actual.shape}")


# Plot results
plt.plot(y_actual, label='Actual')
plt.plot(train_predict_original, label='Predicted')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'rnn_model.pth')

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")
