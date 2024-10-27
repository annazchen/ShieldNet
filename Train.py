import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import joblib
from keras import models

start_time = time.time()

# Load dataset
data = pd.read_csv('Tuesday.csv')  # Replace with your dataset path
print("Original number of rows:", len(data))

# Device Set
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("yes" if torch.cuda.is_available() else "no")


# Specify columns to keep
columns_to_keep = [
    ' Destination Port', 
    ' Flow Duration', 
    ' Total Fwd Packets', 
    ' Total Backward Packets', 
    'Total Length of Fwd Packets', 
    ' Total Length of Bwd Packets', 
    ' Flow IAT Mean', 
    ' Fwd Packet Length Mean', 
    ' Bwd Packet Length Mean', 
    'Fwd PSH Flags', 
    ' Bwd PSH Flags', 
    'FIN Flag Count', 
    ' SYN Flag Count', 
    ' ACK Flag Count', 
    ' Down/Up Ratio', 
    'Active Mean', 
    'Idle Mean', 
    'Fwd Avg Bytes/Bulk', 
    ' Bwd Avg Bytes/Bulk',
    ' Label'
]



# Filter data and sample rows
filtered_data = data[columns_to_keep].sample(frac=.1, random_state=1)
print("Sampled number of rows:", len(filtered_data))

# Convert 'Label' to binary (0 = BENIGN, 1 = ATTACK)
filtered_data[' Label'] = filtered_data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(filtered_data.drop(' Label', axis=1))
labels = filtered_data[' Label'].values

# Create sequences
def create_sequences(data, labels, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = labels[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(data_scaled, labels, seq_length)

# Convert to tensors
X_tensor = torch.FloatTensor(X).to(device)
y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)  # Reshape for binary output

# Define the RNN model
class SimpleRNNBinary(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleRNNBinary, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output size is 1 for binary classification

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step output
        return torch.sigmoid(out)  # Sigmoid activation for binary classification

# Hyperparameters
input_size = X.shape[2]  # Number of features
hidden_size = 20
num_epochs = 200
learning_rate = 0.001

# Model, loss function, and optimizer
model = SimpleRNNBinary(input_size, hidden_size).to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
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
    
    torch.cuda.empty_cache()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    


# Model evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_tensor)
    predictions = (predictions > 0.5).float()  # Threshold at 0.5 for binary decision

# Calculate accuracy
correct = (predictions == y_tensor).sum().item()
accuracy = correct*100 / y_tensor.size(0)
print(f'Accuracy: {accuracy:.2f}%')

# Save the model
torch.save(model.state_dict(), 'rnn_model.pth')

# Save scalar as pickle file (yum)
joblib.dump(scaler, 'LSTM_tue.pkl')  
print("Scaler saved to LSTM_tue.pkl")

# Add model.eval() after training is complete, before saving
model.eval()
torch.save(model.state_dict(), 'rnn_model.pth')

# In the analyze_input_file function, load the model and set it to evaluation mode
def analyze_input_file(file_path):
    # Load the trained model weights
    model = SimpleRNNBinary(input_size, hidden_size).to(device)  # Initialize the model architecture
    model.load_state_dict(torch.load('rnn_model.pth'))  # Load the trained model weights
    model.eval()  # Set model to evaluation mode
    
    # Load the scaler
    scaler = joblib.load('LSTM_tue.pkl')  # Load the saved scaler
    
    # Load and process data
    data = pd.read_csv(file_path)  # Load the new file
    columns_to_keep = [
        ' Destination Port', 
    ' Flow Duration', 
    ' Total Fwd Packets', 
    ' Total Backward Packets', 
    'Total Length of Fwd Packets', 
    ' Total Length of Bwd Packets', 
    ' Flow IAT Mean', 
    ' Fwd Packet Length Mean', 
    ' Bwd Packet Length Mean', 
    'Fwd PSH Flags', 
    ' Bwd PSH Flags', 
    'FIN Flag Count', 
    ' SYN Flag Count', 
    ' ACK Flag Count', 
    ' Down/Up Ratio', 
    'Active Mean', 
    'Idle Mean', 
    'Fwd Avg Bytes/Bulk', 
    ' Bwd Avg Bytes/Bulk'
    ]
    filtered_data = data[columns_to_keep]
    
    # Scale the features using the previously saved scaler
    data_scaled = scaler.transform(filtered_data)  # Use .transform() instead of .fit_transform()
    
    # Convert 'Label' to binary (0 = BENIGN, 1 = ATTACK) for ground truth comparison
    true_labels = (data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)).values[:len(data) - seq_length]
    
    # Create sequences for the new data
    def create_sequences(data, seq_length):
        xs = []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            xs.append(x)
        return np.array(xs)
    
    seqlength = 10  # Match sequence length
    X = create_sequences(data_scaled, seqlength)
    X_tensor = torch.FloatTensor(X).to(device)  # Move to the same device as the model (CPU or GPU)

    # Predict with the loaded model
    with torch.no_grad():
        predictions = model(X_tensor)
        predictions_np = predictions.cpu().numpy()  # Move predictions back to CPU for further processing
    
    # Convert probabilities to binary classification (0 or 1)
    binary_predictions = (predictions_np[:, 0] > 0.5).astype(int)
    
    # Calculate accuracy against true labels
    accuracy = accuracy_score(true_labels, binary_predictions)
    print(f"Prediction Accuracy: {accuracy * 100:.2f}%")
    
    return binary_predictions


# Example usage
file_path = 'Wednesday.csv'
result = analyze_input_file(file_path)


end_time = time.time()
