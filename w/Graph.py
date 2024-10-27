import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import pickle

start_time = time.time()

# Load dataset
data = pd.read_csv('Tuesday.csv')  # Replace with your dataset path
print("Original number of rows:", len(data))

# Specify columns to keep
columns_to_keep = [
    ' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min', ' Label'

]


'''columns_to_keep = [
    ' Destination Port', ' Flow Duration', ' Total Fwd Packets', 
    ' Total Backward Packets', ' Flow IAT Mean', ' Fwd Packet Length Mean', 
    ' Bwd Packet Length Mean', 'FIN Flag Count', ' SYN Flag Count', 
    ' ACK Flag Count', ' Down/Up Ratio', 'Active Mean', 'Idle Mean', ' Label'
]'''

# Filter data and sample rows
filtered_data = data[columns_to_keep].sample(frac=0.1, random_state=1)
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
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).unsqueeze(1)  # Reshape for binary output

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
input_size = 76  # Number of features
hidden_size = 50
num_epochs = 100
learning_rate = 0.001

# Model, loss function, and optimizer
model = SimpleRNNBinary(input_size, hidden_size).to('cpu')
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
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model using pickle
with open('rnn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

'''# Model evaluation
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

# Add model.eval() after training is complete, before saving
model.eval()
torch.save(model.state_dict(), 'rnn_model.pth')'''

# In the analyze_input_file function, load the model and set it to evaluation mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

def analyze_input_file(file_path):
    # Load the model
    with open('rnn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    model.eval() 

    # Load and process data
    data = pd.read_csv(file_path)
    columns_to_keep = [
        ' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 
        'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', 
        ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 
        'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean', 
        ' Bwd Packet Length Std', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', 
        ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', 
        ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', 
        ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', 
        ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', 
        ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', 
        ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', 
        ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', 
        ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size', 
        ' Avg Bwd Segment Size', ' Fwd Header Length', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', 
        ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 
        'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 
        'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', 
        ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 
        'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'
    ]

    filtered_data = data[columns_to_keep]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(filtered_data)
    
    # Convert 'Label' to binary values for evaluation
    true_labels = (data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)).values

    # Sequence preparation
    seq_length = 10
    def create_sequences(data, seq_length):
        xs = []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            xs.append(x)
        return np.array(xs)
    
    X = create_sequences(data_scaled, seq_length)
    X_tensor = torch.FloatTensor(X)

    # Model prediction
    with torch.no_grad():
        predictions = model(X_tensor)
        predictions_np = predictions.detach().cpu().numpy()
    
    binary_predictions = (predictions_np[:, 0] > 0.5).astype(int)
    accuracy = accuracy_score(true_labels[:len(binary_predictions)], binary_predictions)
    print(f"Prediction Accuracy: {accuracy * 100:.2f}%")

    # Simulate hourly data aggregation
    interval = 0.5  # assuming logs recorded every 0.5 seconds
    logs_per_hour = int(3600 / interval)
    data['Hour'] = (data.index // logs_per_hour) % 24

    # Process and count attacks by hour
    data[' Label'] = data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    hourly_attacks = data[data[' Label'] == 1].groupby('Hour').size()
    hourly_attacks = hourly_attacks.reindex(range(24), fill_value=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(hourly_attacks.index, hourly_attacks.values, color='red', alpha=0.7, label='Malicious Activity')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Malicious Logs')
    plt.title('Hourly Malicious Activity Detection')
    plt.xticks(ticks=range(24))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot as PNG
    output_path = 'hourly_attacks.png'
    plt.savefig(output_path)
    plt.close()

    return output_path



# Example usage

file_path1 = 'Thursday.csv'
result = analyze_input_file(file_path1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



end_time = time.time()
