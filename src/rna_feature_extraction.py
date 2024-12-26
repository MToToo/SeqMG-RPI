import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-channel RNA feature extractor
class RNAFeatureExtractor(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(RNAFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels * 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=input_channels * 4, out_channels=input_channels * 8, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=input_channels * 8, out_channels=input_channels * 16, kernel_size=3,
                               padding=1)
        self.pool = nn.AdaptiveMaxPool2d((4, 4))
        self.lstm = nn.LSTM(input_size=input_channels * 16, hidden_size=output_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        x = x.squeeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Convert the output of the convolutional layer to the input format of the LSTM
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, height * width, channels)

        lstm_out, (h_n, c_n) = self.lstm(x)
        return lstm_out[:, -1, :]  # Return the output of the last time step of the LSTM


# RNA k-mer frequency feature extractor
class RNAFeatureExtractorKmer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RNAFeatureExtractorKmer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (input_dim // 2 // 2 // 2), 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
