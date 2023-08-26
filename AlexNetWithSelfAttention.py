import torch
import torch.nn as nn
import torch.optim as optim

# Define the AlexNet architecture with a self-attention layer
class AlexNetWithSelfAttention(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetWithSelfAttention, self).__init__()
        
        # Original AlexNet layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # ... other convolutional and pooling layers ...
        )
        
        # Self-attention layer
        self.self_attention = SelfAttentionModule(64)  # Adjust the input channels
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * final_conv_output_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.self_attention(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Define the self-attention module
class SelfAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionModule, self).__init__()
        
        # Implement your self-attention mechanism here
        # You can use convolutional layers to compute attention weights
        
        # Example: A simple convolutional layer for attention weights
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention_weights = self.attention_conv(x)  # Calculate attention weights
        attention_weights = torch.sigmoid(attention_weights)  # Apply sigmoid for values between 0 and 1
        x = x * attention_weights  # Element-wise multiplication with input features
        return x

# Example usage
if __name__ == "__main__":
    num_classes = 2
    model = AlexNetWithSelfAttention(num_classes)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Assuming you have your dataset and dataloader defined, you can start training
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
