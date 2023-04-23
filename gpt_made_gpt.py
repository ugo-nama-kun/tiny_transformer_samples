import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class GPT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout):
        super().__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_embedding = nn.Embedding(1000, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=4*hidden_size, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x is a tensor of shape (batch_size, sequence_length, input_size)

        x = self.embedding(x)
        seq_length = x.shape[1]

        pos = torch.arange(seq_length, device=x.device).unsqueeze(0)
        pos = self.pos_embedding(pos)

        x = x + pos

        # reshape the input tensor to be compatible with the transformer
        x = x.permute(1, 0, 2)
        
        src_mask = torch.zeros((seq_length, seq_length), device=x.device).type(torch.bool)

        # apply the transformer layers
        for transformer in self.transformer_blocks:
            x = transformer(x, src_mask=src_mask, is_causal=True)

        # reshape the output tensor to its original shape
        x = x.permute(1, 0, 2)
        x = self.decoder(x)

        return x


# set up the GPT model
input_size = 10
hidden_size = 24
output_size = 10
num_layers = 2
num_heads = 8
dropout = 0.0

gpt = GPT(input_size, hidden_size, output_size, num_layers, num_heads, dropout)

# set up the loss function and optimizer
criterion = nn.MSELoss()

# generate some example data
batch_size = 32
sequence_length = 20
inputs = torch.randn(batch_size, sequence_length, input_size)
targets = torch.randn(batch_size, sequence_length, output_size)

# train the model
num_epochs = 1000

model = nn.Sequential(
    GPT(input_size, hidden_size, output_size, num_layers, num_heads, dropout),
    nn.Linear(output_size, 1)
)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    running_loss = 0.0

    # loop over the batches in the training data
    for i in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = torch.zeros(batch_size) + 10

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(inputs)))
