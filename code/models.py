import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, hidden_dim_1)
        self.hidden_fc = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.output_fc = nn.Linear(hidden_dim_2, output_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        h_1 = F.relu(self.input_fc(x))
        h_1 = self.dropout(h_1)
        h_2 = F.relu(self.hidden_fc(h_1))
        h_2 = self.dropout(h_2)

        y_pred = self.output_fc(h_2)

        return y_pred
