from torch import nn

class base_Model(nn.Module):
    def __init__(self, features_len, kernel_size, input_channels, 
                 stride, dropout, final_out_channels, num_classes, timesteps):
        super(base_Model, self).__init__()
        feature_length = features_len
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )
        feature_length = feature_length // stride + 1 
        feature_length = feature_length // 2 + 1

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        feature_length += 1
        feature_length = feature_length // 2 + 1

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        feature_length += 1
        feature_length = feature_length // 2 + 1
        timesteps = (feature_length + 1)// 2
        model_output_dim = feature_length
        self.logits = nn.Linear(model_output_dim * final_out_channels, num_classes)

    def forward(self, x_in):
        x_in = x_in
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x
