import torch
import torch.nn as nn

class SubsamplingConv1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),  # halve length
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=2, padding=1),  # halve again
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x: (B, seq_len, speech_embedding_dim)
        """

        return self.conv(x)  # (B, output_dim, seq_len / 4)

class SpeechEncoder(nn.Module):
    def __init__(self, speech_embedding_dim, hidden_dim, sampling_output_dim, output_dim, num_layers=6, num_heads=8):
        super().__init__()

        self.subsampler = SubsamplingConv1D(speech_embedding_dim, hidden_dim, sampling_output_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=sampling_output_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(sampling_output_dim, output_dim)

    def forward(self, x):
        """
        x: (B, seq_len, speech_embedding_dim)
        """

        x = x.transpose(1, 2)
        x = self.subsampler(x)  # (B, sampling_output_dim, seq_len / 4)
        x = x.transpose(1, 2)   # (B, seq_len / 4, sampling_output_dim)
        x = self.transformer_encoder(x)  # (B, seq_len / 4, sampling_output_dim)
        x = self.fc(x)  # (B, seq_len / 4, output_dim)
        
        return x
    
if __name__ == '__main__':
    model = SpeechEncoder(speech_embedding_dim=256, hidden_dim=128, sampling_output_dim=64, output_dim=32)
    x = torch.randn((1, 512, 256)) # (B, seq_len, speech_embedding_dim)
    output = model(x) # (B, seq_len / 4, output_dim)