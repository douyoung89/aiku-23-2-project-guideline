import torch
import torch.nn as nn

class QFormerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, queries, embeddings):
        """
        queries: (B, Q, D_q)
        embeddings: (B, S, D_e)
        """

        # Self-attention between queries
        q1 = self.norm1(queries)    
        sa_out, _ = self.self_attn(q1, q1, q1)  # (B, Q, D_q)
        queries = queries + sa_out

        # Cross-attention between queries and embeddings
        q2 = self.norm2(queries)
        ca_out, _ = self.cross_attn(q2, embeddings, embeddings)  # (B, Q, D_q)
        queries = queries + ca_out

        # Feed Forward
        q3 = self.norm3(queries)
        ff_out = self.ffn(q3)
        queries = queries + ff_out

        return queries  # (B, Q, D_q)

class QFormer(nn.Module):
    def __init__(self, num_layers, num_queries, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

    def forward(self, embeddings):
        """
        embeddings: (B, S, D_e)
        """
        B = embeddings.size(0)
        queries = self.query_tokens.expand(B, -1, -1)  # (B, Q, D)

        for layer in self.layers:
            queries = layer(queries, embeddings)

        return queries  # (B, Q, D)
    
if __name__ == '__main__':
    model = QFormer(num_layers=8, num_queries=4, hidden_dim=256, num_heads=8, ffn_dim=128)
    x = torch.randn((1, 512, 256)) # (B, seq_len, speech_embedding_dim)
    output = model(x) # (B, num_queries, hidden_dim)
