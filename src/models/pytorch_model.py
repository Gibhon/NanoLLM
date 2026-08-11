import torch
import torch.nn as nn


class NanoLLM(nn.Module):
    def __init__(self, tokenizer, embed_dim, n_heads, n_layers, n_feedforward, block_size, dropout) -> None:
        super().__init__()

        vocab_size = tokenizer.vocab_size

        self.embedding = nn.Embedding(
            embedding_dim=embed_dim, num_embeddings=vocab_size
        )
        self.pos_embedding = nn.Embedding(embedding_dim=embed_dim, num_embeddings=block_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=n_feedforward,
            batch_first=True,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers
        )
        self.output_ll = nn.Linear(in_features=embed_dim, out_features=vocab_size)

    def forward(self, token_ids):
        seq_len = token_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=token_ids.device
        )

        positions = torch.arange(seq_len, device=token_ids.device)
        pos_embed = self.pos_embedding(positions)

        embed_tokens = self.embedding(token_ids)
        embed_tokens = embed_tokens + pos_embed

        transformer_output = self.transformer(
            embed_tokens, mask=causal_mask, is_causal=True
        )
        return self.output_ll(transformer_output)
