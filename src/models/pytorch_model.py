import torch
import torch.nn as nn
import torch.nn.functional as F


class NanoLLM(nn.Module):
    def __init__(
        self,
        tokenizer,
        embed_dim,
        n_heads,
        n_layers,
        n_feedforward,
        block_size,
        dropout,
    ) -> None:
        super().__init__()

        vocab_size = tokenizer.vocab_size

        self.embedding = nn.Embedding(
            embedding_dim=embed_dim, num_embeddings=vocab_size
        )
        self.pos_embedding = nn.Embedding(
            embedding_dim=embed_dim, num_embeddings=block_size
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=n_feedforward,
            batch_first=True,
            dropout=dropout,
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

    @torch.no_grad()
    def generate(
        self,
        prompt,
        tokenizer,
        max_new_tokens: int,
        sampling_temperature: float = 1.0,
        top_k_candidate_count:int=None,                #type:ignore
        max_context_size: int = 128,
    ):
        self.eval()
        device = next(self.parameters()).device

        encoded_token_ids = tokenizer.encode(prompt)

        if isinstance(encoded_token_ids, torch.Tensor):
            token_ids_tensor = encoded_token_ids.to(device)
            if token_ids_tensor.ndim==1:
                token_ids_tensor = token_ids_tensor.unsqueeze(dim=0)
        else:
            token_ids_tensor = torch.tensor(data=encoded_token_ids, dtype=torch.long, device=device).unsqueeze(dim=0)

        for steps in range(max_new_tokens):
            if(token_ids_tensor.size(dim=1) <= max_context_size):
                context_index_tensor = token_ids_tensor
            else:
                context_index_tensor = token_ids_tensor[:, -max_context_size:]

            logits = self(context_index_tensor)
            logits = logits[:, -1, :] / sampling_temperature

            if top_k_candidate_count is not None:
                v, _ = torch.topk(logits, min(top_k_candidate_count, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            token_ids_tensor = torch.cat((token_ids_tensor, idx_next), dim=-1)
        self.train()
        return token_ids_tensor
