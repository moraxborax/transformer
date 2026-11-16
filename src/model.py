import math

import torch
import torch.nn as nn  # noqa: PLR0402


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, masked: bool = False):
        """
        Attention module
        It works by projecting the input tensor into three vectors (query, key, value)
        and then computing the attention score between the query and key vectors.
        The attention score is then used to weight the value vectors.
        The output is a weighted sum of the value vectors.
        Multi Head attention is used
        Args:
            d_model: int, the dimension of the model
            n_heads: int, the number of heads in the attention mechanism. n should be divisible by d_model.
        Returns:
            torch.Tensor, the output tensor
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.masked = masked
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the Attention module
        Args:
            x: torch.Tensor, the input tensor. Shape: (batch, seq, d_model)
            memory: encoder output for cross-attention. Queries come from x, keys and values from memory.
        Returns:
            torch.Tensor, the output tensor. Shape: (batch, seq, d_model)
        """
        batch, seq, _ = x.shape
        if memory is None:
            qkv: torch.Tensor = self.qkv(x)
            q, k, v = qkv.split(self.d_model, dim=-1)
        else:
            weight = self.qkv.weight
            bias = self.qkv.bias
            q = nn.functional.linear(x, weight[: self.d_model], None if bias is None else bias[: self.d_model])
            k = nn.functional.linear(
                memory,
                weight[self.d_model : 2 * self.d_model],
                None if bias is None else bias[self.d_model : 2 * self.d_model],
            )
            v = nn.functional.linear(
                memory,
                weight[2 * self.d_model :],
                None if bias is None else bias[2 * self.d_model :],
            )

        # (batch, length, d_model) -> (batch, n_heads, length, d_k)
        key_len = k.shape[1]
        q = q.view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch, key_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, key_len, self.n_heads, self.d_k).transpose(1, 2)

        # calculate how much each word in the sequence is related to the other words in the sequence
        # q means "what i want". k means "what i have" and v means "what i actually mean".
        # q @ k.T calculate how much they correlate. then we normalize it and softmax it for the scores
        # then we times the v 
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if self.masked:
            # scores are (batch, n_heads, seq, seq). True marks keys the query must not see.
            future = torch.ones(seq, seq, device=scores.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        if key_padding_mask is not None:
            # key_padding_mask is (batch, key_len). True marks padding.
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        blocked = ~torch.isfinite(scores).any(dim=-1, keepdim=True)
        scores = scores.masked_fill(blocked, 0.0)
        weights = scores.softmax(dim=-1)
        mixed = weights @ v

        # (batch, n_heads, seq, d_k) -> (batch, seq, d_model)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.out(mixed)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEmbedding module
        Args:
            x: torch.Tensor, the input embedding.
            torch.Tensor, the embedding with positional information.
        """
        seq = x.shape[1]
        return x + self.pe[:seq]


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = Attention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x, key_padding_mask=key_padding_mask)))
        ffn = self.linear2(torch.relu(self.linear1(x)))
        return self.norm2(x + self.dropout(ffn))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_vocab: int,
        max_seq_len: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(n_vocab, d_model)
        self.position = PositionalEmbedding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(token_ids) * math.sqrt(self.d_model)
        x = self.dropout(self.position(x))
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention = Attention(d_model, n_heads, masked=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attention = Attention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attention(x, key_padding_mask=tgt_padding_mask)))
        x = self.norm2(
            x + self.dropout(self.cross_attention(x, memory, key_padding_mask=memory_padding_mask))
        )
        ffn = self.linear2(torch.relu(self.linear1(x)))
        return self.norm3(x + self.dropout(ffn))


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_vocab: int,
        max_seq_len: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(n_vocab, d_model)
        self.position = PositionalEmbedding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.output = nn.Linear(d_model, n_vocab, bias=False)
        self.output.weight = self.embed.weight

    def forward(
        self,
        token_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(token_ids) * math.sqrt(self.d_model)
        x = self.dropout(self.position(x))
        for layer in self.layers:
            x = layer(x, memory, tgt_padding_mask, memory_padding_mask)
        # (batch, seq, d_model) -> (batch, seq, n_vocab)
        # Project back into the vocabulary space.
        return self.output(x)


