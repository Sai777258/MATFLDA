import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_kernels=3, stride=1, padding=0):
        super(DynamicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_kernels = num_kernels

        # Ensure that in_channels divided by 4 is at least 1
        reduced_channels = max(in_channels // 4, 1)

        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, num_kernels, 1),
            nn.Softmax(dim=1)
        )

        self.kernels = nn.Parameter(torch.randn(num_kernels, out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        batch_size, _, _, _ = x.size()

        attention_weights = self.weight_generator(x)

        output = torch.zeros(batch_size, self.out_channels, x.size(2), x.size(3), device=x.device)
        for i in range(self.num_kernels):
            kernel = self.kernels[i]
            conv_out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)
            attention_weight = attention_weights[:, i].view(batch_size, 1, 1, 1)
            output += conv_out * attention_weight

        return output


# Value Embedding
class valueEmbedding(nn.Module):
    def __init__(self, d_input, d_model, value_linear=True, value_sqrt=True):
        super(valueEmbedding, self).__init__()
        self.value_linear = value_linear
        self.value_sqrt = value_sqrt
        self.d_model = d_model
        self.inputLinear = nn.Linear(d_input, d_model)

    def forward(self, x):
        if self.value_linear:
            x = self.inputLinear(x)
        if self.value_sqrt:
            x = x * math.sqrt(self.d_model)
        return x


# Positional Embedding
class positionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(positionalEmbedding, self).__init__()
        pos_emb = torch.zeros(max_len, d_model).float()
        pos_emb.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        pos_emb = pos_emb.unsqueeze(0)
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x):
        return self.pos_emb[:, :x.size(1)]


# Data Embedding Layer
class dataEmbedding(nn.Module):
    def __init__(self, d_input, d_model, value_linear=True, value_sqrt=True, posi_emb=True, input_dropout=0.05):
        super(dataEmbedding, self).__init__()
        self.posi_emb = posi_emb
        self.value_embedding = valueEmbedding(d_input, d_model, value_linear, value_sqrt)
        self.positional_embedding = positionalEmbedding(d_model)
        self.inputDropout = nn.Dropout(input_dropout)

    def forward(self, x):
        if self.posi_emb:
            x = self.value_embedding(x) + self.positional_embedding(x)
        else:
            x = self.value_embedding(x)
        return self.inputDropout(x)


# AgentAttention Mechanism
class AgentAttention(nn.Module):
    def __init__(self, dim, seq_length, num_heads=8, attn_drop=0., proj_drop=0., agent_num=49):
        super(AgentAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.agent_num = agent_num
        self.scale = head_dim ** -0.5

        self.get_v = nn.Linear(dim, dim)  # Replacing convolution with linear transformation

        self.attn_drop = nn.Dropout(attn_drop)

        # Initialize attention bias
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, seq_length))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, seq_length))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)

        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool1d(output_size=pool_size)
        self.softmax = nn.Softmax(dim=-1)

    def get_lepe(self, x, func):
        """
        Linear transformation replaces convolution for matrix data.
        """
        x = func(x)  # Apply linear transformation
        return x

    def forward(self, qkv):
        """
        Args:
        - qkv: tuple of query, key, value matrices (B, L, C) format.
        """
        q, k, v = qkv

        B, L, C = q.shape
        assert L == k.shape[1] == v.shape[1], "Input sequence lengths do not match"

        # Apply linear transformation (LEPE)
        v, lepe = self.get_lepe(v, self.get_v), self.get_lepe(v, self.get_v)

        b, n, c = q.shape
        num_heads, head_dim = self.num_heads, self.dim // self.num_heads

        # Create agent tokens using pooling
        agent_tokens = self.pool(q.permute(0, 2, 1)).reshape(b, -1, num_heads * head_dim).permute(0, 2, 1)

        # Reshape for multi-head attention
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, num_heads, -1, head_dim)

        # Agent-based attention
        position_bias1 = self.an_bias.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias1)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        # Original attention mechanism
        agent_bias1 = self.na_bias.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias1)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        # Reshape attention output back to original form
        x = x.transpose(1, 2).reshape(b, n, c)
        x = x + lepe

        return x


# Define LocalAttention
class MultiHeadLocalAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size):
        super(MultiHeadLocalAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_k = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, L, D = x.size()
        # Linear projections
        Q = self.query_linear(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, L, d_k]
        K = self.key_linear(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value_linear(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # Local attention within windows
        output = torch.zeros_like(Q)
        for i in range(0, L, self.window_size):
            end = min(i + self.window_size, L)
            Q_win = Q[:, :, i:end, :]
            K_win = K[:, :, i:end, :]
            V_win = V[:, :, i:end, :]

            scores = torch.matmul(Q_win, K_win.transpose(-2, -1)) / (self.d_k ** 0.5)
            attn_weights = self.softmax(scores)
            attended = torch.matmul(attn_weights, V_win)
            output[:, :, i:end, :] = attended

        # Combine heads
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_linear(output)


# Update encoderLayer to use both AgentAttention and LocalAttention
class encoderLayer(nn.Module):
    def __init__(self, attention_layer, local_attention_layer, d_model, d_ff, add, norm, ff, encoder_dropout=0.05):
        super(encoderLayer, self).__init__()

        d_ff = int(d_ff * d_model)
        self.attention_layer = attention_layer
        self.local_attention_layer = local_attention_layer

        self.add = add
        self.norm = norm
        self.ff = ff

        # Pointwise feed forward network with ReLU
        self.feedForward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(encoder_dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # AgentAttention output
        agent_attn_output = self.attention_layer((x, x, x))

        # MultiHeadLocalAttention output
        local_attn_output = self.local_attention_layer(x)

        # Combine both attentions (e.g., by summing or concatenating)
        new_x = agent_attn_output + local_attn_output

        if self.add:
            if self.norm:
                out1 = self.norm1(x + self.dropout(new_x))
                out2 = self.norm2(out1 + self.dropout(self.feedForward(out1)))
            else:
                out1 = x + self.dropout(new_x)
                out2 = out1 + self.dropout(self.feedForward(out1))
        else:
            if self.norm:
                out1 = self.norm1(self.dropout(new_x))
                out2 = self.norm2(self.dropout(self.feedForward(out1)))
            else:
                out1 = self.dropout(new_x)
                out2 = self.dropout(self.feedForward(out1))

        if self.ff:
            return out2
        else:
            return out1


class encoder(nn.Module):
    def __init__(self, encoder_layers):
        super(encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(self, x):
        # x [B, L, D]
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x


class LDAformer(nn.Module):
    def __init__(self, seq_len, d_input, d_model=16, agent_n_heads=8, local_n_heads=4, e_layers=5, d_ff=2, agent_num=1,
                 pos_emb=False,
                 value_linear=True, value_sqrt=False, add=True, norm=True, ff=True, dropout=0.05, local_window_size=5):
        super(LDAformer, self).__init__()

        # Five Dynamic Convolutional Layers
        self.dynamic_conv_layers = nn.ModuleList([
            DynamicConv2d(in_channels=d_input, out_channels=d_input, kernel_size=3, padding=1) for _ in range(3)
        ])

        # Encoding
        self.data_embedding = dataEmbedding(d_input, d_model, posi_emb=pos_emb, value_linear=value_linear,
                                            value_sqrt=value_sqrt, input_dropout=dropout)

        # Encoder
        self.encoder = encoder(
            [
                encoderLayer(
                    AgentAttention(d_model, seq_len, num_heads=agent_n_heads, agent_num=agent_num),
                    MultiHeadLocalAttention(d_model, local_n_heads, local_window_size),
                    d_model,
                    d_ff,
                    add=add,
                    norm=norm,
                    ff=ff,
                    encoder_dropout=dropout
                ) for l in range(e_layers)
            ]
        )

        self.prediction = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Ensure input is 4D for convolution (batch_size, channels, height, width)
        x = x.unsqueeze(-1).permute(0, 2, 1, 3)  # Convert (B, L, D) to (B, D, L, 1)

        for conv in self.dynamic_conv_layers:
            # Apply dynamic convolution
            x_conv = conv(x).squeeze(-1).permute(0, 2, 1)  # Convert back to (B, L, D)

            # Residual connection
            x = x.squeeze(-1).permute(0, 2, 1)  # Convert (B, D, L, 1) to (B, L, D) for residual addition
            x = x_conv + x  # Add the residual connection

            # Prepare for next convolution
            x = x.unsqueeze(-1).permute(0, 2, 1, 3)  # Convert (B, L, D) to (B, D, L, 1)

        # After all convolutions
        x = x.squeeze(-1).permute(0, 2, 1)  # Convert back to (B, L, D)

        # Data embedding
        x_embedding = self.data_embedding(x)

        # Encoder output
        enc_out = self.encoder(x_embedding)

        # Prediction
        return self.prediction(enc_out).squeeze()