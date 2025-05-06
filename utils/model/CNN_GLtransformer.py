import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=max(1, in_channels // 4),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class PositionalEncoding2D(nn.Module):
    """2D positional encoding, adds a unique encoding to each spatial position"""

    def __init__(self, d_model, height, width):
        super().__init__()
        self.d_model = d_model

        # Create a constant positional encoding
        pe = torch.zeros(d_model, height, width)
        # Calculate position encodings
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # Encode height dimension
        for h in range(height):
            pe[0::2, h, :] = pe[0::2, h, :] + torch.sin(h * div_term).unsqueeze(
                1
            ).expand(-1, width)
            pe[1::2, h, :] = pe[1::2, h, :] + torch.cos(h * div_term).unsqueeze(
                1
            ).expand(-1, width)

        # Encode width dimension
        for w in range(width):
            pe[0::2, :, w] = pe[0::2, :, w] + torch.sin(w * div_term).unsqueeze(
                1
            ).expand(-1, height)
            pe[1::2, :, w] = pe[1::2, :, w] + torch.cos(w * div_term).unsqueeze(
                1
            ).expand(-1, height)

        # Register as buffer instead of parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, d_model, height, width]

    def forward(self, x):
        # x shape: [batch_size, d_model, height, width]
        return x + self.pe


class Conv2d_patch_embedding(nn.Module):
    """image size: 240*136*1 -> 30*17*32 -> 15*9*64 -> 5*3*64 -> 15*64"""

    def __init__(self, input_height=240, input_width=136, output_dim=64):
        super(Conv2d_patch_embedding, self).__init__()
        self.input_height = input_height
        self.input_width = input_width

        self.feature_extractor = nn.Sequential(
            ConvBlock(1, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            nn.Conv2d(32, output_dim, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=3),
        )

        self.position_encoder = PositionalEncoding2D(output_dim, 5, 3)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x)
        x = self.position_encoder(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 3, 1).reshape(batch_size, 5 * 3, 64)

        return x


class SpeedEmbedding(nn.Module):
    def __init__(self, input_dim=1, output_dim=64):
        super(SpeedEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # print(f"SpeedEmbedding input: {x.shape}")
        output = self.dropout(self.fc(x)).unsqueeze(1)
        # print(f"SpeedEmbedding output: {output.shape}")
        return output  # [batch_size, 1, output_dim]


class Concat(nn.Module):
    def __init__(self, max_timestep=160):
        super(Concat, self).__init__()
        self.speed_embedding = SpeedEmbedding(input_dim=1, output_dim=64)
        self.time_embedding = nn.Embedding(max_timestep + 1, 1)

    def forward(self, patch, speed, time_step):
        speed_token = self.speed_embedding(speed)
        combined_token = torch.cat(
            (patch, speed_token), dim=1
        )  # [batch_size, 5*3+1, 64]
        if isinstance(time_step, int):
            time_step = torch.full(
                (patch.size(0),), time_step, dtype=torch.long, device=patch.device
            )
        time_emb = self.time_embedding(time_step)
        batch_size = combined_token.size(0)
        time_feature = time_emb.unsqueeze(1).expand(
            batch_size, combined_token.size(1), -1
        )
        combined_token = torch.cat((combined_token, time_feature), dim=2)
        return combined_token  # [batch_size, 5*3+1, 64+1]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=65, num_heads=5, ff_dim=4 * 65, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.norm1 = nn.LayerNorm(embed_dim)
        self.globad_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.local_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 16, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

    def forward(self, input, memory_size=160):
        # input shape: [batch_size, (5*3+1)*40+memory_size, 65]
        # frame_queue shape: [batch_size, 640, 65]
        # memory_queue shape: [batch_size, memory_size, 65]
        _input = self.norm1(input)
        memory_queue = _input[:, :memory_size, :]
        frame_queue = _input[:, memory_size:, :]
        global_attention_output, _ = self.globad_attention(
            query=memory_queue, key=_input, value=_input
        )

        mask = ~torch.tril(torch.ones(frame_queue.size(1), _input.size(1))).bool()
        mask = mask.to(device=input.device)

        local_attention_output, _ = self.local_attention(
            query=frame_queue, key=_input, value=_input, attn_mask=mask
        )

        g_l_concat = torch.cat((global_attention_output, local_attention_output), dim=1)
        g_l_concat = g_l_concat + input
        ffn_output = self.ffn(g_l_concat)
        g_l_concat = g_l_concat + ffn_output

        g_l_concat = self.dropout(g_l_concat)
        new_memory_queue = g_l_concat[
            :, :memory_size, :
        ]  # use to combine the new frame queue
        # print(f"new_memory_queue: {new_memory_queue.shape}")
        last_16_tokens = g_l_concat[:, -16:, :]  # [batch_size, 16, 65]
        last_16_tokens_flat = last_16_tokens.reshape(
            last_16_tokens.size(0), -1
        )  # [batch_size, 16*65]
        output = self.decoder(last_16_tokens_flat)
        steering = output[:, 0]
        acceleration = output[:, 1]

        return steering, acceleration, new_memory_queue


class CNNT(nn.Module):
    def __init__(
        self, input_height=240, input_width=136, maxtime_step=40, memory_size=160
    ):
        super(CNNT, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.maxtime_step = maxtime_step

        self.patch_embedding = Conv2d_patch_embedding(
            input_height=input_height, input_width=input_width, output_dim=64
        )
        self.concat = Concat(max_timestep=maxtime_step)
        self.transformer_block = TransformerBlock(
            embed_dim=65, num_heads=5, ff_dim=4 * 65, dropout_rate=0.1
        )
        self.memory_size = memory_size

    def forward(self,frame_queue, speed_queue, new_frame, speed, memory_tensor=None, device="cpu"):
        new_frame_patch = self.patch_embedding(new_frame)
        # print(f"new_frame_patch: {new_frame_patch.shape}, speed: {speed.shape}")
        if frame_queue is None:
            frame_queue = torch.cat([new_frame_patch] * self.maxtime_step, dim=1)
            speed_queue = torch.cat([speed] * self.maxtime_step, dim=1)
            # print(
            #     f"frame_queue: {frame_queue.shape}, speed_queue: {speed_queue.shape}"
            # )
        else:
            # print(
            #     f"frame_queue: {frame_queue.shape}, new_frame_patch: {new_frame_patch.shape}"
            # )
            frame_queue = torch.cat(
                (frame_queue[:, :-15], new_frame_patch), dim=1
            )
            speed_queue = torch.cat((speed_queue[:, :-1], speed), dim=1)
        # print(
        #     f"frame_queue: {frame_queue[:,0:15].shape}, speed_queue: {speed_queue[:,0].shape}"
        # )

        # add new frame and speed to the queue
        concat_results = []
        time_steps = torch.arange(1, self.maxtime_step + 1, device=device)
        # print(time_steps, time_steps.shape)
        # print(f"time_steps: {time_steps.size()}")
        for i in range(frame_queue.shape[1] // 15):
            speed_input = speed_queue[:, i]
            if speed_input.dim() == 1:
                speed_input = speed_input.unsqueeze(1)
            concat_results.append(
                self.concat(
                    frame_queue[:, i * 15 : i * 15 + 15],
                    speed_input,
                    time_steps[i],
                )
            )
            # print(f"concat_input_queue: {concat_results[i].shape}")
        concat_tensor = torch.cat(concat_results, dim=1)

        if memory_tensor is None:
            memory_tensor = concat_tensor[: self.memory_size]
            # print(f"memory_queue: {memory_queue[0].shape}, None")

        # print(f"concat_tensor: {concat_tensor.shape}")
        # print(f"memory_tensor: {memory_tensor.shape}")

        transformer_input = torch.cat((memory_tensor, concat_tensor), dim=1)
        steering, acceleration, new_memory_queue = self.transformer_block(
            transformer_input, memory_size=self.memory_size
        )
        return steering, acceleration, new_memory_queue, frame_queue, speed_queue
