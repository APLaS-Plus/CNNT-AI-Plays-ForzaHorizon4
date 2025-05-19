import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.fused_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=1 if in_channels < 4 else 4,
                bias=False,  # 与批归一化配合使用时不需要偏置
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.fused_conv(x)
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
    """image size: 240*144*1 -> 30*18*32 -> 15*9*64 -> 5*3*64 -> 15*128"""

    def __init__(self, input_height=240, input_width=144, output_dim=128):
        super(Conv2d_patch_embedding, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.output_dim = output_dim
        self.position_height = input_height // 48
        self.position_width = input_width // 48
        self.feature_extractor_and_position_embedding = nn.Sequential(
            ConvBlock(1, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            nn.Conv2d(64, self.output_dim, kernel_size=3, stride=3),
            PositionalEncoding2D(output_dim, self.position_height, self.position_width),
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor_and_position_embedding(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 3, 1).reshape(
            batch_size, self.position_height * self.position_width, self.output_dim
        )
        return x


class Concat(nn.Module):
    def __init__(self, max_timestep=160, attr_dim=16):
        super(Concat, self).__init__()

        self.time_embedding = nn.Embedding(max_timestep + 1, attr_dim)

        def make_scalar_mlp(out_dim: int):
            return nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, out_dim),
            )

        self.speed_embedding = make_scalar_mlp(attr_dim)
        self.steering_embedding = make_scalar_mlp(attr_dim)
        self.acceleration_embedding = make_scalar_mlp(attr_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, attr_dim))

    def forward(
        self,
        patch,
        speed,
        steering,
        acceleration,
        time_step,
        mask_attr: bool = False,
    ):
        if isinstance(time_step, int):
            time_step = torch.full(
                (patch.size(0),),
                time_step,
                dtype=torch.long,
                device=patch.device,
            )
        time_emb = self.time_embedding(time_step)
        batch_size, seq_len, _ = patch.size()

        speed_emb = self.speed_embedding(speed.unsqueeze(1))  # [batch, 16]
        steering_emb = self.steering_embedding(steering.unsqueeze(1))  # [batch, 16]
        acceleration_emb = self.acceleration_embedding(
            acceleration.unsqueeze(1)
        )  # [batch, 16]

        def expand_to_seq(x):
            # print(f"expand_to_seq input shape: {x.shape}")
            # x = x.squeeze(2)
            # print(f"expand_to_seq after squeeze shape: {x.shape}")
            x = x.expand(batch_size, seq_len, -1)
            return x

        speed_emb = expand_to_seq(speed_emb)  # [batch, seq_len, 16]
        steering_emb = expand_to_seq(steering_emb)  # [batch, seq_len, 16]
        acceleration_emb = expand_to_seq(acceleration_emb)  # [batch, seq_len, 16]
        time_emb = expand_to_seq(time_emb)  # [batch, seq_len, 16]

        if mask_attr:  # 若打开掩码，就换成 mask_token
            steering_emb = self.mask_token.expand_as(steering_emb).detach()
            acceleration_emb = self.mask_token.expand_as(acceleration_emb).detach()

        combined_token = torch.cat(
            (patch, speed_emb, steering_emb, acceleration_emb, time_emb), dim=2
        )
        return combined_token  # [batch, seq_len, 128+16+16+16+16]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=192, num_heads=8, ff_dim=4 * 192, dropout_rate=0.2):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.norm1 = nn.LayerNorm(embed_dim)
        self.global_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.local_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 15, embed_dim * 5),
            nn.LayerNorm(embed_dim * 5),
            nn.GELU(),
            nn.Linear(embed_dim * 5, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),  # Output: steering and acceleration
            nn.Tanh(),
        )

    def forward(self, input, memory_size=160):
        # input shape: [batch_size, (5*3)*20+memory_size, 192]
        _input = self.norm1(input)
        memory_queue = _input[:, :memory_size, :]
        frame_queue = _input[:, memory_size:, :]

        global_attention_output, _ = self.global_attention(
            query=memory_queue, key=_input, value=_input
        )

        mask = ~torch.tril(torch.ones(frame_queue.size(1), _input.size(1))).bool()
        mask = mask.to(device=input.device)
        local_attention_output, _ = self.local_attention(
            query=frame_queue, key=_input, value=_input, attn_mask=mask
        )

        g_l_concat = torch.cat((global_attention_output, local_attention_output), dim=1)
        ffn_output = self.ffn(g_l_concat)
        g_l_concat = self.dropout(g_l_concat) + ffn_output
        g_l_concat = self.dropout(g_l_concat)
        new_memory_queue = g_l_concat[:, :memory_size, :]

        last_15_tokens = g_l_concat[:, -15:, :]
        last_15_tokens_flat = last_15_tokens.reshape(last_15_tokens.size(0), -1)
        output = self.decoder(last_15_tokens_flat)
        steering = output[:, 0]
        acceleration = output[:, 1]

        return steering, acceleration, new_memory_queue


class CNNT(nn.Module):
    def __init__(
        self, input_height=240, input_width=144, maxtime_step=40, memory_size=160
    ):
        super(CNNT, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((input_height, input_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self.maxtime_step = maxtime_step

        self.patch_embedding = Conv2d_patch_embedding(
            input_height=input_height, input_width=input_width, output_dim=128
        )
        self.concat = Concat(max_timestep=maxtime_step)
        self.transformer_block = TransformerBlock(
            embed_dim=192, num_heads=8, ff_dim=4 * 192, dropout_rate=0.2
        )
        self.memory_size = memory_size

    def forward(
        self,
        frame_queue,
        speed_queue,
        steering_queue,
        acceleration_queue,
        new_frame,
        new_speed,
        new_steering,
        new_acceleration,
        memory_tensor=None,
        device="cpu",
        timer=None,
    ):
        # 使用timer测量处理新帧的时间
        if timer:
            with timer.measure("处理新帧"):
                if not isinstance(new_frame, torch.Tensor):
                    new_frame = self.transform(new_frame)
                new_frame_patch = self.patch_embedding(new_frame)
        else:
            if not isinstance(new_frame, torch.Tensor):
                new_frame = self.transform(new_frame)
            new_frame_patch = self.patch_embedding(new_frame)

        # 使用timer测量帧队列管理时间
        if timer:
            with timer.measure("队列管理"):
                if frame_queue is None:
                    frame_queue = torch.cat(
                        [new_frame_patch] * self.maxtime_step, dim=1
                    )
                    speed_queue = torch.cat([new_speed] * self.maxtime_step, dim=1)
                    steering_queue = torch.cat(
                        [new_steering] * self.maxtime_step, dim=1
                    )
                    acceleration_queue = torch.cat(
                        [new_acceleration] * self.maxtime_step, dim=1
                    )
                else:
                    frame_queue = torch.cat(
                        (frame_queue[:, :-16], new_frame_patch), dim=1
                    )
                    speed_queue = torch.cat((speed_queue[:, :-1], new_speed), dim=1)
                    steering_queue = torch.cat(
                        (steering_queue[:, :-1], new_steering), dim=1
                    )
                    acceleration_queue = torch.cat(
                        (acceleration_queue[:, :-1], new_acceleration), dim=1
                    )
        else:
            if frame_queue is None:
                frame_queue = torch.cat([new_frame_patch] * self.maxtime_step, dim=1)
                speed_queue = torch.cat([new_speed] * self.maxtime_step, dim=1)
                steering_queue = torch.cat([new_steering] * self.maxtime_step, dim=1)
                acceleration_queue = torch.cat(
                    [new_acceleration] * self.maxtime_step, dim=1
                )
            else:
                frame_queue = torch.cat((frame_queue[:, :-16], new_frame_patch), dim=1)
                speed_queue = torch.cat((speed_queue[:, :-1], new_speed), dim=1)
                steering_queue = torch.cat(
                    (steering_queue[:, :-1], new_steering), dim=1
                )
                acceleration_queue = torch.cat(
                    (acceleration_queue[:, :-1], new_acceleration), dim=1
                )

        # 使用timer测量特征串联时间
        if timer:
            with timer.measure("特征串联"):
                concat_results = []
                time_steps = torch.arange(1, self.maxtime_step + 1, device=device)
                for i in range(frame_queue.shape[1] // 15):
                    speed_input = speed_queue[:, i]
                    steering_input = steering_queue[:, i]
                    acceleration_input = acceleration_queue[:, i]
                    if speed_input.dim() == 1:
                        speed_input = speed_input.unsqueeze(1)
                    if steering_input.dim() == 1:
                        steering_input = steering_input.unsqueeze(1)
                    if acceleration_input.dim() == 1:
                        acceleration_input = acceleration_input.unsqueeze(1)
                    concat_results.append(
                        self.concat(
                            patch=frame_queue[:, i * 15 : i * 15 + 15],
                            speed=speed_input,
                            steering=steering_input,
                            acceleration=acceleration_input,
                            time_step=time_steps[i],
                            mask_attr=(i == frame_queue.shape[1] // 15 - 1),
                        )
                    )
                concat_tensor = torch.cat(concat_results, dim=1)
        else:
            concat_results = []
            time_steps = torch.arange(1, self.maxtime_step + 1, device=device)
            for i in range(frame_queue.shape[1] // 15):
                speed_input = speed_queue[:, i]
                steering_input = steering_queue[:, i]
                acceleration_input = acceleration_queue[:, i]
                if speed_input.dim() == 1:
                    speed_input = speed_input.unsqueeze(1)
                if steering_input.dim() == 1:
                    steering_input = steering_input.unsqueeze(1)
                if acceleration_input.dim() == 1:
                    acceleration_input = acceleration_input.unsqueeze(1)
                concat_results.append(
                    self.concat(
                        patch=frame_queue[:, i * 15 : i * 15 + 15],
                        speed=speed_input,
                        steering=steering_input,
                        acceleration=acceleration_input,
                        time_step=time_steps[i],
                        mask_attr=(i == frame_queue.shape[1] // 15 - 1),
                    )
                )
            concat_tensor = torch.cat(concat_results, dim=1)

        # 使用timer测量内存队列管理时间
        if timer:
            with timer.measure("内存队列管理"):
                if memory_tensor is None:
                    memory_tensor = concat_tensor[: self.memory_size]
                transformer_input = torch.cat((memory_tensor, concat_tensor), dim=1)
        else:
            if memory_tensor is None:
                memory_tensor = concat_tensor[: self.memory_size]
            transformer_input = torch.cat((memory_tensor, concat_tensor), dim=1)

        # 使用timer测量Transformer处理时间
        if timer:
            with timer.measure("Transformer推理"):
                steering, acceleration, new_memory_queue = self.transformer_block(
                    transformer_input, memory_size=self.memory_size
                )
        else:
            steering, acceleration, new_memory_queue = self.transformer_block(
                transformer_input, memory_size=self.memory_size
            )

        return (
            steering,
            acceleration,
            new_memory_queue,
            frame_queue,
            speed_queue,
            steering_queue,
            acceleration_queue,
        )
