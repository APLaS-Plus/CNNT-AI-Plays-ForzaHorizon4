import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import math
from tqdm import tqdm
import os
import pathlib

from utils.model.transformer import Transformer
from utils.data_processing import TranslationDataset, SimpleTokenizer, create_masks

# 配置参数
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
dropout = 0.1
max_seq_len = 100
batch_size = 64
epochs = 10
learning_rate = 0.0001
# 添加模型保存目录参数
model_save_dir = pathlib.Path(__file__).parent.resolve() / "run"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model():
    # 确保保存目录存在
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"模型将被保存到: {model_save_dir}")
    
    # 加载数据（这里使用示例数据，实际应用中需要替换为真实数据）
    # 假设我们有英语到中文的翻译任务
    en_texts = ["hello world", "how are you", "nice to meet you"]
    zh_texts = ["你好世界", "你好吗", "很高兴认识你"]

    # 创建tokenizer
    en_tokenizer = SimpleTokenizer()
    zh_tokenizer = SimpleTokenizer()

    # 构建词表
    en_tokenizer.fit(en_texts)
    zh_tokenizer.fit(zh_texts)

    # 计算词表大小
    input_vocab_size = len(en_tokenizer.word2idx)
    target_vocab_size = len(zh_tokenizer.word2idx)

    # 创建数据集和数据加载器
    dataset = TranslationDataset(
        en_texts, zh_texts, en_tokenizer, zh_tokenizer, max_seq_len
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding标记（0）
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            src = batch["src"].to(device)
            tgt_input = batch["tgt_input"].to(device)
            tgt_output = batch["tgt_output"].to(device)

            # 前向传播
            output = model(src, tgt_input)

            # 计算损失
            loss = 0
            for i in range(output.size(1)):
                loss += criterion(output[:, i, :], tgt_output[:, i])

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新进度条
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # 更新学习率
        scheduler.step()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

        # 保存模型检查点到指定目录
        checkpoint_path = os.path.join(model_save_dir, f"transformer_checkpoint_epoch_{epoch+1}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss,
            },
            checkpoint_path,
        )

    # 保存最终模型到指定目录
    model_path = os.path.join(model_save_dir, "transformer_model.pt")
    torch.save(model.state_dict(), model_path)

    # 同时保存tokenizer到指定目录
    tokenizers_path = os.path.join(model_save_dir, "tokenizers.pt")
    torch.save(
        {"en_tokenizer": en_tokenizer, "zh_tokenizer": zh_tokenizer}, tokenizers_path
    )

    print(f"训练完成！模型和tokenizer已保存到 {model_save_dir}")
    return model, en_tokenizer, zh_tokenizer


if __name__ == "__main__":
    train_model()
