import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    """翻译数据集类"""

    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_len):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # 编码源文本和目标文本
        src_encoded = self.src_tokenizer.encode(
            src_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        tgt_encoded = self.tgt_tokenizer.encode(
            tgt_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        # 转换为tensor
        src_tensor = torch.tensor(src_encoded)
        tgt_tensor = torch.tensor(tgt_encoded)

        # 为教师强制训练准备输入和目标
        tgt_input = tgt_tensor[:-1]  # 移除结束标记
        tgt_output = tgt_tensor[1:]  # 移除开始标记

        return {"src": src_tensor, "tgt_input": tgt_input, "tgt_output": tgt_output}


def create_masks(src, tgt_input):
    """创建源序列和目标序列的遮罩"""
    # 源序列遮罩
    src_mask = (src == 0).unsqueeze(1).unsqueeze(2)

    # 目标序列遮罩
    tgt_mask = (tgt_input == 0).unsqueeze(1).unsqueeze(2)

    seq_len = tgt_input.shape[1]
    nopeak_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1).astype("uint8") == 0
    nopeak_mask = torch.from_numpy(nopeak_mask).to(tgt_input.device)

    tgt_mask = tgt_mask & nopeak_mask

    return src_mask, tgt_mask


# 简单的Tokenizer示例（实际项目中可能需要更复杂的实现或使用现有库如BERT、sentencepiece等）
class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.word_count = {}
        self.next_idx = 4

    def fit(self, texts):
        """根据文本构建词表"""
        for text in texts:
            for word in text.split():
                if word not in self.word_count:
                    self.word_count[word] = 0
                self.word_count[word] += 1

        # 按频率排序并截取前N个词
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[: self.vocab_size - 4]:  # 减去特殊标记的数量
            self.word2idx[word] = self.next_idx
            self.idx2word[self.next_idx] = word
            self.next_idx += 1

    def encode(
        self,
        text,
        add_special_tokens=True,
        max_length=None,
        padding=None,
        truncation=None,
    ):
        """将文本转换为索引序列"""
        words = text.split()

        if add_special_tokens:
            tokens = [self.word2idx["<sos>"]]
        else:
            tokens = []

        for word in words:
            tokens.append(self.word2idx.get(word, self.word2idx["<unk>"]))

        if add_special_tokens:
            tokens.append(self.word2idx["<eos>"])

        # 截断
        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        # 填充
        if padding == "max_length" and max_length:
            padding_length = max_length - len(tokens)
            if padding_length > 0:
                tokens.extend([self.word2idx["<pad>"]] * padding_length)

        return tokens

    def decode(self, ids):
        """将索引序列转换回文本"""
        return " ".join(
            self.idx2word.get(id, "<unk>") for id in ids if id not in [0, 1, 2]
        )  # 排除特殊标记
