import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
from config import Config


class TranslationDataset(Dataset):
    def __init__(self, de_file, en_file, de_vocab=None, en_vocab=None, build_vocab=True):
        self.config = Config()

        # 读取数据
        with open(de_file, 'r', encoding='utf-8') as f:
            self.de_sentences = [line.strip() for line in f if line.strip()]
        with open(en_file, 'r', encoding='utf-8') as f:
            self.en_sentences = [line.strip() for line in f if line.strip()]

        # 构建词汇表
        if build_vocab:
            self.de_vocab = self._build_vocab(self.de_sentences)
            self.en_vocab = self._build_vocab(self.en_sentences)
        else:
            self.de_vocab = de_vocab
            self.en_vocab = en_vocab

    def _build_vocab(self, sentences):
        counter = Counter()
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            counter.update(tokens)

        vocab = {
            '<pad>': Config.PAD_IDX,
            '<sos>': Config.SOS_IDX,
            '<eos>': Config.EOS_IDX,
            '<unk>': Config.UNK_IDX
        }

        # 保留最常见的8000个词
        for word, count in counter.most_common(8000):
            if word not in vocab and len(vocab) < 8000:
                vocab[word] = len(vocab)

        return vocab

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def _sentence_to_indices(self, sentence, vocab):
        tokens = self._tokenize(sentence)
        indices = [vocab.get(token, Config.UNK_IDX) for token in tokens[:self.config.max_length - 2]]
        indices = [Config.SOS_IDX] + indices + [Config.EOS_IDX]

        # 填充或截断
        if len(indices) < self.config.max_length:
            indices.extend([Config.PAD_IDX] * (self.config.max_length - len(indices)))
        else:
            indices = indices[:self.config.max_length]
            indices[-1] = Config.EOS_IDX

        return indices

    def __len__(self):
        return len(self.de_sentences)

    def __getitem__(self, idx):
        src_indices = self._sentence_to_indices(self.de_sentences[idx], self.de_vocab)
        tgt_indices = self._sentence_to_indices(self.en_sentences[idx], self.en_vocab)

        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long)
        }


def get_data_loaders():
    config = Config()

    # 训练集
    train_dataset = TranslationDataset(
        f"{config.data_path}/train.de",
        f"{config.data_path}/train.en"
    )

    # 验证集（使用训练集的词汇表）
    val_dataset = TranslationDataset(
        f"{config.data_path}/val.de",
        f"{config.data_path}/val.en",
        train_dataset.de_vocab,
        train_dataset.en_vocab,
        build_vocab=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.de_vocab, train_dataset.en_vocab