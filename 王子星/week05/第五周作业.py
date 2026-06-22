import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ==================== 数据集 ====================

class TextDataset(Dataset):
    def __init__(self, text_file, seq_length=20):
        with open(text_file, 'r', encoding='utf-8') as f:
            self.text = f.read()

        # 创建词汇表
        self.chars = list(set(self.text))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # 编码文本
        self.encoded = [self.char_to_idx[c] for c in self.text]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.encoded) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.encoded[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y


# ==================== Transformer语言模型 ====================

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()

        # 词嵌入 + 位置嵌入
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出层
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # 词嵌入
        tok_emb = self.token_embed(x)

        # 位置编码
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos)

        # 合并
        x = tok_emb + pos_emb

        # 创建因果遮掩（causal mask）：防止模型看到未来的token
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Transformer（使用因果遮掩）
        x = self.transformer(x, mask=causal_mask)

        # 预测下一个字符
        return self.fc(x)

    def generate(self, start_text, char_to_idx, idx_to_char, max_len=50, temperature=1.0,
                 top_p=0.9, beam_width=3, device='cpu', use_beam_search=False):
        """
        生成文本，支持Temperature + Top-p 和 Beam Search
        """
        self.eval()

        if use_beam_search:
            return self._beam_search_generate(start_text, char_to_idx, idx_to_char,
                                              max_len, beam_width, device)
        else:
            return self._top_p_generate(start_text, char_to_idx, idx_to_char,
                                        max_len, temperature, top_p, device)

    def _top_p_generate(self, start_text, char_to_idx, idx_to_char, max_len,
                        temperature, top_p, device):
        """Temperature + Top-p 采样生成（增加重复惩罚）"""
        tokens = [char_to_idx.get(c, 0) for c in start_text]

        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([tokens], device=device)
                logits = self(x)[0, -1, :]

                # 重复惩罚：降低最近生成字符的概率
                repetition_penalty = 1.5
                for token_id in set(tokens[-20:]):  # 惩罚最近20个字符
                    logits[token_id] /= repetition_penalty

                # 应用temperature
                logits = logits / temperature

                # Top-p (nucleus) 采样
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)

        return ''.join([idx_to_char[i] for i in tokens])

    def _beam_search_generate(self, start_text, char_to_idx, idx_to_char,
                              max_len, beam_width, device):
        """Beam Search生成"""
        initial_tokens = [char_to_idx.get(c, 0) for c in start_text]

        # 初始化beams: [(tokens, score), ...]
        beams = [(initial_tokens, 0.0)]

        with torch.no_grad():
            for step in range(max_len):
                candidates = []

                for tokens, score in beams:
                    x = torch.tensor([tokens], device=device)
                    logits = self(x)[0, -1, :]
                    log_probs = torch.log_softmax(logits, dim=-1)

                    # 获取概率最高的beam_width个token
                    top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                    for log_prob, idx in zip(top_log_probs, top_indices):
                        new_tokens = tokens + [idx.item()]
                        new_score = score + log_prob.item()
                        candidates.append((new_tokens, new_score))

                # 按分数排序，保留最好的beam_width个
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_width]

        # 返回得分最高的序列
        best_tokens = beams[0][0]
        return ''.join([idx_to_char[i] for i in best_tokens])


# ==================== 训练函数 ====================

def train_model():
    # 文件路径配置
    TEXT_FILE = r'corpus.txt'
    MODEL_PATH = r'transformer_model.pth'

    SEQ_LEN = 30
    BATCH_SIZE = 64
    EPOCHS = 50
    EMBED_DIM = 256
    N_HEADS = 4
    N_LAYERS = 4
    LR = 0.001

    best_val_acc = 100.0
    best_model_state = None

    print("=" * 50)
    print("简单Transformer语言模型 - 训练")
    print("=" * 50)

    # 加载数据
    print("\n[1/3] 加载数据...")
    dataset = TextDataset(TEXT_FILE, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"词汇表大小: {dataset.vocab_size}")
    print(f"样本数: {len(dataset)}")

    # 创建模型
    print("\n[2/3] 创建模型...")
    model = TransformerLM(
        vocab_size=dataset.vocab_size,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS
    )
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n[3/3] 开始训练 (设备: {device})...")
    print("-" * 50)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        for x, y in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.view(-1, dataset.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f'\nEpoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}')

        if avg_loss < best_val_acc:
            best_val_acc = avg_loss
            # 保存模型和配置
            torch.save({
                'model': model.state_dict(),
                'char_to_idx': dataset.char_to_idx,
                'idx_to_char': dataset.idx_to_char,
                'config': {
                    'vocab_size': dataset.vocab_size,
                    'embed_dim': EMBED_DIM,
                    'n_heads': N_HEADS,
                    'n_layers': N_LAYERS,
                }
            }, MODEL_PATH)
            print(f"✓ 模型已更新: {MODEL_PATH}")
        else:
            pass


def generate_text(prompts=None):
    MODEL_PATH = r'transformer_model.pth'

    # 默认配置（用于兼容旧模型文件）
    DEFAULT_CONFIG = {
        'vocab_size': 2575,
        'embed_dim': 64,
        'n_heads': 4,
        'n_layers': 2,
    }

    # 加载模型
    print("=" * 50)
    print("加载模型...")
    print("=" * 50)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')

    # 兼容旧模型文件：如果没有config，使用默认配置
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("✓ 使用模型中的配置")
    else:
        config = DEFAULT_CONFIG
        print("⚠ 旧模型文件，使用默认配置")
        print(f"  如果加载失败，请重新训练模型")

    model = TransformerLM(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers']
    )
    model.load_state_dict(checkpoint['model'])

    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"✓ 模型已加载")
    print(f"✓ 设备: {device}")
    print(f"✓ 词汇表大小: {config['vocab_size']}")

    # 如果没有提供提示词，使用默认测试
    if prompts is None:
        prompts = ["中国证券报", "消费继续恢复性增长", "投资"]

    print("=" * 50)
    print("生成结果")
    print("=" * 50)

    for prompt in prompts:
        result = model.generate(prompt, char_to_idx, idx_to_char, top_p=0.92, max_len=20, temperature=1.3,
                                use_beam_search=False, beam_width=5, device=device)
        print(f"提示: {prompt}")
        print(f"生成: {result}")
        print("-" * 50)

    print("\n完成！")


if __name__ == '__main__':
    train_model()

    # print(f"输入提示词，多个词用空格隔开：")
    # generate_word = input()
    # sents = generate_word.split()
    # generate_text(sents)
