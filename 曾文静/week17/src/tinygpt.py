# -*- coding: utf-8 -*-
"""
tinygpt.py —— 微型 GPT（字符级），从零实现，CPU/MPS 都能跑
=============================================================
作业的最小闭环不依赖 TRL / transformers / 大模型下载：
一个 1M 参数左右的字符级 GPT 就足够演示 GRPO 的全部核心机制。
"""
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. 字符级词表（所有数据只用这些字符）
# ---------------------------------------------------------------------------
CHARS = sorted(set("0123456789+-×=?:<>/\n QAanswer"))
PAD, BOS, EOS = "<PAD>", "<BOS>", "<EOS>"


def build_vocab():
    itos = [PAD, BOS, EOS] + CHARS
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos


STOI, ITOS = build_vocab()
VOCAB_SIZE = len(ITOS)


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS[i] for i in ids if ITOS[i] not in (PAD, EOS, BOS))


class GPTConfig:
    def __init__(self, vocab_size=VOCAB_SIZE, block_size=64, n_embd=128,
                 n_layer=4, n_head=4, **kw):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# 2. 微型 GPT（标准 decoder-only transformer）
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, attn_mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 因果 mask + padding mask
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal[None, None], float("-inf"))
        if attn_mask is not None:  # [B, T] 0/1, 1=有效
            pad = (~attn_mask.bool())[:, None, None, :]  # [B,1,1,T]
            att = att.masked_fill(pad.expand(B, self.n_head, T, T), float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd))

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, attn_mask=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.lm_head(self.ln_f(x))

    # ---------------- 推理 ----------------
    @torch.no_grad()
    def generate(self, prompt_ids, prompt_mask=None, max_new=32, temperature=1.0,
                 greedy=False):
        """自回归采样，遇到 <EOS> 即停止（各序列可长度不一，右侧 PAD 对齐）。
        返回 (completions[B, Lmax], logp[B, Lmax], mask[B, Lmax])，
        logp 是采样策略逐 token 的对数概率（GRPO 的 old_logprobs），
        无效位置（PAD）上 logp=0、mask=0。"""
        B = prompt_ids.shape[0]
        seq = prompt_ids.clone()
        mask = prompt_mask.clone() if prompt_mask is not None else \
            torch.ones_like(seq)
        done = torch.zeros(B, dtype=torch.bool, device=seq.device)
        logp_acc = torch.zeros(B, device=seq.device)
        for _ in range(max_new):
            x = seq[:, -self.cfg.block_size:]
            m = mask[:, -self.cfg.block_size:]
            logits = self.forward(x, m)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            if greedy:
                nxt = probs.argmax(dim=-1)
            else:
                nxt = torch.multinomial(probs, num_samples=1).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, STOI[PAD]), nxt)
            logp_acc += torch.where(done, torch.zeros_like(logp_acc),
                                    probs.log().gather(1, nxt.unsqueeze(1)).squeeze(1))
            done = done | (nxt == STOI[EOS])
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
            mask = torch.cat([mask, (~done).long().unsqueeze(1)], dim=1)
        comps = seq[:, prompt_ids.shape[1]:]
        cmask = mask[:, prompt_ids.shape[1]:]
        return comps, logp_acc.unsqueeze(1).expand(B, comps.shape[1]).contiguous(), cmask

    @torch.no_grad()
    def completion_logprobs(self, full_ids, prompt_len, attn_mask=None):
        """对给定完整序列，返回补全部分逐 token 的 log 概率（old/ref 通用）。
        返回 (logp[B, Lc], valid[B, Lc])，valid 标记真实补全 token。"""
        logits = self.forward(full_ids, attn_mask)
        logp = F.log_softmax(logits, dim=-1)
        tok = full_ids[:, prompt_len:]
        Lc = tok.shape[1]
        logp_tok = logp[:, prompt_len - 1:-1].gather(2, tok.unsqueeze(2)).squeeze(2)
        valid = (tok != STOI[PAD]).long()
        # EOS 之后预测的 PAD 也算无效（其实 EOS 后就是 PAD，天然被覆盖）
        return logp_tok * valid, valid

    # ---------------- 存取 ----------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state": self.state_dict(),
                    "cfg": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        d = torch.load(path, map_location=device, weights_only=False)
        cfg = GPTConfig(**d["cfg"])
        m = cls(cfg).to(device)
        m.load_state_dict(d["state"])
        m.eval()
        return m


def prompt_tensor(texts, max_len=None, device="cpu"):
    """把 prompt 文本拼成 [B, max_len] 张量（**自动前置 BOS**，与训练序列对齐），
    返回 (ids, mask)。max_len=None 时取本批最大长度（保证生成起始位置与训练分布一致）。"""
    if max_len is None:
        max_len = max(len(t) for t in texts) + 1  # +BOS
    ids = torch.full((len(texts), max_len), STOI[PAD], dtype=torch.long)
    for i, t in enumerate(texts):
        e = [STOI[BOS]] + encode(t)[:max_len - 1]
        ids[i, :len(e)] = torch.tensor(e)
    mask = (ids != STOI[PAD]).long()
    return ids.to(device), mask.to(device)
