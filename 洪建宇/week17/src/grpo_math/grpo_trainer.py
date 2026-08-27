"""GRPO 核心训练器。

实现流程（与 GRPO 论文 / DeepSeekMath 一致）：
  1. 组采样   : 对每个 prompt 用旧策略 π_θ_old 采样 G 条完成
  2. 奖励     : 用可验证奖励函数（正确性 + 格式）打分
  3. 优势计算 : 组内归一化 A_i = (r_i - mean) / (std + ε)，无需 value/critic
  4. 策略更新  : 类 PPO 截断目标 + 相对参考模型的 KL 惩罚
  5. 训练循环 : prompt mini-batch × epoch
"""
import math
import os
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
)

from .config import GRPOConfig
from .data import build_prompt
from .rewards import compute_reward


class GRPOTrainer:
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if config.bf16 and self.device.type == "cuda" \
            else torch.float32

        # ---- tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- policy (可训练) ----
        self.policy = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, torch_dtype=self.dtype
        ).to(self.device)
        self.policy.config.use_cache = False
        if config.gradient_checkpointing:
            self.policy.gradient_checkpointing_enable()

        # ---- reference policy (冻结) ----
        ref_path = config.ref_model_name_or_path or config.model_name_or_path
        self.ref_policy = AutoModelForCausalLM.from_pretrained(
            ref_path, torch_dtype=self.dtype
        ).to(self.device)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        # ---- optimizer / scheduler ----        decay, no_decay = [], []
        for n, p in self.policy.named_parameters():
            (no_decay if p.ndim < 2 or n.endswith("bias") else decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.0},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=config.learning_rate,
        )
        self._scheduler = None  # 在 train() 中按总步数初始化

        self.global_step = 0

    # ------------------------------------------------------------------
    # 组采样：对每个 prompt 采样 G 条完成
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_completions(self, prompts: List[str]) -> Dict:
        """左 padding 批量采样。返回完整序列、attention、prompt 长度 P、文本。"""
        self.policy.eval()
        tok = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            padding_side="left", truncation=True,
            max_length=self.config.max_prompt_length,
        ).to(self.device)
        prompt_ids, prompt_attn = tok["input_ids"], tok["attention_mask"]
        P = prompt_ids.shape[1]  # padding 后 prompt 长度

        gen_kwargs = dict(
            do_sample=self.config.temperature > 0,
            temperature=max(self.config.temperature, 1e-2),
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
            num_return_sequences=self.config.num_generations,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 生成阶段临时打开 KV cache、关闭梯度检查点，加速自回归采样
        gc_on = bool(getattr(self.policy, "is_gradient_checkpointing", False))
        try:
            if gc_on:
                self.policy.gradient_checkpointing_disable()
            self.policy.config.use_cache = True
            full_ids = self.policy.generate(
                input_ids=prompt_ids, attention_mask=prompt_attn, **gen_kwargs
            )  # [B*G, P + max_new]
        finally:
            self.policy.config.use_cache = False
            if gc_on:
                self.policy.gradient_checkpointing_enable()

        # 完整序列的 attention：左 pad(0) + prompt(1) + 生成(1) + 右 pad(0)
        full_attn = (full_ids != self.tokenizer.pad_token_id).long()
        # 恢复 prompt 段的 attention，防止 prompt 自身含 pad token 被误清
        full_attn[:, :P] = prompt_attn.repeat_interleave(
            self.config.num_generations, dim=0
        )

        texts = self.tokenizer.batch_decode(
            full_ids[:, P:], skip_special_tokens=True
        )

        # 在 eval/no_grad 态计算旧策略对数概率，保证与采样分布一致（含 dropout 关闭）
        old_logps, mask = self.per_token_logps(self.policy, full_ids, full_attn, P)
        return {
            "full_ids": full_ids,
            "full_attn": full_attn,
            "prompt_len": P,
            "texts": texts,
            "old_logps": old_logps.detach(),
            "mask": mask.detach(),
        }

    # ------------------------------------------------------------------
    # 逐 token 对数概率（mask 掉 prompt，只保留完成部分）
    # ------------------------------------------------------------------
    def per_token_logps(self, model, full_ids: torch.Tensor,
                        full_attn: torch.Tensor, prompt_len: int):
        """返回 per-token logp [N, L-1] 与 completion mask [N, L-1]。

        为避免一次性物化 [N, T, V]（V~15 万）导致 OOM：
        - 按 batch 维分块（logps_chunk_size）前向；
        - 用 gather + logsumexp 取目标 token 的 logp，避免 log_softmax 全量张量。
        """
        N, L = full_ids.size()
        shift_labels = full_ids[:, 1:]                    # 预测目标
        chunk = max(1, self.config.logps_chunk_size)

        parts = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            logits = model(input_ids=full_ids[s:e],
                           attention_mask=full_attn[s:e]).logits[:, :-1, :]
            # logp(label) = logit(label) - logsumexp(logits)
            gathered = logits.gather(-1, shift_labels[s:e].unsqueeze(-1)).squeeze(-1)
            lse = torch.logsumexp(logits, dim=-1)
            parts.append(gathered - lse)
        token_logps = torch.cat(parts, dim=0)             # [N, L-1]

        # 完成位置 = 位置 p>=P 且 attention=1；对齐到 label(p) -> logp 索引 p-1
        pos = torch.arange(L, device=full_ids.device)
        is_completion = (pos.unsqueeze(0) >= prompt_len) & (full_attn.bool())
        mask = is_completion[:, 1:].to(token_logps.dtype)
        return token_logps, mask

    # ------------------------------------------------------------------
    # 优势计算：组内归一化
    # ------------------------------------------------------------------
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """rewards: [N=B*G] -> 优势 [N]，按 (B, G) 归一化。"""
        G = self.config.num_generations
        r = rewards.view(-1, G)
        mean = r.mean(dim=1, keepdim=True)
        std = r.std(dim=1, keepdim=True)
        adv = (r - mean) / (std + self.config.advantage_eps)
        return adv.view(-1)

    # ------------------------------------------------------------------
    # 策略更新：截断目标 + KL 惩罚
    # ------------------------------------------------------------------
    def update_policy(self, batch: Dict, advantages: torch.Tensor):
        full_ids = batch["full_ids"]
        full_attn = batch["full_attn"]
        P = batch["prompt_len"]
        old_logps = batch["old_logps"]   # 采样时（eval 态）已计算
        mask = batch["mask"]
        G = self.config.num_generations
        N = full_ids.size(0)
        assert N % G == 0

        # 参考策略对数概率（无梯度；ref 始终 eval）
        with torch.no_grad():
            ref_logps, _ = self.per_token_logps(self.ref_policy, full_ids, full_attn, P)
        # 新策略对数概率（有梯度，policy 处于 train 态）
        new_logps, _ = self.per_token_logps(self.policy, full_ids, full_attn, P)

        # 统一升到 float32 计算目标，避免 bf16 下 exp(ratio)/KL 数值溢出
        old = old_logps.float()
        new = new_logps.float()
        ref = ref_logps.float()
        maskf = mask.float()
        adv = advantages.view(N, 1).float()

        ratio = torch.exp(new - old)                         # per-token 概率比
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * maskf).sum() / maskf.sum().clamp(min=1.0)

        # k3 KL 估计：exp(d) - d - 1，d = new - ref
        d = new - ref
        kl = torch.exp(d) - d - 1.0
        kl_loss = (kl * maskf).sum() / maskf.sum().clamp(min=1.0)

        loss = policy_loss + self.config.beta * kl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                       self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {
            "loss": loss.detach().float().item(),
            "policy_loss": policy_loss.detach().float().item(),
            "kl": kl_loss.detach().float().item(),
            "clip_frac": ((ratio > 1 + self.config.clip_eps) |
                          (ratio < 1 - self.config.clip_eps)).float().mean().item(),
        }

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------
    def train(self, examples: List[Dict[str, str]]):
        cfg = self.config
        torch.manual_seed(cfg.seed)

        # 每 batch 取 prompt_batch_size 个 prompt
        def chunked():
            for i in range(0, len(examples), cfg.prompt_batch_size):
                yield examples[i:i + cfg.prompt_batch_size]
        steps_per_epoch = math.ceil(len(examples) / cfg.prompt_batch_size)
        total_steps = steps_per_epoch * cfg.num_epochs

        self._scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=cfg.warmup_steps,
            num_training_steps=total_steps,
        )
        os.makedirs(cfg.output_dir, exist_ok=True)

        for epoch in range(cfg.num_epochs):
            for batch_examples in chunked():
                prompts = [build_prompt(ex["problem"], self.tokenizer)
                           for ex in batch_examples]
                gt_answers = [ex["answer"] for ex in batch_examples]

                # 1) 组采样
                gen = self.generate_completions(prompts)
                # 2) 奖励
                rewards_list, reward_details = [], []
                for txt, gt in zip(gen["texts"], gt_answers * cfg.num_generations):
                    r, d = compute_reward(
                        txt, gt, cfg.reward_correctness_weight, cfg.reward_format_weight
                    )
                    rewards_list.append(r)
                    reward_details.append(d)
                rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
                # 3) 优势
                advantages = self.compute_advantages(rewards)

                # 4) 策略更新（对整组一次性更新；如需更细粒度可再拆分 mini-batch）
                self.policy.train()
                stats = self.update_policy(gen, advantages)
                self._scheduler.step()

                self.global_step += 1
                if self.global_step % cfg.logging_steps == 0:
                    acc = sum(d["correctness"] for d in reward_details) / len(reward_details)
                    print(f"[epoch {epoch} step {self.global_step}] "
                          f"loss={stats['loss']:.4f} policy={stats['policy_loss']:.4f} "
                          f"kl={stats['kl']:.4f} clip={stats['clip_frac']:.3f} "
                          f"reward={rewards.mean().item():.3f} acc={acc:.3f}")
                if self.global_step % cfg.save_steps == 0:
                    self.save()

        self.save()
        print(f"训练完成，共 {self.global_step} 步。")

    def save(self):
        path = os.path.join(self.config.output_dir, f"step_{self.global_step}")
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"模型已保存到 {path}")
