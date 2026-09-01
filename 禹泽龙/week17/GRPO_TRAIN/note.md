全量 GRPO 训练日志分析
训练时间：197.8 秒（200 步）

核心指标变化
指标	             Step 5 (初始)	    Step 200 (结束)	变化
reward_correct/mean  0.77	            0.94	       ↑
reward_format/mean	 0.09	            0.20（到顶）	↑↑
entropy	             0.24	            0.017  	        ↓↓↓
frac_reward_zero_std	0.0	            0.9	            ↑↑↑
clip_ratio	            0	            0	            —

关键发现
1. 格式分到顶了（0.20 = 100% 带标签）
格式率在第 20 步就触及天花板（0.20 = 每个 completion 都带 <answer> 标签），说明格式这个"简单任务"被模型迅速掌握。

2. 正确率从 0.77 → 0.94
提升明显，但注意到 entropy 也从 0.24 压到了 0.017，策略已经极度确定性——模型基本只生成一种答案路径。
高 entropy（0.24）：策略多样，模型输出五花八门，探索空间大
低 entropy（0.017）：策略单一，模型几乎只生成一种答案
0.24 → 0.017：策略从"多路径探索"收敛为"几乎确定"
这是 RL 的双刃剑：演示任务里熵降是收敛标志；长训练中持续降意味着多样性枯竭

3. frac_reward_zero_std 暴涨到 0.9
90% 的组变成全对或全错，意味着后期梯度信号枯竭：组内没有方差，advantage ≈ 0，几乎学不到东西。

4. clip_ratio = 0 全程没触发
PPO clip 从未发生，说明策略变化太小，KL/clip 约束根本没触碰到边界——正常但也说明模型更新幅度很小。


训练后评估
需要用 probe_baseline.py 对训练后的 checkpoint 做评估：
python src/probe_baseline.py --model outputs/grpo_ckpt --out outputs/post_train_probe.json

全量 GRPO 训练前后对比（Qwen2.5-0.5B）
难度	        指标	        训练前	训练后	变化
L1 (个位加法)	greedy_strict	0.60	1.00	+0.40
               greedy_format	0.60	1.00	+0.40
L2 (两位加减)	greedy_strict	0.40	1.00	+0.60
                greedy_format	0.40	1.00	+0.60
L3 (三位加减)	greedy_strict	0.80	0.92	+0.12
               greedy_format	0.90	1.00	+0.10
L4 (表内乘法)	greedy_strict	0.30	1.00	+0.70
                greedy_format	0.30	1.00	+0.70
L5 (两位×一位)	greedy_strict	0.60	0.96	+0.36
               greedy_format	0.60	1.00	+0.40
L6 (两位×两位)	greedy_strict	0.30	0.58	+0.28
               greedy_format	0.50	1.00	+0.50
关键观察
1. 格式遵循率全面 100%
所有难度包括未训练的 L4/L6，格式率全部到 1.0。格式是"表层行为"，RL 极容易学会且完全泛化。

2. 训练集内难度提升巨大
L2：0.40 → 1.00（+0.60）
L5：0.60 → 0.96（+0.36）
L3：0.80 → 0.92（+0.12，已接近天花板）
3. 未训练难度也在涨（泛化）
L4（表内乘法）：0.30 → 1.00，完全没进训练集
L6（两位×两位）：0.30 → 0.58，进训练集但配比低
4. 训练后 informative group rate 塌了
L1: 1.0 → 0.06
L2: 1.0 → 0.08
因为模型太强了——几乎所有采样全对，组内无方差 → advantage ≈ 0 → 梯度信号枯竭。这就是 ARCHITECTURE.md 里说的"后期退化组比例高达 0.8~0.95"的体现。

结论：200 步 GRPO 效果显著，但训练余地已经不大——模型在训练集上已经接近收敛。

trl（Transformer Reinforcement Learning）是 HuggingFace 生态的 RL 训练库，专门用来训练 LLM 的 RL 环节。
trl 能做什么：
**trl + 奖励函数 → 直接 RL 训练 LLM**
算法	trl 对应的 Trainer
PPO	    PPOTrainer
GRPO	GRPOTrainer
DPO	    DPOTrainer
KTO	    KTOTrainer

GRPO 为什么用 trl
手写 GRPO 需要自己实现：
    组内采样 + 奖励计算
    advantage 归一化
    PPO-clip 策略更新
    KL 约束（参考模型）

trl.GRPOTrainer 把这些全部封装好了，你只需要提供：
    模型（model）
    奖励函数（reward_funcs）
    训练数据（train_dataset）
然后调 .train() 就行，40 行代码实现完整 GRPO。
和 transformers 的关系
    transformers  — 模型定义 / 加载 / 生成
    trl            — RL 训练循环封装
trl 底层依赖 transformers 的模型和 tokenizer，自己只负责 RL 逻辑（advantage、clip、gradient update）。

参考资料
pip install trl
官网：https://huggingface.co/docs/trl
GitHub：https://github.com/huggingface/trl
ARCHITECTURE.md §3.1 里那张配置表（num_generations=8、beta=0.0、epsilon=0.2）就是 GRPO 的核心超参，通过 GRPOConfig 传进去。