"""
奖励函数单测：纯 CPU 运行，无需 GPU/模型/trl

覆盖：
  1. 默认配置与原 train_grpo.py 行为一致（correct 1.0 + format 0.2）
  2. 各奖励分量单独工作（correct/format/cot_step/length_penalty）
  3. 复合奖励总分计算正确
  4. 权重为 0 的分量不纳入 build_reward_funcs
  5. TRL 兼容签名：completions 为 [{"content": ...}] 列表
  6. 边界情况：空输出、超长输出、无数字输出

运行：
  python src/test_rewards.py
  python -m pytest src/test_rewards.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rewards import (
    RewardConfig, build_reward_funcs, compute_total_reward,
    make_reward_correct, make_reward_format,
    make_reward_cot_step, make_reward_length_penalty,
)


# ── 辅助：构造 TRL 格式的 completions ──────────────────────────────────────
def comp(text: str):
    """TRL 传入的 completion 格式: [{"role": "assistant", "content": text}]"""
    return [{"content": text}]


def comp_batch(texts):
    """批量构造 completions（TRL 传入的是列表的列表）。"""
    return [comp(t) for t in texts]


# ── 测试用例 ───────────────────────────────────────────────────────────────
def test_default_config_matches_original():
    """默认配置: correct=1.0 + format=0.2，与原 train_grpo 行为一致。"""
    cfg = RewardConfig()  # 默认值
    assert cfg.weight_correct == 1.0
    assert cfg.weight_format == 0.2
    assert cfg.weight_cot_step == 0.0
    assert cfg.weight_length_penalty == 0.0

    # 对+格式 -> 1.2
    r = compute_total_reward("<answer>85</answer>", 85, cfg)
    assert r["correct"] == 1.0
    assert r["format"] == 0.2
    assert r["total"] == 1.2

    # 对无格式 -> 1.0
    r = compute_total_reward("85", 85, cfg)
    assert r["correct"] == 1.0
    assert r["format"] == 0.0
    assert r["total"] == 1.0

    # 错+格式 -> 0.2
    r = compute_total_reward("<answer>90</answer>", 85, cfg)
    assert r["correct"] == 0.0
    assert r["format"] == 0.2
    assert r["total"] == 0.2

    # 错无格式 -> 0.0
    r = compute_total_reward("90", 85, cfg)
    assert r["total"] == 0.0
    print("[PASS] 默认配置与原 train_grpo 行为一致")


def test_reward_correct_factory():
    """reward_correct 工厂函数签名兼容 TRL。"""
    cfg = RewardConfig(weight_correct=1.0)
    fn = make_reward_correct(cfg)
    completions = comp_batch(["<answer>85</answer>", "90", "<answer>85</answer>"])
    answers = [85, 85, 85]
    rewards = fn(completions, answers)
    assert rewards == [1.0, 0.0, 1.0], f"期望 [1.0, 0.0, 1.0]，得到 {rewards}"
    print("[PASS] reward_correct 工厂函数工作正常")


def test_reward_format_factory():
    """reward_format 工厂函数：只看格式不看对错。"""
    cfg = RewardConfig(weight_format=0.2)
    fn = make_reward_format(cfg)
    completions = comp_batch(["<answer>85</answer>", "90", "<answer>999</answer>"])
    rewards = fn(completions)
    assert rewards == [0.2, 0.0, 0.2], f"期望 [0.2, 0.0, 0.2]，得到 {rewards}"
    print("[PASS] reward_format 工厂函数工作正常")


def test_cot_step_reward():
    """CoT 步骤奖励：命中推理模式给分。"""
    cfg = RewardConfig(weight_cot_step=0.3)
    fn = make_reward_cot_step(cfg)
    completions = comp_batch([
        "47 + 38 = 85，<answer>85</answer>",   # 命中 "=" 数字
        "<answer>85</answer>",                  # 无推理痕迹
        "因此答案是 <answer>85</answer>",       # 命中 "因此"
        "Step 1: 计算 47+38\n<answer>85</answer>",  # 命中 step
    ])
    rewards = fn(completions)
    assert rewards[0] == 0.3, f"应命中 CoT: {rewards[0]}"
    assert rewards[1] == 0.0, f"无推理不应给分: {rewards[1]}"
    assert rewards[2] == 0.3, f"应命中 '因此': {rewards[2]}"
    assert rewards[3] == 0.3, f"应命中 'Step': {rewards[3]}"
    print("[PASS] CoT 步骤奖励工作正常")


def test_length_penalty():
    """长度惩罚：过短扣分、理想区间不罚、过长扣分。"""
    cfg = RewardConfig(
        weight_length_penalty=1.0,
        length_ideal_min=5,
        length_ideal_max=80,
        length_penalty_strength=0.1,
    )
    fn = make_reward_length_penalty(cfg)
    completions = comp_batch([
        "85",                              # 长度 2 < 5，罚 (5-2)*0.1=0.3
        "<answer>42</answer>",             # 长度 19，在 [5,80] 内，不罚
        "<answer>85</answer>" + " " * 100, # 长度 119 > 80，罚 (119-80)*0.1=3.9
    ])
    rewards = fn(completions)
    assert abs(rewards[0] - (-0.3)) < 1e-6, f"过短惩罚错误: {rewards[0]}"
    assert rewards[1] == 0.0, f"理想区间不应罚: {rewards[1]}"
    assert abs(rewards[2] - (-3.9)) < 1e-6, f"过长惩罚错误: {rewards[2]}"
    print("[PASS] 长度惩罚工作正常")


def test_build_reward_funcs_filtering():
    """权重为 0 的分量不纳入 build_reward_funcs。"""
    # 全开
    cfg_all = RewardConfig(1.0, 0.2, 0.3, 1.0)
    funcs, names = build_reward_funcs(cfg_all)
    assert len(funcs) == 4
    assert names == ["reward_correct", "reward_format", "reward_cot_step", "reward_length_penalty"]

    # 只开 correct
    cfg_one = RewardConfig(1.0, 0.0, 0.0, 0.0)
    funcs, names = build_reward_funcs(cfg_one)
    assert len(funcs) == 1
    assert names == ["reward_correct"]

    # 全关应报错
    try:
        build_reward_funcs(RewardConfig(0, 0, 0, 0))
        raise AssertionError("全零权重应报错")
    except ValueError:
        pass
    print("[PASS] build_reward_funcs 权重过滤正确")


def test_total_reward_with_all_components():
    """四个分量全开时的总分计算。"""
    cfg = RewardConfig(1.0, 0.2, 0.3, 1.0)
    # 对 + 格式 + CoT + 理想长度
    text = "47 + 38 = 85，<answer>85</answer>"  # 长度 27，在 [5,80] 内
    r = compute_total_reward(text, 85, cfg)
    assert r["correct"] == 1.0
    assert r["format"] == 0.2
    assert r["cot_step"] == 0.3
    assert r["length_penalty"] == 0.0
    assert abs(r["total"] - 1.5) < 1e-6
    print("[PASS] 四分量全开时总分计算正确")


def test_edge_cases():
    """边界情况：空输出、无数字、纯标签。"""
    cfg = RewardConfig()
    # 空输出
    r = compute_total_reward("", 85, cfg)
    assert r["total"] == 0.0
    # 无数字
    r = compute_total_reward("我不知道", 85, cfg)
    assert r["total"] == 0.0
    # 纯标签无内容
    r = compute_total_reward("<answer></answer>", 85, cfg)
    assert r["format"] == 0.0  # 正则要求标签内有数字
    print("[PASS] 边界情况处理正确")


def main():
    print("=" * 60)
    print("奖励函数单测（纯 CPU，无需 GPU/模型）")
    print("=" * 60)
    test_default_config_matches_original()
    test_reward_correct_factory()
    test_reward_format_factory()
    test_cot_step_reward()
    test_length_penalty()
    test_build_reward_funcs_filtering()
    test_total_reward_with_all_components()
    test_edge_cases()
    print("=" * 60)
    print("全部通过 [OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()
