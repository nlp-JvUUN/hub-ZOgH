"""
通过 vLLM 服务进行文本分类（与 classify_llm.py 功能相同，但通过 HTTP API 调用）

对比视角：
  1. 本地推理（classify_llm.py）：直接加载模型到内存，适合单进程低并发场景
  2. vLLM 服务（本脚本）：通过 HTTP API 调用，支持多客户端并发，资源集中管理

使用方式：
  # 先启动 vLLM 服务
  python server_vllm.py --model_path ../../pretrain_models/Qwen2-0.5B-Instruct

  # 在另一个终端运行客户端
  python classify_vllm.py                    # 默认连接 http://localhost:8000/v1
  python classify_vllm.py --base_url http://192.168.1.100:8000/v1  # 远程服务
  python classify_vllm.py --num_samples 500  # 采样 500 条
  python classify_vllm.py --demo             # 只跑 5 条示例

依赖：
  pip install openai>=1.0.0
"""

import argparse
import json
import random
import time
import sys
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"

LABEL_NAMES = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]

SYSTEM_PROMPT = (
    "你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
    "只输出类别名称，不要输出任何其他内容。\n"
    "可选类别：" + "、".join(LABEL_NAMES)
)


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM 服务客户端 - 文本分类")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1",
                        help="vLLM 服务地址，默认 http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="sk-xxx",
                        help="API 密钥，vLLM 本地服务可任意填写")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称；不指定则使用服务端返回的第一个模型")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--num_samples", type=int, default=200,
                        help="从验证集随机采样的样本数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true",
                        help="只跑 5 条示例（快速演示）")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="温度参数，0.0 为确定性输出")
    parser.add_argument("--max_new_tokens", type=int, default=8,
                        help="最大生成 token 数")
    return parser.parse_args()


def build_messages(text: str) -> list:
    """构建 chat completion 的 messages 格式"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"新闻标题：{text}\n类别："},
    ]


def parse_prediction(raw_output: str) -> str | None:
    """从模型输出中提取类别名，支持模糊匹配"""
    if not raw_output:
        return None
    for name in LABEL_NAMES:
        if name in raw_output:
            return name
    return None


def main():
    if not OPENAI_AVAILABLE:
        print("[错误] 无法导入 openai 库")
        print("请安装 openai: pip install openai>=1.0.0")
        sys.exit(1)

    args = parse_args()

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    try:
        models = client.models.list()
        model_list = [m.id for m in models.data]
        if not model_list:
            print("[错误] 服务端未返回可用模型")
            sys.exit(1)
        print(f"已连接到 vLLM 服务: {args.base_url}")
        print(f"可用模型: {', '.join(model_list)}")
        if args.model:
            if args.model not in model_list:
                print(f"[警告] 指定的模型 '{args.model}' 不在可用列表中，将使用默认模型")
                args.model = model_list[0]
            print(f"使用模型: {args.model}")
        else:
            args.model = model_list[0]
            print(f"使用默认模型: {args.model}")
    except Exception as e:
        print(f"[错误] 无法连接到 vLLM 服务: {e}")
        print("请先启动服务：python server_vllm.py --model_path <模型路径>")
        print(f"检查服务是否运行在: {args.base_url}")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[错误] 数据目录不存在: {data_dir}")
        sys.exit(1)

    try:
        with open(data_dir / "val.json", encoding="utf-8") as f:
            val_data = json.load(f)
        with open(data_dir / "label_map.json", encoding="utf-8") as f:
            label_map = json.load(f)
    except Exception as e:
        print(f"[错误] 加载数据失败: {e}")
        sys.exit(1)

    id2name = {int(k): v for k, v in label_map["id2name"].items()}

    random.seed(args.seed)
    n = 5 if args.demo else args.num_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"\n评估样本数: {len(samples)}")

    correct, total, unparseable = 0, 0, 0
    results = []
    t0 = time.time()

    for i, item in enumerate(samples):
        text     = item["sentence"]
        true_id  = item["label"]
        true_name = id2name[true_id]

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=build_messages(text),
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                stop=["<|im_end|>", "\n"],
            )
            raw_output = response.choices[0].message.content.strip()
            pred_name  = parse_prediction(raw_output)
        except Exception as e:
            print(f"[警告] 请求失败: {e}")
            raw_output = ""
            pred_name  = None

        is_correct = (pred_name == true_name)
        if pred_name is None:
            unparseable += 1
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "text":        text,
            "true_label":  true_name,
            "pred_label":  pred_name,
            "raw_output":  raw_output,
            "correct":     is_correct,
        })

        status = "✓" if is_correct else ("?" if pred_name is None else "✗")
        print(f"[{i+1:3d}/{len(samples)}] {status} "
              f"真实:{true_name:4s} 预测:{str(pred_name):4s} | {text[:35]}")

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"vLLM 服务分类结果")
    print(f"{'='*60}")
    print(f"  服务地址 : {args.base_url}")
    print(f"  模型     : {args.model}")
    print(f"  样本数   : {total}")
    print(f"  准确率   : {correct}/{total} = {acc:.4f}")
    print(f"  无法解析 : {unparseable} 条 ({unparseable/total*100:.1f}%)")
    print(f"  总耗时   : {elapsed:.1f}s, 均值 {elapsed/total:.2f}s/条")

    print(f"""
对比参考：
  本地推理 (classify_llm.py)       : 适合单进程、低延迟场景
  vLLM 服务 (本脚本)               : 适合多客户端并发、资源集中管理

性能特点：
  vLLM 使用 PagedAttention + Continuous Batching
  - 吞吐量比本地 transformers 高 2~3 倍
  - 支持动态批处理，多个请求自动合并
  - 显存利用率更高，可服务更多并发用户
""")

    try:
        out_path = ROOT / "outputs" / "llm_vllm_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "accuracy": acc, "total": total, "correct": correct,
                "unparseable": unparseable, "results": results,
                "base_url": args.base_url, "model": args.model,
            }, f, ensure_ascii=False, indent=2)
        print(f"结果已保存 → {out_path}")
    except Exception as e:
        print(f"[警告] 保存结果失败: {e}")


if __name__ == "__main__":
    main()
clas
