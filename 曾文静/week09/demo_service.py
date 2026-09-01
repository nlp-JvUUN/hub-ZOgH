"""
通过 vLLM 服务调用大模型（OpenAI 兼容 API）

三步验证"部署的服务能干活"：
  1. 列出服务已加载的模型
  2. 普通对话（chat completions）
  3. guided_json 约束解码：金融意图抽取，输出强制符合 JSON Schema，并用 jsonschema 校验

用法（需先启动服务）：
  bash start_server.sh          # 终端 1
  python demo_service.py        # 终端 2
"""

import json

from openai import OpenAI
from jsonschema import validate

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL = "qwen2-0.5b"

# 金融意图抽取 Schema：字段值全部被约束（枚举/正则/必填）
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string",
                   "enum": ["stock_query", "report_query", "news_query", "compare", "other"]},
        "stock_code": {"type": "string", "pattern": r"^\d{6}$"},
        "field": {"type": "string",
                  "enum": ["open", "close", "high", "low", "volume"]},
    },
    "required": ["intent"],
    "additionalProperties": False,
}

SYSTEM = ("你是金融问答意图识别器。根据用户问题输出纯 JSON："
          "{\"intent\": 意图, \"stock_code\": 6位代码, \"field\": 查询字段}。"
          "intent ∈ [stock_query, report_query, news_query, compare, other]；"
          "field ∈ [open, close, high, low, volume]。不要输出解释。")

TEST_CASES = [
    "查一下 600519 贵州茅台的收盘价",
    "宁德时代今天最高价是多少",
    "帮我看看宁德时代的最新财报",
    "对比一下茅台和五粮液今年的表现",
]


def main():
    # 1. 服务可用性
    models = client.models.list()
    print("已连接 vLLM 服务，可用模型：", [m.id for m in models.data])

    # 2. 普通对话
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用一句话介绍你自己"}],
        max_tokens=50, temperature=0,
    )
    print("\n[普通对话]", resp.choices[0].message.content.strip())

    # 3. guided_json 约束解码
    print(f"\n[guided_json 意图抽取] {len(TEST_CASES)} 条测试")
    ok = 0
    for q in TEST_CASES:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": q}],
            max_tokens=120, temperature=0,
            extra_body={"guided_json": INTENT_SCHEMA},   # vLLM 扩展字段
        )
        raw = resp.choices[0].message.content.strip()
        try:
            obj = json.loads(raw)
            validate(instance=obj, schema=INTENT_SCHEMA)
            ok += 1
            print(f"  ✓ {q}  →  {raw}")
        except Exception as e:
            print(f"  ✗ {q}  →  {raw}  ({e})")

    print(f"\nSchema 完全通过率：{ok}/{len(TEST_CASES)} (100% 则说明约束解码生效)")


if __name__ == "__main__":
    main()
