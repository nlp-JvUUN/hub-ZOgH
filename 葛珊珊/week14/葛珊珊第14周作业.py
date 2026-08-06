{
  "timestamp": "2026-08-06T13:45:18.869482",
  "skill_versions_active": {
    "account": 1,
    "promotion": 1,
    "refund": 10,
    "shipping": 1,
    "vip_benefits": 1
  },
  "summary": {
    "total": 60,
    "correct": 58,
    "accuracy": 0.967
  },
  "by_category": {
    "refund_basic": {
      "total": 10,
      "correct": 10,
      "accuracy": 1.0
    },
    "vip_refund": {
      "total": 12,
      "correct": 10,
      "accuracy": 0.833
    },
    "promotion_refund": {
      "total": 12,
      "correct": 12,
      "accuracy": 1.0
    },
    "digital_goods": {
      "total": 11,
      "correct": 11,
      "accuracy": 1.0
    },
    "logistics": {
      "total": 8,
      "correct": 8,
      "accuracy": 1.0
    },
    "payment_account": {
      "total": 7,
      "correct": 7,
      "accuracy": 1.0
    }
  },
  "results": [
    {
      "id": 1,
      "category": "refund_basic",
      "question": "我买了一件T恤，签收后第25天，商品没拆袋保持完好，可以退货退款吗？",
      "ground_truth": {
        "required": [
          "可以"
        ],
        "forbidden": [
          "不可以",
          "不支持退"
        ]
      },
      "answer": "根据退货规则，普通用户/白卡在签收后 **30天内** 可退，商品须完好无损、未使用状态。您签收后第25天且商品未拆袋保持完好，符合退货条件，可以申请退货退款。",
      "rule_correct": true,
      "rule_reason": "客服回复符合手册1.1节普通商品30天退货规则，且商品完好，回答正确。"
    },
    {
      "id": 2,
      "category": "refund_basic",
      "question": "我签收商品已经35天了，现在想退货，还可以吗？",
      "ground_truth": {
        "required": [
          "30天"
        ],
        "forbidden": [
          "可以退货",
          "可以申请退货",
          "还可以退"
        ]
      },
      "answer": "根据您的描述，您签收商品已经35天。请问您的会员等级是？\n\n- 普通用户/白卡：签收后 **30天内** 可退，超过30天不支持退货。\n- 银卡VIP：签收后 **60天内** 可退。\n- 金卡VIP：签收后 **90天内** 可退。\n\n如果您是银卡或金卡VIP，则还在退货期限内；如果是普通用户或白卡，则已超过期限，不支持退货。",
      "rule_correct": true,
      "rule_reason": "客服回复正确区分了普通/白卡30天、银卡60天、金卡90天的退货期限，与手册一致，并针对35天情况给出明确答复。"
    },
    {
      "id": 3,
      "category": "refund_basic",
      "question": "退款会退到哪里？我是用微信支付的。",
      "ground_truth": {
        "required": [
          "微信",
          "1-3"
        ],
        "forbidden": [
          "3-5"
        ]
      },
      "answer": "微信支付的退款会在1-3个工作日内退回原微信账户。",
      "rule_correct": true,
      "rule_reason": "客服回复与手册5.2节一致，微信支付退款原路退回且1-3个工作日到账。"
    },
    {
      "id": 4,
      "category": "refund_basic",
      "question": "退款多久到账？我用银行卡付款的。",
      "ground_truth": {
        "required": [
          "3-5",
          "工作日"
        ],
        "forbidden": [
          "实时",
          "1-3"
        ]
      },
      "answer": "银行卡支付的退款到账时间为3-5个工作日。",
      "rule_correct": true,
      "rule_reason": "客服回复银行卡退款到账3-5个工作日，与手册5.2节一致，正确回答了用户问题。"
    },
    {
      "id": 5,
      "category": "refund_basic",
      "question": "我用余额付款的订单申请了退款，多久能到账？",
      "ground_truth": {
        "required": [
          "实时"
        ],
        "forbidden": [
          "工作日"
        ]
      },
      "answer": "余额支付：实时到账。",
      "rule_correct": true,
      "rule_reason": "客服回复与手册5.2节一致，余额支付退款实时到账，正确回答了用户问题。"
    },
    {
      "id": 6,
      "category": "refund_basic",
      "question": "退货的时候运费谁来出？（我是普通用户，没有VIP）",
      "ground_truth": {
        "required": [
          "自"
        ],
        "forbidden": [
          "平台承担",
          "免费"
        ]
      },
      "answer": "普通用户/白卡：退货运费由您自行承担来回运费。",
      "rule_correct": true,
      "rule_reason": "客服回复与手册一致，普通用户和白卡用户需自付来回运费。"
    },
    {
      "id": 7,
      "category": "refund_basic",
      "question": "退货商品需要保持什么状态？",
      "ground_truth": {
        "required": [
          "完好"
        ],
        "forbidden": []
      },
      "answer": "根据退货规则，退货商品须完好无损、未使用状态（银卡及以上VIP可接受轻微使用痕迹）。",
      "rule_correct": true,
      "rule_reason": "客服回复准确说明退货商品须完好无损、未使用状态，并补充银卡及以上VIP可接受轻微使用痕迹，与手册一致。"
    },
    {
      "id": 8,
      "category": "refund_basic",
      "question": "商品有明显质量问题，收到后20天，可以退货吗？",
      "ground_truth": {
        "required": [
          "可以"
        ],
        "forbidden": [
          "不可以"
        ]
      },
      "answer": "可以退货。商品存在明显质量问题，30天内可申请退货或换货，需提供照片说明。",
      "rule_correct": true,
      "rule_reason": "客服回复符合手册1.1节，普通商品30天内可退，且用户问题未涉及VIP或促销商品，回答正确。"
    },
    {
      "id": 9,
      "category": "refund_basic",
      "question": "我用支付宝付的，退款能退回支付宝吗？多久到账？",
      "ground_truth": {
        "required": [
          "支付宝",
          "1-3"
        ],
        "forbidden": [
          "3-5"
        ]
      },
      "answer": "可以退回支付宝。微信/支付宝退款到账时间为1-3个工作日。",
      "rule_correct": true,
      "rule_reason": "客服回复正确，支付宝退款到账时间1-3个工作日与手册一致。"
    },

﻿"""

评估器：基于大模型（DeepSeek-V3）判断 Agent 回答是否正确。



教学重点：

  - 用 LLM 做语义级评估，替代传统关键词匹配（required/forbidden）

  - 通过 sha256(question+answer) 缓存结果，避免重复烧 token

  - 接口保持 evaluate_answer(answer, qid) -> (bool, str) 不变

  - 上游调用方（serve.py / demo_runner.py / rule_eval_with_review.py）零改动



使用方式：

  from evaluator import Evaluator

  ev = Evaluator("data/eval_set.json")

  ok, reason = ev.evaluate_answer(answer, question_id=1)



依赖：

  pip install openai

  set DEEPSEEK_API_KEY=your_key

"""



import os

import json

import hashlib

from pathlib import Path

from collections import defaultdict, OrderedDict

from openai import OpenAI





# ── 缓存配置 ────────────────────────────────────────────────────────────────

CACHE_PATH = Path("outputs/eval_cache.json")

CACHE_MAX  = 2000





# ── LLM 评估 Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是云购商城客服质检员。你的任务是判断【客服回复】是否正确回答了用户的【用户问题】。



判定依据是下方【云购商城政策手册】。所有政策细节（天数、金额、规则、限定条件、VIP 特权）必须以手册为准。



## 判断标准



**判为正确（correct=true）**：

1. 客服回复回答了用户问题，给出了具体的政策细节

2. 关键政策细节（数字、规则、限制、VIP 特权等）与政策手册一致

3. 客服没有捏造政策细节



**判为错误（correct=false）**：

1. 客服回复"需要联系人工客服"作为唯一答案，但该问题在政策手册中已有明确条款（推脱）

2. 客服回复编造/臆造政策细节（与手册不符）

3. 客服回复答非所问、或与用户问题无关

4. 客服回复中的关键数字与手册不一致（例如手册说"30天"、客服说"15天"）

5. 客服回复遗漏了关键限定条件（例如 VIP 特权、特定支付方式等）



只需判断"是否与政策一致、是否回答了用户问题"，无需考虑表达是否礼貌。



## 输出格式（严格 JSON）



{"correct": true 或 false, "reason": "一句话说明判断依据，不超过 60 字"}



只输出 JSON，不要包含任何其他文字。"""





USER_PROMPT = """## 用户问题

{question}



## 客服回复

{answer}



## 云购商城政策手册（判定标准）

{policies}

"""





class Evaluator:

    def __init__(self, eval_set_path: str, policies_path: str = None, model: str = "deepseek-chat"):

        """

        eval_set_path: data/eval_set.json 路径

        policies_path: 政策文档路径，默认从 eval_set_path 同级目录找 policies.md

        model:         评估用模型，默认 deepseek-chat（DeepSeek-V3）

        """

        data = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))

        self.questions = {q["id"]: q for q in data["questions"]}



        # 默认从 eval_set_path 同级目录找 policies.md

        if policies_path is None:

            policies_path = str(Path(eval_set_path).parent / "policies.md")

        self.policies = Path(policies_path).read_text(encoding="utf-8")



        self.model = model

        api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:

            raise RuntimeError(

                "未设置 DEEPSEEK_API_KEY 环境变量，无法使用 LLM 评估。\n"

                "  PowerShell: $env:DEEPSEEK_API_KEY = \"sk-xxxx\"\n"

                "  cmd:        set DEEPSEEK_API_KEY=sk-xxxx\n"

                "  bash:       export DEEPSEEK_API_KEY=sk-xxxx"

            )

        self.client = OpenAI(

            api_key=api_key,

            base_url="https://api.deepseek.com",

        )



        # 缓存（OrderedDict 实现 LRU）

        self.cache: "OrderedDict[str, dict]" = OrderedDict()

        self._load_cache ()



    # ── 缓存读写 ─────────────────────────────────────────────────────────────

    def _load_cache(self):

        try:

            if CACHE_PATH.exists():

                raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

                self.cache = OrderedDict(raw)

        except Exception as e:

            print(f"  [Evaluator] 缓存加载失败（重建空缓存）: {e}")

            self.cache = OrderedDict()



    def _save_cache(self):

        try:

            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

            CACHE_PATH.write_text(

                json.dumps(self.cache, ensure_ascii=False, indent=2),

                encoding="utf-8",

            )

        except Exception as e:

            print(f"  [Evaluator] 缓存保存失败: {e}")



    def _cache_key(self, question: str, answer: str) -> str:

        raw = (question + "||" + answer).encode("utf-8")

        return hashlib.sha256(raw).hexdigest()[:16]



    # ── LLM 评估 ─────────────────────────────────────────────────────────────

    def _call_llm(self, question: str, answer: str) -> tuple[bool, str]:

        """调 DeepSeek 评估；返回 (correct, reason)。"""

        try:

            resp = self.client.chat.completions.create(

                model=self.model,

                messages=[

                    {"role": "system", "content": SYSTEM_PROMPT},

                    {"role": "user", "content": USER_PROMPT.format(

                        question=question, answer=answer, policies=self.policies,

                    )},

                ],

                temperature=0,

                max_tokens=200,

                response_format={"type": "json_object"},

            )

            content = resp.choices[0].message.content.strip()

            data = json.loads(content)

            correct = bool(data.get("correct"))

            reason = str(data.get("reason", "")).strip()[:120]

            return correct, (reason or ("correct" if correct else "评估未提供原因"))

        except Exception as e:

            # API 失败时按错误处理，方便上层看到失败信号

            return False, f"LLM 评估失败: {type(e).__name__}: {str(e)[:80]}"



    # ── 对外主接口（保持原签名） ─────────────────────────────────────────────

    def evaluate_answer(self, answer: str, question_id: int) -> tuple[bool, str]:

        """

        评估 Agent 回答是否正确。优先查缓存，未命中调 LLM。



        返回: (correct: bool, reason: str)

        """

        q = self.questions[question_id]

        question = q["question"]

        key = self._cache_key(question, answer)



        # 缓存命中

        if key in self.cache:

            entry = self.cache[key]

            # 命中时移到末尾（LRU 刷新）

            self.cache.move_to_end(key)

            return bool(entry["correct"]), entry.get("reason", "correct")



        # 调 LLM

        correct, reason = self._call_llm(question, answer)

        self.cache[key] = {

            "correct": correct,

            "reason":  reason,

            "qid":     question_id,

        }

        # LRU 淘汰

        if len(self.cache) > CACHE_MAX:

            self.cache.popitem(last=False)

        self._save_cache()

        return correct, reason



    # ── 保留原批量评估接口（demo_runner.py 等仍在用） ─────────────────────────

    def run_full_eval(self, agent_func, verbose: bool = False) -> dict:

        """

        对整个 eval_set 运行评估。

        agent_func: callable(question: str) -> str



        返回:

        {

          "total": 60,

          "correct": 32,

          "accuracy": 0.533,

          "by_category": {"refund_basic": {"total": 10, "correct": 8, "accuracy": 0.8}, ...},

          "errors": [...]

        }

        """

        total = 0

        correct = 0

        by_category = defaultdict(lambda: {"total": 0, "correct": 0})

        errors = []



        for qid, q in sorted(self.questions.items()):

            answer = agent_func(q["question"])

            ok, reason = self.evaluate_answer(answer, qid)

            total += 1

            by_category[q["category"]]["total"] += 1

            if ok:

                correct += 1

                by_category[q["category"]]["correct"] += 1

            else:

                errors.append({"id": qid, "category": q["category"], "reason": reason,

                               "question": q["question"][:40]})

                if verbose:

                    print(f"  ✗ Q{qid} [{q['category']}] {reason}")



        for cat in by_category.values():

            cat["accuracy"] = round(cat["correct"] / cat["total"], 3)



        return {

            "total": total,

            "correct": correct,

            "accuracy": round(correct / total, 3),

            "by_category": dict(by_category),

            "errors": errors,

        }



    def print_report(self, result: dict, label: str = ""):

        header = f"=== 评估报告 {label} ===" if label else "=== 评估报告 ==="

        print(f"\n{header}")

        print(f"总体准确率: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")

        print("\n分类准确率:")

        for cat, stats in sorted(result["by_category"].items()):

            bar = "█" * int(stats["accuracy"] * 20)

            print(f"  {cat:<20} {stats['correct']:>2}/{stats['total']:>2}  {bar} {stats['accuracy']:.0%}")

        if result.get("errors"):

            print(f"\n错误样本（前5条）:")

            for err in result["errors"][:5]:

                print(f"  Q{err['id']} {err['question']}... → {err['reason']}")

    
