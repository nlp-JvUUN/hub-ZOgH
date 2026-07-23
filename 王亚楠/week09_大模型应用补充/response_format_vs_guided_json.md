# response_format vs guided_json：底层实现原理

## 一句话总结

`response_format` 像只检查标点符号的校对员，`guided_json` 像拿着需求文档逐字核对的审核员。两者底层机制完全不同，**只有 `guided_json` 走了 FSM**。

---

## 1. 背景：为什么需要约束 JSON 输出？

在 Function Call / Agent 场景中，LLM 需要输出结构化的 JSON 供下游代码解析。但小模型（如 Qwen2-0.5B）经常"犯规"：

- **JSON 语法错**：多逗号、少引号、花括号不配对
- **字段名拼错或缺失**：漏掉必选字段、字段名打错
- **字段值不符合约束**：枚举值超出范围、正则不匹配、类型错误

两种约束方式 `response_format` 和 `guided_json` 就是为解决这些问题而设计的，但**实现原理和约束力完全不同**。

---

## 2. response_format: {"type": "json_object"} 的底层实现

### 2.1 CFG 语法约束（vLLM 路线）

vLLM 内部使用 **XGrammar**（或旧版 outlines）来实现。它构建的是一个**极端宽松的 JSON 语法**，是一套**上下文无关文法（CFG，Context-Free Grammar）**：

```
value  → object | array | string | number | true | false | null
object → "{" (string ":" value ("," string ":" value)*)? "}"
array  → "[" (value ("," value)*)? "]"
...
```

这个 CFG 只保证：
- 花括号 `{}` 和方括号 `[]` 正确配对
- 引号正确闭合
- 冒号/逗号位置正确
- 值的基本类型合法（字符串、数字、布尔等）

但它**完全不知道你的业务 schema**——不知道有 `symbol` 字段，也不知道 `market` 只能是 `SH`。

### 2.2 Token 前缀注入（OpenAI 路线）

OpenAI 的做法更简单粗暴：

1. 检测到 `response_format: {"type": "json_object"}` 时，强制把 assistant 首条消息的前缀设为 `{`
2. 在 system prompt 末尾追加 "You must output valid JSON."

本质上是给了一个看得见的"起跑线"，其余全靠模型自觉。**没有涉及任何 FSM**。

### 2.3 约束力边界

```
          response_format               guided_json
          ──────────────                ──────────
JSON 语法     ✓                          ✓
字段名约束    ✗ 模型自己猜                 ✓ 来自 schema
枚举约束      ✗ 模型自己猜                 ✓ 来自 schema
正则约束      ✗ 模型自己猜                 ✓ 编译进 FSM
数值范围      ✗ 基本不管                   ✓ 严格检查
```

---

## 3. guided_json 的底层实现：FSM 字符级约束

### 3.1 编译流程

`guided_json` 将 JSON Schema **编译成字符级有限状态机（FSM）**，完整流程：

```
JSON Schema
    ↓ ① 编译
正则语法 (Regular Grammar)
    ↓ ② 展开成状态图
FSM — 每个状态 = "当前在 JSON 结构的哪个位置"
    ↓ ③ 运行时应用
每步生成 token 时，查 FSM 算出合法 token 集合
把不合法的 token logit 设为 -∞
```

### 3.2 枚举约束如何工作？

给定 schema：

```json
{"market": {"enum": ["SH", "SZ", "BJ"]}}
```

FSM 走到 `"market":` 后面的状态时：

```
当前状态: 刚读完 "market":"
            ↓
          必须输出 "    ← 只有这一个合法字符
            ↓
          分支点: S | B  ← FSM 的分叉（SZ 和 SH 都从 S 开始，BJ 从 B 开始）
            ↓ (选了 S)
          分支点: Z | H  ← SZ 和 SH 在此分叉
            ↓ (选了 H)
          必须是 "        ← 收尾
```

模型在这个状态空间的每一步，`SH`/`SZ`/`BJ` 之外的 token **全部被物理屏蔽**（logit = -∞），模型根本没机会犯错。

### 3.3 正则约束如何工作？

```json
{"symbol": {"pattern": "^\\d{6}$"}}
```

6 位数字在 FSM 里展开为 7 个状态：

```
状态0 ─→ 状态1 ─→ 状态2 ─→ ... ─→ 状态6 ─→ 状态7(结束)
       输出第1位     输出第2位          输出第6位    必须是结束
       数字          数字               数字
```

`[0-9]` 是一个字符集，FSM 知道"当前位置，只有 `0`~`9` 这 10 个字符合法"，其余全部屏蔽。

### 3.4 数值范围如何工作？

```json
{"quantity": {"minimum": 1, "maximum": 100}}
```

FSM 在数字位置展开成约束：

- 首位只能是 `1`~`9`（不能是 `0`，因为 min=1）
- 如果首位是 `1`，下一位可以是 `0`~`0`（即 `10`）或结束；如果是 `2`~`9`，则只能结束（即 `2`~`9`）
- 100 的情况：只有 `1` → `0` → `0` 这一条路径

---

## 4. 直观对比

```
你的 schema:
  symbol 必须是 6 位数字
  market 只能是 SH/SZ/BJ

response_format 的视角:
  "输出合法 JSON 就行，里面有什么字段我不关心"
  → 可能输出 {"symbol": "贵州茅台", "market": "上海证券"}  ← 合法 JSON，但完全不对

guided_json 的视角:
  "第 14 个 token 必须来自集合 {S, B}"
  "第 15 个 token 取决于第 14 个选了啥"
  → 被迫输出 {"symbol": "600519", "market": "SH"}  ← FSM 逼的，不可能错
```

---

## 5. 技术对比总结

| 维度 | response_format | guided_json |
|---|---|---|
| **底层技术** | CFG（XGrammar）/ 前缀注入 | **FSM**（编译 JSON Schema） |
| **约束层级** | JSON 语法结构 | 字符级 |
| **知道业务 schema？** | 不知道 | 全部知道 |
| **可移植性** | OpenAI 标准，多厂商兼容 | vLLM 私有扩展 |
| **JSON 合法率** | ~95% | 100% |
| **Schema 完全通过率** | ~60%（小模型） | **100%** |
| **适用场景** | 大模型 / 多厂商切换 | 小模型 / 严格下游解析 |

---

## 6. 选型建议

```
裸 prompt 指令        → 大模型 + 容错要求低的场景
response_format       → 需要跨平台可移植 + JSON 合法率要求高
guided_json           → 小模型 + 下游严格解析（生产环境 Agent 系统首选）
```

## 7. 相关 Demo

| 文件 | 演示内容 |
|---|---|
| [demo_response_format.py](../src/demo_response_format.py) | 裸 prompt vs response_format 效果对比 |
| [demo_function_call.py](../src/demo_function_call.py) | 裸 prompt vs response_format vs guided_json 三种模式全面对比 |
