 环境：AutoDL 云 GPU / RTX 4090D 24GB / vLLM 0.9.2 / Qwen2-0.5B-Instruct

  【速度验证】
  - transformers 串行：79.14s（0.63 QPS）
  - transformers batch=8：14.89s（3.36 QPS）
  - vLLM continuous batching：1.64s（30.40 QPS）
  - vLLM 相对串行加速：48.1×
  - vLLM 相对 batch=8 加速：9.1×

  【约束解码可靠性验证】
  - 裸 prompt：60% schema 通过率（42% 失败在字段语义错误）
  - response_format：68% schema 通过率（JSON 合法率 100%，但字段仍错）
  - guided_json：100% schema 通过率（FSM 解码层强制约束）

  结论：vLLM 将 50 并发请求从 79 秒压缩到 1.6 秒；guided_json 将 Function Call 可靠性从 60% 提升到 100%。