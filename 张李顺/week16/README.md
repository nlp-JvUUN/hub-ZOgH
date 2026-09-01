# Qwen 数学 GRPO

这个项目只做三件事：训练前评估、最多四分钟 GRPO 训练、训练后评估。
提示词只包含一个格式示例。

## 运行

在 VS Code 中选择这个 Python 解释器：

```text
D:\mySoftwares\Anaconda\envs\ai_learning_1\python.exe
```

然后在 VS Code 终端执行：

```text
python train.py
```

使用的是已有环境和本地 Qwen2-0.5B-Instruct，不会下载模型或安装依赖。
GRPO 只保留组内相对奖励和裁剪更新；为节省显存，不加载参考模型。

## 奖励

- 数值答案正确：1分。
- 输出严格符合 `<hahaha>整数<gagaga>`：0.5分。
- 两项都正确：1.5分。
- 两项都错误：0分。

数值判断取输出中的最后一个整数。评估时要求整个输出与目标完全相同，才计为正确。

## 输出

- `output/before.json`：训练前的逐题回答。
- `output/after.json`：训练后的逐题回答。
- `output/result.json`：训练前后的汇总结果。
- `output/qwen_math_lora/`：训练得到的 LoRA 权重。

训练循环限制为240秒，模型加载和训练前后评估不计入训练时间。
程序结束时会打印数值正确、格式正确和严格正确三项训练前后对比。
