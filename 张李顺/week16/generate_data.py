import json
import random
from pathlib import Path


random.seed(16)
used_questions = set()


def make_question(operator):
    while True:
        low = 2 if operator == "*" else 0
        high = 99 if operator == "*" else 999
        left = random.randint(low, high)
        right = random.randint(low, high)
        question = f"{left} {operator} {right}"
        if question not in used_questions:
            used_questions.add(question)
            break

    if operator == "+":
        answer = left + right
    elif operator == "-":
        answer = left - right
    else:
        answer = left * right

    return {
        "question": question,
        "answer": str(answer),
        "target": f"<hahaha>{answer}<gagaga>",
    }


train_data = []
eval_data = []

for operator in ("+", "-", "*"):
    questions = [make_question(operator) for _ in range(100)]
    train_data.extend(questions[:80])
    eval_data.extend(questions[80:])

random.shuffle(train_data)
random.shuffle(eval_data)

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)
(data_dir / "train.json").write_text(
    json.dumps(train_data, ensure_ascii=False, indent=2), encoding="utf-8"
)
(data_dir / "eval.json").write_text(
    json.dumps(eval_data, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(f"训练集：{len(train_data)} 题")
print(f"评估集：{len(eval_data)} 题")
