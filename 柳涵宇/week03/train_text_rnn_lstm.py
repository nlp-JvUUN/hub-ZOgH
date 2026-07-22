import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


def zeros(n):
    return [0.0 for _ in range(n)]


def zeros_mat(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def rand_mat(rows, cols, scale):
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def dot_mv(mat, vec):
    return [sum(row[j] * vec[j] for j in range(len(vec))) for row in mat]


def add_to(a, b):
    for i in range(len(a)):
        a[i] += b[i]


def vec_add3(a, b, c):
    return [a[i] + b[i] + c[i] for i in range(len(a))]


def tanh_vec(v):
    return [math.tanh(x) for x in v]


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid_vec(v):
    return [sigmoid(x) for x in v]


def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


def outer_add(grad, a, b):
    for i in range(len(a)):
        ai = a[i]
        row = grad[i]
        for j in range(len(b)):
            row[j] += ai * b[j]


def mat_t_vec(mat, vec):
    cols = len(mat[0])
    out = [0.0 for _ in range(cols)]
    for i, row in enumerate(mat):
        vi = vec[i]
        for j in range(cols):
            out[j] += row[j] * vi
    return out


def clip_grads(grads, max_norm):
    total = 0.0
    for g in grads:
        if isinstance(g[0], list):
            for row in g:
                for x in row:
                    total += x * x
        else:
            for x in g:
                total += x * x
    norm = math.sqrt(total)
    if norm <= max_norm or norm == 0.0:
        return norm
    scale = max_norm / norm
    for g in grads:
        if isinstance(g[0], list):
            for row in g:
                for j in range(len(row)):
                    row[j] *= scale
        else:
            for j in range(len(g)):
                g[j] *= scale
    return norm


def sgd_update(param, grad, lr):
    if isinstance(param[0], list):
        for i in range(len(param)):
            for j in range(len(param[i])):
                param[i][j] -= lr * grad[i][j]
    else:
        for i in range(len(param)):
            param[i] -= lr * grad[i]


def make_dataset(size, vocab, seq_len=5):
    chars = [ch for ch in vocab if ch != "你"]
    data = []
    for _ in range(size):
        label = random.randrange(seq_len)
        seq = [random.choice(chars) for _ in range(seq_len)]
        seq[label] = "你"
        data.append(("".join(seq), label))
    return data


def encode_dataset(data, stoi):
    return [([stoi[ch] for ch in text], label, text) for text, label in data]


def accuracy(model, data):
    correct = 0
    for x, y, _ in data:
        pred = model.predict(x)
        correct += int(pred == y)
    return correct / len(data)


@dataclass
class TrainResult:
    model: str
    epoch: int
    loss: float
    train_acc: float
    test_acc: float


class SimpleRNN:
    def __init__(self, vocab_size, emb_dim, hidden_dim, class_count):
        scale = 0.25
        self.E = rand_mat(vocab_size, emb_dim, scale)
        self.Wxh = rand_mat(hidden_dim, emb_dim, scale)
        self.Whh = rand_mat(hidden_dim, hidden_dim, scale)
        self.bh = zeros(hidden_dim)
        self.Why = rand_mat(class_count, hidden_dim, scale)
        self.by = zeros(class_count)

    def forward(self, x_ids):
        hs = [zeros(len(self.bh))]
        xs = []
        for token_id in x_ids:
            x = self.E[token_id][:]
            xs.append(x)
            h = tanh_vec(vec_add3(dot_mv(self.Wxh, x), dot_mv(self.Whh, hs[-1]), self.bh))
            hs.append(h)
        logits = vec_add3(dot_mv(self.Why, hs[-1]), self.by, zeros(len(self.by)))
        return logits, xs, hs

    def loss_and_grads(self, x_ids, y):
        logits, xs, hs = self.forward(x_ids)
        probs = softmax(logits)
        loss = -math.log(probs[y] + 1e-12)
        dlogits = probs[:]
        dlogits[y] -= 1.0

        dE = zeros_mat(len(self.E), len(self.E[0]))
        dWxh = zeros_mat(len(self.Wxh), len(self.Wxh[0]))
        dWhh = zeros_mat(len(self.Whh), len(self.Whh[0]))
        dbh = zeros(len(self.bh))
        dWhy = zeros_mat(len(self.Why), len(self.Why[0]))
        dby = dlogits[:]

        outer_add(dWhy, dlogits, hs[-1])
        dh = mat_t_vec(self.Why, dlogits)

        for t in reversed(range(len(x_ids))):
            h = hs[t + 1]
            h_prev = hs[t]
            dtanh = [dh[i] * (1.0 - h[i] * h[i]) for i in range(len(h))]
            add_to(dbh, dtanh)
            outer_add(dWxh, dtanh, xs[t])
            outer_add(dWhh, dtanh, h_prev)
            dx = mat_t_vec(self.Wxh, dtanh)
            token_id = x_ids[t]
            for j in range(len(dx)):
                dE[token_id][j] += dx[j]
            dh = mat_t_vec(self.Whh, dtanh)

        grads = [dE, dWxh, dWhh, dbh, dWhy, dby]
        return loss, grads

    def update(self, grads, lr):
        params = [self.E, self.Wxh, self.Whh, self.bh, self.Why, self.by]
        for p, g in zip(params, grads):
            sgd_update(p, g, lr)

    def predict(self, x_ids):
        logits, _, _ = self.forward(x_ids)
        return max(range(len(logits)), key=lambda i: logits[i])


class LSTM:
    def __init__(self, vocab_size, emb_dim, hidden_dim, class_count):
        scale = 0.25
        self.E = rand_mat(vocab_size, emb_dim, scale)
        self.Wxi = rand_mat(hidden_dim, emb_dim, scale)
        self.Whi = rand_mat(hidden_dim, hidden_dim, scale)
        self.bi = zeros(hidden_dim)
        self.Wxf = rand_mat(hidden_dim, emb_dim, scale)
        self.Whf = rand_mat(hidden_dim, hidden_dim, scale)
        self.bf = [1.0 for _ in range(hidden_dim)]
        self.Wxo = rand_mat(hidden_dim, emb_dim, scale)
        self.Who = rand_mat(hidden_dim, hidden_dim, scale)
        self.bo = zeros(hidden_dim)
        self.Wxg = rand_mat(hidden_dim, emb_dim, scale)
        self.Whg = rand_mat(hidden_dim, hidden_dim, scale)
        self.bg = zeros(hidden_dim)
        self.Why = rand_mat(class_count, hidden_dim, scale)
        self.by = zeros(class_count)

    def forward(self, x_ids):
        h = zeros(len(self.bi))
        c = zeros(len(self.bi))
        caches = []
        for token_id in x_ids:
            x = self.E[token_id][:]
            h_prev = h[:]
            c_prev = c[:]
            i = sigmoid_vec(vec_add3(dot_mv(self.Wxi, x), dot_mv(self.Whi, h_prev), self.bi))
            f = sigmoid_vec(vec_add3(dot_mv(self.Wxf, x), dot_mv(self.Whf, h_prev), self.bf))
            o = sigmoid_vec(vec_add3(dot_mv(self.Wxo, x), dot_mv(self.Who, h_prev), self.bo))
            g = tanh_vec(vec_add3(dot_mv(self.Wxg, x), dot_mv(self.Whg, h_prev), self.bg))
            c = [f[k] * c_prev[k] + i[k] * g[k] for k in range(len(c))]
            tanh_c = tanh_vec(c)
            h = [o[k] * tanh_c[k] for k in range(len(h))]
            caches.append((token_id, x, h_prev, c_prev, i, f, o, g, c, tanh_c, h))
        logits = vec_add3(dot_mv(self.Why, h), self.by, zeros(len(self.by)))
        return logits, caches

    def loss_and_grads(self, x_ids, y):
        logits, caches = self.forward(x_ids)
        probs = softmax(logits)
        loss = -math.log(probs[y] + 1e-12)
        dlogits = probs[:]
        dlogits[y] -= 1.0

        hidden_dim = len(self.bi)
        emb_dim = len(self.E[0])
        dE = zeros_mat(len(self.E), emb_dim)
        dWxi = zeros_mat(hidden_dim, emb_dim)
        dWhi = zeros_mat(hidden_dim, hidden_dim)
        dbi = zeros(hidden_dim)
        dWxf = zeros_mat(hidden_dim, emb_dim)
        dWhf = zeros_mat(hidden_dim, hidden_dim)
        dbf = zeros(hidden_dim)
        dWxo = zeros_mat(hidden_dim, emb_dim)
        dWho = zeros_mat(hidden_dim, hidden_dim)
        dbo = zeros(hidden_dim)
        dWxg = zeros_mat(hidden_dim, emb_dim)
        dWhg = zeros_mat(hidden_dim, hidden_dim)
        dbg = zeros(hidden_dim)
        dWhy = zeros_mat(len(self.Why), len(self.Why[0]))
        dby = dlogits[:]

        outer_add(dWhy, dlogits, caches[-1][-1])
        dh_next = mat_t_vec(self.Why, dlogits)
        dc_next = zeros(hidden_dim)

        for cache in reversed(caches):
            token_id, x, h_prev, c_prev, i, f, o, g, c, tanh_c, _h = cache
            do = [dh_next[k] * tanh_c[k] for k in range(hidden_dim)]
            dc = [
                dh_next[k] * o[k] * (1.0 - tanh_c[k] * tanh_c[k]) + dc_next[k]
                for k in range(hidden_dim)
            ]
            df = [dc[k] * c_prev[k] for k in range(hidden_dim)]
            di = [dc[k] * g[k] for k in range(hidden_dim)]
            dg = [dc[k] * i[k] for k in range(hidden_dim)]

            dai = [di[k] * i[k] * (1.0 - i[k]) for k in range(hidden_dim)]
            daf = [df[k] * f[k] * (1.0 - f[k]) for k in range(hidden_dim)]
            dao = [do[k] * o[k] * (1.0 - o[k]) for k in range(hidden_dim)]
            dag = [dg[k] * (1.0 - g[k] * g[k]) for k in range(hidden_dim)]

            outer_add(dWxi, dai, x)
            outer_add(dWhi, dai, h_prev)
            add_to(dbi, dai)
            outer_add(dWxf, daf, x)
            outer_add(dWhf, daf, h_prev)
            add_to(dbf, daf)
            outer_add(dWxo, dao, x)
            outer_add(dWho, dao, h_prev)
            add_to(dbo, dao)
            outer_add(dWxg, dag, x)
            outer_add(dWhg, dag, h_prev)
            add_to(dbg, dag)

            dxi = mat_t_vec(self.Wxi, dai)
            dxf = mat_t_vec(self.Wxf, daf)
            dxo = mat_t_vec(self.Wxo, dao)
            dxg = mat_t_vec(self.Wxg, dag)
            for j in range(emb_dim):
                dE[token_id][j] += dxi[j] + dxf[j] + dxo[j] + dxg[j]

            dhi = mat_t_vec(self.Whi, dai)
            dhf = mat_t_vec(self.Whf, daf)
            dho = mat_t_vec(self.Who, dao)
            dhg = mat_t_vec(self.Whg, dag)
            dh_next = [dhi[k] + dhf[k] + dho[k] + dhg[k] for k in range(hidden_dim)]
            dc_next = [dc[k] * f[k] for k in range(hidden_dim)]

        grads = [
            dE,
            dWxi,
            dWhi,
            dbi,
            dWxf,
            dWhf,
            dbf,
            dWxo,
            dWho,
            dbo,
            dWxg,
            dWhg,
            dbg,
            dWhy,
            dby,
        ]
        return loss, grads

    def update(self, grads, lr):
        params = [
            self.E,
            self.Wxi,
            self.Whi,
            self.bi,
            self.Wxf,
            self.Whf,
            self.bf,
            self.Wxo,
            self.Who,
            self.bo,
            self.Wxg,
            self.Whg,
            self.bg,
            self.Why,
            self.by,
        ]
        for p, g in zip(params, grads):
            sgd_update(p, g, lr)

    def predict(self, x_ids):
        logits, _ = self.forward(x_ids)
        return max(range(len(logits)), key=lambda i: logits[i])


def train_model(name, model, train_data, test_data, epochs, lr):
    history = []
    for epoch in range(1, epochs + 1):
        random.shuffle(train_data)
        total_loss = 0.0
        for x, y, _ in train_data:
            loss, grads = model.loss_and_grads(x, y)
            clip_grads(grads, 5.0)
            model.update(grads, lr)
            total_loss += loss
        avg_loss = total_loss / len(train_data)
        train_acc = accuracy(model, train_data)
        test_acc = accuracy(model, test_data)
        history.append(TrainResult(name, epoch, avg_loss, train_acc, test_acc))
        print(
            f"{name:4s} epoch {epoch:02d} | "
            f"loss={avg_loss:.4f} train_acc={train_acc:.3f} test_acc={test_acc:.3f}"
        )
    return history


def demo_predictions(model, encoded, count=10):
    samples = random.sample(encoded, min(count, len(encoded)))
    rows = []
    for x, y, text in samples:
        pred = model.predict(x)
        rows.append({"text": text, "true_class": y + 1, "pred_class": pred + 1})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train pure-Python RNN/LSTM text classifiers.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    random.seed(args.seed)
    vocab_chars = list("你我他她它们的是在有和中学天气好坏爱看听说走跑吃喝玩猫狗山水火木金土日月星")
    stoi = {ch: idx for idx, ch in enumerate(vocab_chars)}
    train_raw = make_dataset(args.train_size, vocab_chars)
    test_raw = make_dataset(args.test_size, vocab_chars)
    train_data = encode_dataset(train_raw, stoi)
    test_data = encode_dataset(test_raw, stoi)

    print("Task: five-character text classification.")
    print("Label: the 1-based position of the character '你'.")
    print("Example:", train_raw[0][0], "class", train_raw[0][1] + 1)
    print()

    rnn = SimpleRNN(vocab_size=len(vocab_chars), emb_dim=8, hidden_dim=16, class_count=5)
    rnn_history = train_model("RNN", rnn, train_data[:], test_data, args.epochs, lr=0.05)
    print()
    lstm = LSTM(vocab_size=len(vocab_chars), emb_dim=8, hidden_dim=16, class_count=5)
    lstm_history = train_model("LSTM", lstm, train_data[:], test_data, args.epochs, lr=0.05)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "epoch", "loss", "train_acc", "test_acc"])
        writer.writeheader()
        for item in rnn_history + lstm_history:
            writer.writerow(
                {
                    "model": item.model,
                    "epoch": item.epoch,
                    "loss": f"{item.loss:.6f}",
                    "train_acc": f"{item.train_acc:.6f}",
                    "test_acc": f"{item.test_acc:.6f}",
                }
            )

    predictions = {
        "task": "五字文本中'你'的位置分类，类别为1到5",
        "vocab": "".join(vocab_chars),
        "rnn_examples": demo_predictions(rnn, test_data),
        "lstm_examples": demo_predictions(lstm, test_data),
    }
    pred_path = args.out_dir / "predictions.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    summary_path = args.out_dir / "experiment_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# RNN/LSTM 文本多分类实验\n\n")
        f.write("任务：输入任意 5 个汉字且恰好包含一个“你”，预测“你”出现在第几位，类别为 1 到 5。\n\n")
        f.write("数据：脚本随机合成训练集和测试集；非“你”的位置从固定汉字表中随机采样。\n\n")
        f.write("模型：纯 Python 标准库实现字符 embedding、vanilla RNN、LSTM、softmax 多分类和 BPTT。\n\n")
        f.write("最终结果：\n\n")
        for hist in (rnn_history, lstm_history):
            last = hist[-1]
            f.write(
                f"- {last.model}: epoch={last.epoch}, loss={last.loss:.4f}, "
                f"train_acc={last.train_acc:.3f}, test_acc={last.test_acc:.3f}\n"
            )
        f.write("\n运行方式：\n\n")
        f.write("```bash\npython outputs/train_text_rnn_lstm.py\n```\n")

    print()
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {pred_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
