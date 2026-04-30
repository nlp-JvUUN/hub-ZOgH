"""
设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。
任务名称：客服意图分类
输入：一段用户与客服对话时的自然语言文本
输出：该文本所属的客服意图类别，共 5 个互斥类别

类别定义：0-查询订单 1-申请退款 2-咨询产品 3-投诉问题 4-修改信息

tips:以下数据生成的部分由AI生成
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as pyplot

# ==================== 超参数 ====================
SEED = 36
N_SAMPLES = 5000 
MAXLEN = 32
EMBED_DIM = 16
HIDDEN_DIM = 16
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ==================== 1. 数据生成 ====================
# 类别映射
CATEGORIES = ["查询订单", "申请退款", "咨询产品", "投诉问题", "修改信息"]

# ----- 查询订单 -----
query_templates = [
    "我的{order_ref}{query_verb}？",
    "订单{status_query}",
    "请问{order_ref}的{logistics_info}",
    "{order_ref}{time_ref}能到吗",
    "我买的东西现在到哪里了",
    "帮我查一下{order_ref}",
    "{order_ref}怎么还没有{status_action}",
    "发货了吗",
    "{order_ref}{delivery_query}",
]
order_ref_list = ["我的订单", "快递", "包裹", "刚下的单", "昨天买的商品", "那个订单"]
status_query_list = ["什么状态了", "物流更新了吗", "有没有发货", "是否已签收"]
logistics_info_list = ["物流信息", "配送进度", "预计到达时间", "当前位置"]
time_ref_list = ["今天", "明天", "后天", "这周内", "预计时间"]
status_action_list = ["发货", "更新物流", "送到"]
delivery_query_list = ["什么时候配送", "快递员电话多少", "能改派吗"]

# ----- 申请退款 -----
refund_templates = [
    "我不想要了，{refund_verb}",
    "申请{refund_type}，订单",
    "收到的不对，我要{refund_verb}",
    "买错了，帮我{refund_verb}",
    "{refund_verb}，还没发货",
    "订单，{refund_reason}，申请{refund_type}",
    "可以{refund_verb}吗？我不想等了",
    "质量太差，坚决{refund_verb}",
    "少发了东西，{refund_verb}",
    "{refund_verb}流程是什么",
]
refund_verb_list = ["退货退款", "申请退款", "取消订单", "办理退货", "退款"]
refund_type_list = ["仅退款", "退货退款"]
refund_reason_list = ["不想要了", "买重了", "送人不需要了", "等待太久"]

# ----- 咨询产品 -----
product_templates = [
    "{feature}怎么样？",
    "请问{usage_query}",
    "这个支持{function}吗",
    "{spec}是多少",
    "适合{user_scene}使用吗",
    "有没有{color}的",
    "质保多久",
]
feature_list = ["质量", "续航", "性能", "防水等级", "材质", "重量", "尺寸", "充电速度"]
usage_query_list = ["怎么使用", "如何清洁", "能带上飞机吗", "需要安装驱动吗", "兼容Mac吗"]
function_list = ["快充", "无线充电", "蓝牙5.3", "降噪", "指纹解锁", "NFC"]
spec_list = ["电池容量", "内存大小", "屏幕尺寸", "像素", "功率"]
user_scene_list = ["户外运动", "办公", "学生", "老人", "儿童"]
color_list = ["黑色", "白色", "蓝色", "红色", "粉色", "绿色"]

# ----- 投诉问题 -----
complaint_templates = [
    "收到的是坏的，{dissatisfaction}",
    "物流{logistics_issue}，我要投诉",
    "{demand}",
    "售后服务{service_issue}，没人处理",
    "虚假宣传，根本没有{function}",
    "包装{packaging_issue}，导致商品损坏",
]
dissatisfaction_list = ["太失望了", "不会再买了", "要求赔偿", "差评"]
logistics_issue_list = ["太慢", "虚假签收", "丢件", "被雨淋湿"]
demand_list = ["咋回事", "无语", "气死我了"]
service_issue_list = ["踢皮球", "不回复", "态度差", "不承认问题"]
packaging_issue_list = ["破损", "脏污", "挤压变形"]

# ----- 修改信息 -----
modify_templates = [
    "帮我{modify_action}{modify_target}",
    "地址写错了，新地址是",
    "能改{info_field}吗？",
    "我想换成{new_spec}",
    "改发{new_color}的",
    "订单取消之前的修改，恢复原状",
    "帮我备注一下{note_content}",
]
modify_action_list = ["改", "修改", "调整", "更新"]
modify_target_list = ["收货地址", "手机号", "颜色规格", "尺码", "数量"]
info_field_list = ["地址", "电话", "姓名", "规格", "颜色", "尺码", "数量"]
new_spec_list = ["蓝色", "L码", "升级款", "64G"]
note_content_list = ["留门卫室", "周末送货", "按门铃", "不要打电话"]

# 数据生成辅助函数
def random_element(lst):
    return random.choice(lst)

def generate_query_order():
    """生成查询订单样本"""
    template = random.choice(query_templates)
    return template.format(
        order_ref=random_element(order_ref_list),
        query_verb=random_element(status_query_list + ["查询", "查一下", "追踪"]),
        status_query=random_element(status_query_list),
        logistics_info=random_element(logistics_info_list),
        time_ref=random_element(time_ref_list),
        status_action=random_element(status_action_list),
        delivery_query=random_element(delivery_query_list),
    )

def generate_refund():
    """生成申请退款样本"""
    template = random.choice(refund_templates)
    return template.format(
        refund_verb=random_element(refund_verb_list),
        refund_type=random_element(refund_type_list),
        refund_reason=random_element(refund_reason_list),
    )

def generate_product_consult():
    """生成咨询产品样本"""
    template = random.choice(product_templates)
    return template.format(
        feature=random_element(feature_list),
        usage_query=random_element(usage_query_list),
        function=random_element(function_list),
        spec=random_element(spec_list),
        user_scene=random_element(user_scene_list),
        color=random_element(color_list),
    )

def generate_complaint():
    """生成投诉问题样本"""
    template = random.choice(complaint_templates)
    return template.format(
        dissatisfaction=random_element(dissatisfaction_list),
        logistics_issue=random_element(logistics_issue_list),
        demand=random_element(demand_list),
        service_issue=random_element(service_issue_list),
        function=random_element(function_list),
        packaging_issue=random_element(packaging_issue_list),
    )

def generate_modify():
    """生成修改信息样本"""
    template = random.choice(modify_templates)
    return template.format(
        modify_action=random_element(modify_action_list),
        modify_target=random_element(modify_target_list),
        info_field=random_element(info_field_list),
        new_spec=random_element(new_spec_list),
        new_color=random_element(new_spec_list),
        note_content=random_element(note_content_list),
    )

# 生成函数映射
GENERATORS = {
    "查询订单": generate_query_order,
    "申请退款": generate_refund,
    "咨询产品": generate_product_consult,
    "投诉问题": generate_complaint,
    "修改信息": generate_modify,
}

def generate_samples_per_category(category, count, category_idx):
    """生成指定类别和数量的样本列表"""
    samples = []
    gen_func = GENERATORS[category]
    for _ in range(count):
        text = gen_func()
        samples.append((text, category_idx))
    return samples

# ==================== 2. 词表构建与编码 ====================
def build_vocab(data):
    """构建词表"""
    vocab = {
        "[pad]":0,
        "[unk]":1
    }
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab, max_len=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

# ==================== 3. Dataset / DataLoader ====================
class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len=MAXLEN):
        super().__init__()
        self.X = [encode(x, vocab, max_len) for x, _ in data]
        self.Y = [y for _, y in data]

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.X[index], dtype=torch.long),
            torch.tensor(self.Y[index], dtype=torch.long)
        )
    
# ==================== 4. 模型定义 ====================
class KeywordModel(nn.Module):
    """
    Input -> Embedding -> LSTM -> Pooling -> BN -> Dropout -> Linear -> Output
    (B, L) -> (B, L, Embed_dim) -> (B, L, hidden_dim) -> (B, hidden_dim) -> (B, hidden_dim) -> (B, hidden_dim) -> (B, nums_class)
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, nums_class=5, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1) # 最后一个维度变成1维
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, nums_class)
        self.loss = nn.CrossEntropyLoss() # 损失函数

    def forward(self, x, y_true=None):
        x, _ = self.lstm(self.embed(x)) # (B, L) -> (B, L, Embed_dim) -> (B, L, hidden_dim)
        x_permuted = x.permute(0, 2, 1) # 调整维度 (B, L, hidden_dim) -> (B, hidden_dim, L)
        pooled = self.pool(x_permuted).squeeze(-1) # 池化层降维 (B, hidden_dim, L) -> (B, hidden_dim, 1) -> (B, hidden_dim)
        dropout = self.dropout(self.bn(pooled)) # 归一化、dropout层不改变维度 (B, hidden_dim)
        y = self.linear(dropout) # 线性层 (B, hidden_dim) -> (B, nums_class)
        if y_true is not None:
            # 计算损失
            return self.loss(y, y_true)
        else:
            # 预测 输出概率值
            return torch.softmax(y, dim=1)
        
# ==================== 5. 训练与评估 ====================
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            y_pred = torch.argmax(model(x), dim=1)
            correct += (y == y_pred).sum().item()
            total += len(y)
    return correct / total

# ==================== 主程序 ====================

def main():
    # 生成样本
    total_per_category = N_SAMPLES // len(CATEGORIES)   # 每个类别1000条，总共5000条
    data = []
    
    for idx, cat in enumerate(CATEGORIES):
        samples = generate_samples_per_category(cat, total_per_category, idx)
        data.extend(samples)
    
    # 随机打乱顺序
    random.shuffle(data)

    # 构建词表
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")
    
    # 训练集和验证集
    split = int(N_SAMPLES * TRAIN_RATIO)
    train_set = DataLoader(TextDataset(data[:split], vocab), batch_size=BATCH_SIZE)
    val_set = DataLoader(TextDataset(data[split:], vocab), batch_size=BATCH_SIZE)

    # 模型
    model = KeywordModel(len(vocab))
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    # 模型总参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    # 记录日志
    log = []

    # 训练过程
    for epoch_idx in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for X, Y in train_set:
            loss = model(X, Y)  # 计算损失
            optim.zero_grad()   # 梯度清零
            loss.backward()     # 反向传播 计算梯度
            optim.step()        # 更新权重
            total_loss += loss.item()
        # 每轮平均损失
        avg_loss = total_loss / len(train_set)
        val_acc = evaluate(model, val_set)
        log.append((avg_loss, val_acc))
        print(f"Epoch {epoch_idx:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
    
    # 根据训练日志绘图
    pyplot.plot(range(len(log)), [loss for loss, _ in log], label="loss")
    pyplot.plot(range(len(log)), [acc for _, acc in log], label="acc")
    pyplot.legend()
    pyplot.show()

    # 推理
    test_sents = [
        "你们咋回事啊？气死我了。",
        "那个包裹到底发没发？",
        "这东西我不想要了，能退不？",
        "这个和那个有啥不一样？",
        "客服咋回事，真让人无语。",
        "你这个做的太差了，服务还那么差，有没有搞错",
        "我想看看我买的西瓜到哪里了",
        "这个空调是多少匹的",
        '我想修改一下收货地址',
        "这件衣服尺码不合适，我想退货",
        "颜色发错了，我要的是黑的。",
        "这款耳机的续航怎么样？",
        "我买的东西怎么还没影儿？",
    ]
    model.eval()
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            y_pred = model(ids)
            prob, pred_idx = torch.max(y_pred, dim=1)
            print(f" {pred_idx.item()}-{CATEGORIES[pred_idx]}(概率:{prob.item():.2f}) : {sent}")


if __name__ == "__main__":
    main()
