'''
设计一个以文本为输入的五分类任务,实验一下用RNN,LSTM等模型的跑通训练。
可以选择如下任务:对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。

使用RNN/LSTM进行文本分类


模型结构：
输入层 -> 词向量层 -> RNN层/LSTM层 ->池化层 ->归一化 ->dropout -> 全连接层 -> 输出层
优化器：Adam
损失函数：交叉熵损失函数
'''



import torch
import torch.nn as nn
import random


# 1.数据生成
vocab_chars = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞" 
vocab = {char: i + 2 for i, char in enumerate(vocab_chars)}  # 从2开始，留出0和1
vocab['<PAD>'] = 0 #  在词汇表中添加特殊标记 '<PAD>'用于填充序列，使其长度一致
vocab['<UNK>'] = 1 #  '<UNK>'用于表示未知字符，当遇到词汇表中不存在的字符时使用
vocab_size = len(vocab)

def generate_data(num_samples, seq_length=5):
    """生成包含'你'字的5字文本及对应标签"""
    data_x, data_y = [], [] #  初始化数据列表，用于存储输入数据和对应的位置标签
    target_char = '你' #  设置目标字符，这里为'你'
    # 确保目标字符存在于词汇表中，若不存在则添加新索引
    if target_char not in vocab:
        vocab[target_char] = len(vocab) + 1 #  如果目标字符不在词汇表中，则添加到词汇表，分配新的索引
        print(f"新字符 '{target_char}' 已添加到词汇表中，索引为 {vocab[target_char]}")
    for _ in range(num_samples): #  循环生成指定数量的样本
        other_chars = random.sample([c for c in vocab_chars if c != target_char], seq_length - 1) #  从词汇表中随机选取不同的字符，数量为序列长度减1（因为要留一个位置给目标字符）
        pos = random.randint(0, seq_length - 1) #  随机生成目标字符在句子中的位置
        sentence = other_chars[:pos] + [target_char] + other_chars[pos:] #  构建句子：在其他字符的随机位置插入目标字符
        indices = [vocab[char] for char in sentence] #  将句子中的每个字符转换为对应的索引
        data_x.append(indices) #  将转换后的索引序列添加到输入数据列表中
        data_y.append(pos) #  将目标字符的位置添加到标签数据列表中
    return torch.tensor(data_x, dtype=torch.long), torch.tensor(data_y, dtype=torch.long) #  将数据转换为PyTorch张量，数据类型为长整型
    print(f"数据生成完毕，共有 {len(data_x)} 个样本。") 
    print(f"data_x: {data_x}, data_y: {data_y}") #  打印生成的数据样本和标签

X_train, y_train = generate_data(5000)

# 2. 创建模型
class TextPositionClassifier(nn.Module): 
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, model_type='lstm', dropout=0.3):
        super().__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embed_size) #词向量层 
        
        if model_type == 'rnn': #  根据模型类型选择不同的循环神经网络结构
            self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        else:
            raise ValueError("Invalid model type. Choose 'rnn' or 'lstm'.")
        
        # 层归一化 (Layer Normalization)：对 RNN 输出的特征维度进行归一化
        # 输入维度为 hidden_size，eps 是为了防止除以0的极小值
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)    
        self.dropout = nn.Dropout(dropout)   #dropout层，随机丢弃部分神经元，防止过拟合
        self.fc = nn.Linear(hidden_size, num_classes)  #全连接层

    def forward(self, x):
        embed = self.embedding(x)  # [batch, seq_len, embed_size]  # 将输入x通过嵌入层转换为词向量
        out, _ = self.rnn(embed)   # [batch, seq_len, hidden_size]  # 将词向量输入RNN层，得到输出
        
        # 层归一化：在池化之前先对特征进行标准化处理，稳定分布
        out = self.layer_norm(out)   # 对RNN输出的特征进行层归一化，有助于稳定训练过程
            
        # 沿着序列长度维度 (dim=1) 求最大值，提取整句话的综合特征
        # 形状从 [batch, seq_len, hidden_size] 变为 [batch, hidden_size]
        pooled_out = out.max(dim=1)[0]   # 对序列维度进行池化，将变长序列转换为固定长度的特征向量
        pooled_out = self.dropout(pooled_out) # 在池化后的特征向量上应用dropout，进一步防止过拟合

        return self.fc(pooled_out)  # 将池化后的特征向量通过全连接层进行最终的分类或回归预测

# 3. 训练配置
EMBED_SIZE = 128   # 词向量的维度
HIDDEN_SIZE = 128  # 隐藏层的维度
NUM_CLASSES = 5    # 类别数
LEARNING_RATE = 0.0001    # 学习率
EPOCHS = 40     # 训练轮数
BATCH_SIZE = 64    # 批次大小
#  设置随机种子，保证结果可复现
torch.manual_seed(42)
random.seed(42)


# 实例化模型
model = TextPositionClassifier(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_CLASSES, model_type='lstm', dropout=0.3)
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数适用于多分类任务，要求模型输出的形状为 [batch_size, num_classes]，标签为 [batch_size] 的类别索引
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4.训练和评估
def evaluate(model, X, y):
    model.eval() #  将模型设置为评估模式，关闭dropout等训练特定的行为
    with torch.no_grad(): #  在评估过程中不计算梯度，节省内存和计算资源
        outputs = model(X) #  前向传播，得到模型的输出
        _, predicted = torch.max(outputs.data, 1) #  获取每个样本预测的类别索引
        total = y.size(0) #  样本总数
        correct = (predicted == y).sum().item() #  计算预测正确的样本数量
    return correct / total * 100 #  返回准确率百分比

def train():
    print(f"\n开始训练带 LayerNorm 和 maxPooling 的 {model.model_type.upper()} 模型...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
    
        indices = torch.randperm(X_train.size(0)) #  使用torch.randperm生成一个随机排列的索引，用于打乱数据
        X_shuffled, y_shuffled = X_train[indices], y_train[indices]
     
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_x = X_shuffled[i:i+BATCH_SIZE]
            batch_y = y_shuffled[i:i+BATCH_SIZE]
        
            optimizer.zero_grad() #  清空之前的梯度，准备进行新的反向传播
            outputs = model(batch_x) #  前向传播
            loss = criterion(outputs, batch_y) #  计算损失
            loss.backward()   # 进行反向传播
            optimizer.step() #  根据梯度更新模型参数
        
            total_loss += loss.item() #  累加损失值
            _, predicted = torch.max(outputs.data, 1) 
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()  # 计算准确率
        
        avg_loss = total_loss / (len(X_train) / BATCH_SIZE) #  计算平均损失
        acc = evaluate(model, X_train, y_train) #  评估当前模型在训练集上的准确率    
    
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    print(f"\n训练完成！最终训练集准确率: {evaluate(model, X_train, y_train):.2f}%")


    print("\n--- 测试 ---")
    model.eval() 
    test_text = ["你好呀朋友", "我是你朋友", "清风是你呀","我你他她它","终于找到你","你是朋友","朋友是你"]  # 测试文本    
    with torch.no_grad():
        for text in (test_text): #  遍历测试文本
            print(f"测试文本: {text}") 
            ids = [vocab.get(c, vocab['<UNK>']) for c in text]
            ids += [vocab['<PAD>']] * (5 - len(ids))
            test_indices = torch.tensor([ids], dtype=torch.long)
            output = model(test_indices)
            pred_pos = torch.argmax(output, dim=1).item()
            print(f"预测'你'的位置是第 {pred_pos + 1} 位 (索引: {pred_pos})")

if __name__ == "__main__":
    train()
