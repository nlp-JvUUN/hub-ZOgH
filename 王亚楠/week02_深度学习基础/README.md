深度学习
训练流程：模型随机初始化——前向传播——计算loss——反向传播——根据优化器和学习率调整参数
想要快速的获得正确的模型：
  - 模型参数的随机初始化：
    - 在一定范围内随机初始化
    - NLP中的预训练模型实际上就是对随机初始化的技术优化（好的开始是成功的一半）
  - 优化损失函数
  - 调整参数的策略，优化器
  - 调整模型结构
导数和梯度：告诉模型调整参数的方向
梯度下降：梯度的反方向就是下降最快的方向
优化器：决定参数调整的步幅大小
要点：
- 模型结构选择
- 初始化方式选择
- 损失函数选择
- 优化器选择
- 样本质量选择
线性代数
向量加和：需维度相同
向量内积：对应位置元素相乘在相加，需维度相同
向量夹角余弦值：$$cos\theta=\frac{A * B}{|A||B|}$$
矩阵乘法（matmul）：不满足交换律，左矩阵列数需要等于右矩阵的行数，符合分配率和结合率
矩阵点乘（dot）：两矩阵必须形状一致
矩阵转置、矩阵reshape、矩阵flatten
张量（tensor）：若干组矩阵排列在一起，是神经网络的训练中最为常见的数据形式
numpy常用操作：
import numpy as np
# =================================================================
# 1. 数组基本属性 (Attributes)
# 用于查看数组的固有特征。
# =================================================================
def show_attributes():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print("--- 数组属性展示 ---")
    print(f"数组内容:\n{a}")
    print(f"`.ndim`  - 秩 (Rank)：{a.ndim}")            # 轴的数量或维度的数量
    print(f"`.shape` - 尺度：{a.shape}")               # 对象的大小，n行m列
    print(f"`.size`  - 对象个数：{a.size}")             # 元素总个数 (n*m)
    print(f"`.dtype` - 对象类型：{a.dtype}")           # 元素的数据类型
    print("\n")
# =================================================================
# 2. 数组创建 (Creation)
# 用于初始化并生成新的数组。
# =================================================================
def show_creation():
    print("--- 数组创建展示 ---")
    # `np.arange(n)`          - 类似range，返回0到n-1的数组
    print(f"`np.arange(5)`:\n{np.arange(5)}")
    # `np.ones(shape)`        - 生成全 1 数组
    print(f"`np.ones((2,3))`:\n{np.ones((2, 3))}")
    # `np.zeros(shape)`       - 生成全 0 数组
    print(f"`np.zeros((2,2))`:\n{np.zeros((2, 2))}")
    # `np.full(shape, val)`   - 每个元素值都为 val
    print(f"`np.full((2,2), 7)`:\n{np.full((2, 2), 7)}")
    # `np.eye(n)`             - 创建 n*n 单位矩阵
    print(f"`np.eye(3)`:\n{np.eye(3)}")
    # `np.linspace(b, e, n)`  - 起止范围内等间距填充数据
    print(f"`np.linspace(0, 10, 5)`:\n{np.linspace(0, 10, 5)}")
    print("\n")
# =================================================================
# 3. 形状变换与维度处理 (Shape Manipulation)
# =================================================================
def show_shape_manipulation():
    print("--- 形状变换展示 ---")
    a = np.arange(6)
    # `.reshape(shape)`       - 返回新形状数组，原数组不变
    print(f"`.reshape(2,3)`:\n{a.reshape(2, 3)}")
    # `.resize(shape)`        - 直接修改原数组
    a.resize((3, 2))
    print(f"`.resize(3,2)` 后原数组变为:\n{a}")
    # `.swapaxes(ax1, ax2)`   - 交换两个维度
    b = a.reshape(3, 2)
    print(f"`.swapaxes(0,1)`:\n{b.swapaxes(0, 1)}")
    # `.flatten()`            - 降维回一维数组，原数组不变
    print(f"`.flatten()`:\n{b.flatten()}")
    print("\n")
# =================================================================
# 4. 数学运算 (Mathematical Functions)
# =================================================================
def show_math_ops():
    print("--- 数学运算展示 ---")
    x = np.array([-1.5, 1.5, 4, 9])
    # `np.abs(x)` / `np.fabs(x)`     - 绝对值
    # `np.sqrt(x)`                   - 平方根
    # `np.square(x)`                 - 平方
    # `np.log(x)` / `np.log10(x)`    - 对数
    # `np.ceil(x)` / `np.floor(x)`   - 向上/下取整
    # `np.rint(x)`                   - 四舍五入
    # `np.modf(x)`                   - 拆分小数和整数部分
    # `np.exp(x)`                    - 指数数值 (e^x)
    # `np.sign(x)`                   - 符号值: 1(+), 0(0), -1(-) 
    print(f"原数组: {x}")
    print(f"`np.abs`   : {np.abs(x)}")
    print(f"`np.ceil`  : {np.ceil(x)}")
    print(f"`np.sign`  : {np.sign(x)}")
    # 三角函数示例
    print(f"`np.sin`   : {np.sin(np.pi/2)}") 
    print("\n")
# =================================================================
# 5. 统计分析 (Statistics)
# =================================================================
def show_statistics():
    print("--- 统计分析展示 ---")
    a = np.array([[1, 2], [3, 4]])
    # `sum(a, axis=None)`             - 计算和
    # `mean(a, axis=None)`            - 计算期望（算术平均值）
    # `average(a, axis, weights)`     - 加权平均值
    # `std(a, axis=None)`             - 标准差
    # `var(a, axis=None)`             - 方差
    # `min(a)` / `max(a)`             - 最小值/最大值
    # `ptp(a)`                        - 极差（最大减最小）
    # `median(a)`                     - 中位数
    print(f"数组:\n{a}")
    print(f"`sum`     : {np.sum(a)}")
    print(f"`mean`    : {np.mean(a)}")
    print(f"`std`     : {np.std(a)}")
    print(f"`ptp`     : {np.ptp(a)}")
    print("\n")
# =================================================================
# 6. 随机数生成 (Random)
# =================================================================
def show_random():
    print("--- 随机数展示 ---")
    # `seed(s)`                       - 随机数种子，使结果可复现
    np.random.seed(42)
    # `rand(d0, d1, ...)`             - [0, 1) 均匀分布
    print(f"`rand(2,2)`:\n{np.random.rand(2, 2)}")
    # `randn(d0, d1, ...)`            - 标准正态分布
    print(f"`randn(2,2)`:\n{np.random.randn(2, 2)}")  
    # `randint(low, high, shape)`     - [low, high) 范围内的随机整数
    print(f"`randint(0,10,(2,2))`:\n{np.random.randint(0, 10, (2, 2))}")
if __name__ == "__main__":
    show_attributes()
    show_creation()
    show_shape_manipulation()
    show_math_ops()
    show_statistics()
    show_random()
torch张量常用操作
import torch
import numpy as np
# =================================================================
# 1. 基础属性 (Attributes) - 注意与 NumPy 的区别
# =================================================================
def show_attributes():
    # 创建一个 Torch Tensor
    data = [[1, 2, 3], [4, 5, 6]]
    t = torch.tensor(data, dtype=torch.float32)
    print("--- Tensor 属性展示 ---")
    # 【区别】NumPy 使用 .ndim，Torch 也可以用 .ndim 或 .ndimension()
    print(f"`.ndim`      - 秩/维度数量: {t.ndim}")
    # 【注意】NumPy 只有 .shape 属性；
    # Torch 推荐使用 .size() 方法，但也支持 .shape 属性
    print(f"`.size()`    - 尺度 (方法): {t.size()}") 
    print(f"`.shape`     - 尺度 (属性): {t.shape}")
    # 【区别】NumPy 使用 .size 表示元素总数；Torch 使用 .numel() (Number of Elements)
    print(f"`.numel()`   - 元素总个数: {t.numel()}")
    # 【核心区别】Torch 必须关注设备（CPU/GPU）
    print(f"`.device`    - 存储设备: {t.device}")
    print("\n")

# =================================================================
# 2. 创建操作 (Creation)
# =================================================================
def show_creation():
    print("--- Tensor 创建展示 ---")
    # `torch.arange(n)`      - 类似 np.arange
    print(f"`torch.arange(5)`:\n{torch.arange(5)}")
    # `torch.ones/zeros`     - 类似 np.ones/zeros
    # 【注意】Torch 的 shape 可以直接传多个整数，不一定要传元组
    print(f"`torch.ones(2, 3)`:\n{torch.ones(2, 3)}")
    # `torch.full`           - 类似 np.full
    print(f"`torch.full((2,2), 7)`:\n{torch.full((2, 2), 7)}")
    # `torch.eye`            - 单位矩阵
    print(f"`torch.eye(3)`:\n{torch.eye(3)}")
    # `torch.linspace`       - 【注意】Torch 的 steps 是必须参数，且包含终点
    print(f"`torch.linspace(0, 10, steps=5)`:\n{torch.linspace(0, 10, steps=5)}")
    print("\n")
# =================================================================
# 3. 形状变换 (Shape Manipulation)
# =================================================================
def show_shape_manipulation():
    print("--- 形状变换展示 ---")
    t = torch.arange(6)    
    # 【核心区别】NumPy 常用 .reshape()；
    # Torch 常用 .view() 或 .reshape()。view 要求内存连续，reshape 更通用。
    print(f"`.view(2, 3)`:\n{t.view(2, 3)}")  
    # 【注意】Torch 没有 .resize() 这种原地修改形状的方法
    # Torch 对应 np.swapaxes 的方法是 .transpose(dim0, dim1) 或 .permute()
    t2 = t.view(3, 2)
    print(f"`.transpose(0, 1)`:\n{t2.transpose(0, 1)}")   
    # 【区别】NumPy 用 .flatten()；Torch 用 torch.flatten() 或 .view(-1)
    print(f"`.flatten()`:\n{t2.flatten()}")    
    # 【Torch 特有】unsqueeze/squeeze 用于增加或删除长度为 1 的维度（常用于 Batch 处理）
    t_ext = t2.unsqueeze(0) # 变成 (1, 3, 2)
    print(f"`.unsqueeze(0).shape`: {t_ext.shape}")
    print("\n")
# =================================================================
# 4. 数学与统计运算 (Math & Statistics)
# =================================================================
def show_math_stats():
    print("--- 数学与统计展示 ---")
    t = torch.tensor([[1., 2.], [3., 4.]])   
    # 【核心区别】NumPy 使用 `axis` 参数；Torch 使用 `dim` 参数
    # 例如：np.sum(a, axis=0) -> torch.sum(t, dim=0)
    print(f"`torch.sum(dim=0)`:\n{torch.sum(t, dim=0)}")  
    # 【注意】Torch 的统计函数（如 mean, std）通常要求浮点型数据 (float)
    print(f"`torch.mean()`: {torch.mean(t)}")
    # `torch.abs`, `torch.sqrt`, `torch.exp` 等与 NumPy 类似
    # 【区别】NumPy 用 .ptp()；Torch 没有 ptp，需要用 .max() - .min()
    print(f"极差 (max-min): {t.max() - t.min()}")
    print("\n")
# =================================================================
# 5. 随机数与种子 (Random)
# =================================================================
def show_random():
    print("--- 随机数展示 ---")
    # 【区别】NumPy 使用 np.random.seed()；Torch 使用 torch.manual_seed()
    torch.manual_seed(42)  
    # `torch.rand`           - [0, 1) 均匀分布
    print(f"`torch.rand(2, 2)`:\n{torch.rand(2, 2)}") 
    # `torch.randn`          - 标准正态分布
    print(f"`torch.randn(2, 2)`:\n{torch.randn(2, 2)}")
    # `torch.randint`        - 随机整数
    print(f"`torch.randint(0, 10, (2, 2))`:\n{torch.randint(0, 10, (2, 2))}")
# =================================================================
# 6. NumPy 与 Tensor 的互转
# =================================================================
def show_conversion():
    print("--- 互转展示 ---")
    # NumPy -> Tensor
    n = np.ones((2, 2))
    t = torch.from_numpy(n)
    print(f"NumPy 转 Tensor 类型: {type(t)}")
    # Tensor -> NumPy
    # 【注意】如果 Tensor 在 GPU 上，必须先执行 .cpu()
    # 如果 Tensor 需要求导，必须先执行 .detach()
    n2 = t.numpy()
    print(f"Tensor 转 NumPy 类型: {type(n2)}")
if __name__ == "__main__":
    show_attributes()
    show_creation()
    show_shape_manipulation()
    show_math_stats()
    show_random()
    show_conversion()
高等数学
导数：表明函数变化的方向 
  - 定义：$$f'(x_0) = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$$
  - 常见导数：   $$(x^n)' = nx^{n-1}
   $$   $$(e^x)' = e^x$$   $$(\ln x)' = \frac{1}{x}$$   $$(\sin x)' = \cos x$$   $$(\cos x)' = -\sin x $$
  - 链式法则：$$[f(g(x))]' = f'(g(x))g'(x)$$
深度学习求解目标：
  1. 损失函数越小，模型越好
  2. 学习的目标是损失函数最小化
  3. 模型权重影响损失函数值
  4. 通过梯度下降来找到最优权重
学习率：在经验值附近调整
完整的反向传播过程：
  1. 根据输入x和模型当前权重，计算预测值y
  2. 根据y和真实值Y使用loss函数计算loss
  3. 根据loss计算模型权重的梯度
  4. 使用梯度和学习率，根据优化器调整模型权重
网络结构
全连接层（线性层）
计算公式：$$y = xW^T+b$$  （W和b为训练参数） W的维度决定了输出的维度
[图片]
向量的哈达玛积（Hadamard Product），也称为逐元素乘积（Element-wise Product），是指两个向量对应元素相乘得到的新向量。
激活函数
为模型添加非线性因素，使模型具有拟合非线性函数的能力，能拟合更复杂的数据分布
- Sigmoid 激活函数：$$f(x) = \frac{1}{(1 + e^{-x})}$$
- ReLu激活函数：$$f(x) = \begin{cases}
  x, & \text{for } x \geq 0 \\
  0, & \text{for } x \lt 0
\end{cases}$$
- softmax激活函数：$$y_k = \frac{exp(a_k)}{\sum^n_{i=1} exp(a_i)}$$
学习率调度
- 阶梯衰减（step Decay）：每隔固定的epoch将学习率乘以衰减因子
- 余弦退火（Cosine Annealing）：学习率按余弦曲线从max衰减到min，周期性，末尾可以自动重启
- 热身阶段（Warmup）：训练初期先从很小的值线性增大，避免刚开始时梯度爆炸或者不稳定
损失函数
1. 均方误差（ MSE ：Mean Squared Error ）损失函数，也称为平方损失函数，是用于回归任务的一种常见损失函数。
该损失函数 用于衡量 模型对于每个样本预测值与实际值之间的平方差异的平均程度。

均方误差损失函数 的数学表达式为：
$$MSE = \frac{1}{n} \sum^n_{i=1}(y_i-\hat y_i)^2$$
其中：
  - $$n$$ 是样本数量
  - $$y_i$$ 是第 $$i$$个样本的 真实标签
  - $$\hat{y_i}$$ 是第 $$i$$个样本的 预测值
- 均方误差损失函数 对差异的平方进行了求和，这使得较大的误差在计算中得到了更大的权重，这也导致它对离群值敏感，因为它会放大离群值的影响。
- 均方误差损失函数 在回归问题中广泛使用，尤其在对误差敏感度较高的情况下。

2. 交叉熵损失函数：$$H(p,q) = -\sum p(x)logq(x)$$   p(x)真实样本标签、q(x)预测概率
交叉熵损失多用于 多分类任务，下面我们通过拆解交叉熵的公式来理解其作为损失函数的意义
假设我们在做一个  n分类的问题，模型预测的输出结果是 $$[x_1,  x_2, x_3, ...., x_n]$$
然后，我们选择交叉熵损失函数作为目标函数，通过反向传播调整模型的权重
交叉熵损失函数的公式 ：
$$\begin{aligned}
loss(x, class) 
&= -log(\frac{e^{x_{[class]}}}{\sum_je^{x_{j}}})\\
&= -x_{[class]} + log(\sum_j e^{x_{j}})
\end{aligned}$$
- $$x$$是预测结果，是一个向量 $$x=[x_1, x_2, x_3, ...., x_n]$$，其元素个数 和 类别数一样多
- class 表示这个样本的实际标签，比如，样本实际属于分类 2，那么 $$class=2$$  ， $$x_{[class]}$$ 就是$$x_2$$，就是取预测结果向量中的第二个元素，即，取其真实分类对应的那个类别的预测值

---
接下来，我们来拆解公式，理解公式：
1. 首先，交叉熵损失函数中包含了一个最基础的部分：$$softmax(x_i) = \frac{e^{x_i}}{\sum_{j=0}^{n} e^{x_{j}}}$$
  softmax 将分类的结果做了归一化：
  -  $$e^{x}$$ 的作用是将 $$x$$ 转换为非负数
  - 通过 softmax 公式 $$\frac{e^{x_i}}{\sum_{j=0}^ne^{x_j}}$$ 计算出该样本被分到 类别 $$i$$的概率，这里所有分类概率相加的总和等于1
2. 我们想要使预测结果中，真实分类的那个概率接近 100%。 我们取出真实类别的那个概率，(下标为 class)：
  $$\frac{e^{x_{[class]}}}{\sum_{j=0}^{n} e^{x_{j}}}$$，我们希望它的值是 100%
3.  作为损失函数，后面需要参与求导。乘/除法 表达式求导比较麻烦，所以最好想办法转化成加/减法表达式。最自然的想法是取对数，把乘/除法转化为加/减法表达式 ：
$$log{\frac{e^{x_{[class]}}}{\sum_{j=0}^{n} e^{x_{j}}}} = log{e^{x_{[class]}}} - log{\sum_{j=0}^{n} e^{x_{j}}}$$
- 由于对数单调增，那么，求 $$\frac{e^{x_{[class]}}}{\sum_{j=0}^{n} e^{x_{j}}}$$ 的最大值的问题，可以转化为求 $$log{\frac{e^{x_{[class]}}}{\sum_{j=0}^{n} e^{x_{j}}}}$$ 的最大值的问题。
-  $$\frac{e^{x_{[class]}}}{\sum_{j=0}^{n} e^{x_{j}}}$$ 的取值范围是 (0, 1)，最大值为1。 取对数之后，$$log{\frac{e^{x_{[class]}}}{\sum_{j=0}^{n} e^{x_{j}}}}$$ 的取值范围为 [−∞,0]，最大值为0
作为损失函数的意义是：当预测结果越接近真实值，损失函数的值越接近于0
所以，我们把 $$log{\frac{e^{x_{[class]}}}{\sum_{j=0}^{n} e^{x_{j}}}}$$ 取反之后，$$-log{\frac{e^{x_{[class]}}}{\sum_{j=0}^{n} e^{x_{j}}}}$$  最小值为0
这样就能保证当  $$\frac{e^{x_{[class]}}}{\sum_{j=0}^{n} e^{x_{j}}}$$ 越接近于 100%， $$loss=-log(\frac{e^{x_{[class]}}}{\sum_{j=0}^{n} e^{x_{j}}})$$  越接近0。
