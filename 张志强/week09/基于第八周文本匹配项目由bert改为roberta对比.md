## 基于第八周文本匹配项目由 bert 转为 roberta  比较差异  下面BERT记录数据是第八周已经输出好的

1. 数据分析  python explore_data.py
2. 训练 BiEncoder
   2.1 CosineEmbeddingLoss 训练   python train_biencoder.py --loss cosine
       Loss 类型: cosine  池化策略: mean  BERT 层数: 4  Epochs: 3    Batch Size: 16
   2.2 TripletLoss 训练 python train_biencoder.py --loss triplet --margin 0.3
   2.3 评估
4. 训练  CrossEncoder
   4.1 评估
5. 对比

6. bert 和 roberta 都是用同样流程和数据集 最后进行对比

## 使用BERT ---------------开始-------
# 数据分析   data/bq_corpus

【train】共 68,960 条
==================================================
  正样本（相似）  : 34,438 (49.9%)
  负样本（不相似）: 34,522 (50.1%)
  不均衡比 (neg/pos): 1.0x
  句子字符长度 — 均值=13.9  中位数=10  P95=25  最长=51842
  示例正样本：
    ✓  '存款有保障吗'  ||  '不知道安全吗'
    ✓  '你们电话确认要多久？'  ||  '电话确认一般得等候多长时间'
  示例负样本：
    ✗  '比如我今天借了一万分10个月！结果我五个月的时候有能力还清！那么利息还是收取十个月的吗？'  ||  '利息怎么计算，哪一天计起'
    ✗  '现在有什消息吗'  ||  '微粒贷款怎么提醒'

==================================================
【validation】共 8,620 条
==================================================
  正样本（相似）  :  4,329 (50.2%)
  负样本（不相似）:  4,291 (49.8%)
  不均衡比 (neg/pos): 1.0x
  句子字符长度 — 均值=13.4  中位数=10  P95=25  最长=27131
  示例正样本：
    ✓  '怎么改卡还款'  ||  '怎样变更还款卡'
    ✓  '你好能给我发个结清单么'  ||  '你们能否给我提供一下贷款结清证明？'
  示例负样本：
    ✗  '一般电话确认多久会打？'  ||  '一般多久接贷成功'
    ✗  '银行卡邦定了都不可以的'  ||  '我绑定的银行卡能申请信用卡吗'

==================================================
【test】共 8,620 条
==================================================
  正样本（相似）  :  4,382 (50.8%)
  负样本（不相似）:  4,238 (49.2%)
  不均衡比 (neg/pos): 1.0x
  句子字符长度 — 均值=16.1  中位数=10  P95=26  最长=43885
  示例正样本：
    ✓  '什么时候还款'  ||  '什么时候能进行还款'
    ✓  '2天收不到电话'  ||  '24小时都没打电话过来'
  示例负样本：
    ✗  '钱包密码忘了'  ||  '没有密码管理啊我去那里找'
    ✗  '你好，我有一笔贷款，如果现在提前还清，稍后还可以再贷款吗'  ||  '每期账单能提前还吗？'

==================================================

## 训练  data/bq_corpus  loss cosine

Loss 类型: cosine  池化策略: mean  BERT 层数: 4  Epochs: 3  Batch Size: 16

DataLoader 构建中...
  train :  68,960 条,  4310 batch
  val   :   8,620 条,   539 batch
  test  :   8,620 条,   539 batch  (AFQMC test 无正样本，仅供参考)

构建模型...
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)
总训练步数: 12930  Warmup 步数: 1293
Epoch 1/3 | train_loss=0.2250 | val_acc=0.8225 val_f1=0.8219 threshold=0.65 | 839s
  ✓ 新最优模型已保存 → /outputs/checkpoints/biencoder_cosine_best.pt  (val_f1=0.8219)
Epoch 2/3 | train_loss=0.1646 | val_acc=0.8571 val_f1=0.8570 threshold=0.70 | 835s
  ✓ 新最优模型已保存 → /outputs/checkpoints/biencoder_cosine_best.pt  (val_f1=0.8570)
Epoch 3/3 | train_loss=0.1425 | val_acc=0.8675 val_f1=0.8672 threshold=0.63 | 846s 
  ✓ 新最优模型已保存 → /outputs/checkpoints/biencoder_cosine_best.pt  (val_f1=0.8672)
训练完成。最优 val_f1=0.8672
训练日志 → /outputs/logs/biencoder_cosine_log.json
最优 checkpoint → /outputs/checkpoints/biencoder_cosine_best.pt

## 评估  data/bq_corpus  loss cosine

设备: mps
加载 checkpoint: /Users/zhangzhiqiang/Dev/project/Python/study/第八周/课件代码/文本匹配项目_自练习/outputs/checkpoints/biencoder_cosine_best.pt
训练信息: {'bert_path': '/Users/zhangzhiqiang/Dev/project/Python/study/pretrain_models/bert-base-chinese', 'data_dir': '/Users/zhangzhiqiang/Dev/project/Python/study/第八周/课件代码/文本匹配项目_自练习/data/bq_corpus', 'loss': 'cosine', 'pool': 'mean', 'num_hidden_layers': 4, 'epochs': 3, 'batch_size': 16, 'max_length': 64, 'lr': 2e-05, 'head_lr_mult': 5.0, 'warmup_ratio': 0.1, 'grad_accum': 1, 'margin': 0.3}
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

==================================================
BiEncoder 评估结果（validation，8620 条）
  最优阈值: 0.63
  Accuracy: 0.8675
  F1      : 0.8672
  AUC     : 0.9308

precision    recall  f1-score   support

         不相似       0.90      0.82      0.86      4291
          相似       0.84      0.91      0.87      4329

    accuracy                           0.87      8620
   macro avg       0.87      0.87      0.87      8620
weighted avg       0.87      0.87      0.87      8620

## 训练 TripletLoss

设备: mps
Loss 类型: triplet  池化策略: mean  BERT 层数: 4  Epochs: 3    Batch Size: 16

DataLoader 构建中...
  TripletDataset: 构建 34,438 个三元组
  triplet train :  34,438 三元组,  2153 batch
  val (pair)    :   8,620 对,       539 batch

构建模型...
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)
总训练步数: 6459  Warmup 步数: 645

Epoch 1/3 | train_loss=0.1169 | val_acc=0.8296 val_f1=0.8296 threshold=0.55 | 602s                                                                                                 
  ✓ 新最优模型已保存 → /outputs/checkpoints/biencoder_triplet_best.pt  (val_f1=0.8296)
Epoch 2/3 | train_loss=0.0435 | val_acc=0.8559 val_f1=0.8559 threshold=0.55 | 587s                                                                                                 
  ✓ 新最优模型已保存 → /outputs/checkpoints/biencoder_triplet_best.pt  (val_f1=0.8559)
Epoch 3/3 | train_loss=0.0265 | val_acc=0.8686 val_f1=0.8685 threshold=0.55 | 584s                                                                                                 
  ✓ 新最优模型已保存 → /outputs/checkpoints/biencoder_triplet_best.pt  (val_f1=0.8685)

训练完成。最优 val_f1=0.8685
训练日志 → /outputs/logs/biencoder_triplet_log.json
最优 checkpoint → /outputs/checkpoints/biencoder_triplet_best.pt

## 评估 TripletLoss
设备: mps
加载 checkpoint: /outputs/checkpoints/biencoder_triplet_best.pt
训练信息: {'bert_path': '/pretrain_models/bert-base-chinese', 'data_dir': '/文本匹配项目_自练习/data/bq_corpus', 'loss': 'triplet', 'pool': 'mean', 'num_hidden_layers': 4, 'epochs': 3, 'batch_size': 16, 'max_length': 64, 'lr': 2e-05, 'head_lr_mult': 5.0, 'warmup_ratio': 0.1, 'grad_accum': 1, 'margin': 0.3}
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

==================================================
BiEncoder 评估结果（validation，8620 条）
  最优阈值: 0.55
  Accuracy: 0.8686
  F1      : 0.8685
  AUC     : 0.9361
precision    recall  f1-score   support

         不相似       0.88      0.86      0.87      4291
          相似       0.86      0.88      0.87      4329

    accuracy                           0.87      8620
   macro avg       0.87      0.87      0.87      8620
weighted avg       0.87      0.87      0.87      8620


## 训练 CrossEncoder

设备: mps
BERT 层数: 4  Epochs: 3  Batch size: 32

DataLoader 构建中...
  train :  68,960 条,  2155 batch
  val   :   8,620 条,   270 batch
  test  :   8,620 条,   270 batch

构建模型...
模型: CrossEncoder (layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)
总训练步数: 6465  Warmup 步数: 646

Epoch 1/3 | train_loss=0.5021 train_acc=0.7448 | val_acc=0.8325 val_f1=0.8324 | 730s
  ✓ 新最优模型已保存 → /outputs/checkpoints/crossencoder_best.pt  (val_f1=0.8324)
Epoch 2/3 | train_loss=0.3408 train_acc=0.8521 | val_acc=0.8661 val_f1=0.8661 | 733s
  ✓ 新最优模型已保存 → /outputs/checkpoints/crossencoder_best.pt  (val_f1=0.8661)
Epoch 3/3 | train_loss=0.2616 train_acc=0.8925 | val_acc=0.8807 val_f1=0.8807 | 727s
  ✓ 新最优模型已保存 → /outputs/checkpoints/crossencoder_best.pt  (val_f1=0.8807)
训练完成。最优 val_f1=0.8807
训练日志 → /outputs/logs/crossencoder_log.json
最优 checkpoint → /outputs/checkpoints/crossencoder_best.pt


## 评估 CrossEncoder

设备: mps
加载 checkpoint: /outputs/checkpoints/crossencoder_best.pt
训练信息: {'bert_path': '/pretrain_models/bert-base-chinese', 'data_dir': '/文本匹配项目_自练习/data/bq_corpus', 'num_hidden_layers': 4, 'epochs': 3, 'batch_size': 32, 'max_length': 128, 'lr': 2e-05, 'head_lr_mult': 5.0, 'warmup_ratio': 0.1, 'grad_accum': 1}
模型: CrossEncoder (layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

==================================================
CrossEncoder 评估结果（validation，8620 条）
  Accuracy: 0.8807
  F1      : 0.8807

              precision    recall  f1-score   support

         不相似       0.87      0.89      0.88      4291
          相似       0.89      0.87      0.88      4329

    accuracy                           0.88      8620
   macro avg       0.88      0.88      0.88      8620
weighted avg       0.88      0.88      0.88      8620

##  三种对比

设备: mps  评估集: validation

=======================================================
加载 biencoder_cosine ...
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

=======================================================
加载 biencoder_triplet ...
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

=======================================================
加载 crossencoder ...
模型: CrossEncoder (layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

=================================================================
方法                              Accuracy  F1(weighted)            额外信息
-----------------------------------------------------------------
  biencoder_cosine                0.8675        0.8672  threshold=0.63
  biencoder_triplet               0.8686        0.8685  threshold=0.55
  crossencoder                    0.8807        0.8807          argmax

─────────────────────────────────────────────────────────────────
结论速览：
  最高 Accuracy : crossencoder (0.8807)
  最高 F1       : crossencoder  (0.8807)

  Cosine vs Triplet (Δ):
    Accuracy: +0.0010  F1: +0.0013
    → 两种 Loss 差距不大（1 epoch + 少量三元组限制了 Triplet 的优势）

对比日志 → /outputs/logs/method_comparison.json
  图表已保存 → /outputs/figures/method_comparison_bar.png
  图表已保存 → /outputs/figures/biencoder_sim_distributions.png

## 使用BERT ---------------结束-------


-----------------------------------上下两种分割线------------------------------------------------------


## 使用 ROBERTA ---------------开始------- 将之前的模型换成Bert的模型变体 ROBERTA

# 数据分析   data/bq_corpus

==================================================
【train】共 68,960 条
==================================================
  正样本（相似）  : 34,438 (49.9%)
  负样本（不相似）: 34,522 (50.1%)
  不均衡比 (neg/pos): 1.0x
  句子字符长度 — 均值=13.9  中位数=10  P95=25  最长=51842
  示例正样本：
    ✓  '存款有保障吗'  ||  '不知道安全吗'
    ✓  '你们电话确认要多久？'  ||  '电话确认一般得等候多长时间'
  示例负样本：
    ✗  '比如我今天借了一万分10个月！结果我五个月的时候有能力还清！那么利息还是收取十个月的吗？'  ||  '利息怎么计算，哪一天计起'
    ✗  '现在有什消息吗'  ||  '微粒贷款怎么提醒'

==================================================
【validation】共 8,620 条
==================================================
  正样本（相似）  :  4,329 (50.2%)
  负样本（不相似）:  4,291 (49.8%)
  不均衡比 (neg/pos): 1.0x
  句子字符长度 — 均值=13.4  中位数=10  P95=25  最长=27131
  示例正样本：
    ✓  '怎么改卡还款'  ||  '怎样变更还款卡'
    ✓  '你好能给我发个结清单么'  ||  '你们能否给我提供一下贷款结清证明？'
  示例负样本：
    ✗  '一般电话确认多久会打？'  ||  '一般多久接贷成功'
    ✗  '银行卡邦定了都不可以的'  ||  '我绑定的银行卡能申请信用卡吗'

==================================================
【test】共 8,620 条
==================================================
  正样本（相似）  :  4,382 (50.8%)
  负样本（不相似）:  4,238 (49.2%)
  不均衡比 (neg/pos): 1.0x
  句子字符长度 — 均值=16.1  中位数=10  P95=26  最长=43885
  示例正样本：
    ✓  '什么时候还款'  ||  '什么时候能进行还款'
    ✓  '2天收不到电话'  ||  '24小时都没打电话过来'
  示例负样本：
    ✗  '钱包密码忘了'  ||  '没有密码管理啊我去那里找'
    ✗  '你好，我有一笔贷款，如果现在提前还清，稍后还可以再贷款吗'  ||  '每期账单能提前还吗？'

==================================================

## 训练  data/bq_corpus  loss cosine
设备: mps
Loss 类型: cosine  池化策略: mean  BERT 层数: 4  Epochs: 3    Batch Size: 16

DataLoader 构建中...
  train :  68,960 条,  4310 batch
  val   :   8,620 条,   539 batch
  test  :   8,620 条,   539 batch  (AFQMC test 无正样本，仅供参考)

构建模型...
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)
总训练步数: 12930  Warmup 步数: 1293
Epoch 1/3 | train_loss=0.2237 | val_acc=0.8292 val_f1=0.8292 threshold=0.72 | 839s                                                                                                 
  ✓ 新最优模型已保存 → /outputs/checkpoints/biencoder_cosine_mean_best.pt  (val_f1=0.8292)
Epoch 2/3 | train_loss=0.1643 | val_acc=0.8622 val_f1=0.8620 threshold=0.69 | 850s                                                                                                 
  ✓ 新最优模型已保存 → /outputs/checkpoints/biencoder_cosine_mean_best.pt  (val_f1=0.8620)
Epoch 3/3 | train_loss=0.1447 | val_acc=0.8693 val_f1=0.8691 threshold=0.70 | 861s                                                                                                 
  ✓ 新最优模型已保存 → /outputs/checkpoints/biencoder_cosine_mean_best.pt  (val_f1=0.8691)

训练完成。最优 val_f1=0.8691
训练日志 → /outputs/logs/biencoder_cosine_log.json
最优 checkpoint → /outputs/checkpoints/biencoder_cosine_mean_best.pt

## 评估  data/bq_corpus  loss cosine
设备: mps
加载 checkpoint: /Users/zhangzhiqiang/Dev/project/Python/study/第九周/文本匹配项目_自练习_roberta/outputs/checkpoints/biencoder_cosine_mean_best.pt
训练信息: {'bert_path': '/Users/zhangzhiqiang/Dev/project/Python/study/pretrain_models/chinese-roberta-wwm-ext', 'data_dir': '/Users/zhangzhiqiang/Dev/project/Python/study/第九周/文本匹配项目_自练习_roberta/data/bq_corpus', 'loss': 'cosine', 'pool': 'mean', 'num_hidden_layers': 4, 'epochs': 3, 'batch_size': 16, 'max_length': 64, 'lr': 2e-05, 'head_lr_mult': 5.0, 'warmup_ratio': 0.1, 'grad_accum': 1, 'margin': 0.3}
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

==================================================
BiEncoder 评估结果（validation，8620 条）
  最优阈值: 0.70
  Accuracy: 0.8693
  F1      : 0.8691
  AUC     : 0.9281
  图表已保存 → /outputs/figures/biencoder_validation_sim_dist.png

              precision    recall  f1-score   support

         不相似       0.89      0.84      0.86      4291
          相似       0.85      0.90      0.87      4329

    accuracy                           0.87      8620
   macro avg       0.87      0.87      0.87      8620
weighted avg       0.87      0.87      0.87      8620

## 训练 TripletLoss

设备: mps
Loss 类型: triplet  池化策略: mean  BERT 层数: 4  Epochs: 3    Batch Size: 16

DataLoader 构建中...
  TripletDataset: 构建 34,438 个三元组
  triplet train :  34,438 三元组,  2153 batch
  val (pair)    :   8,620 对,       539 batch

构建模型...
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)
总训练步数: 6459  Warmup 步数: 645
 val_acc=0.8349 val_f1=0.8348 threshold=0.54 | 606s                                                                                                 
  ✓ 新最优模型已保存 → /outputs/checkpoints/biencoder_triplet_mean_best.pt  (val_f1=0.8348)
Epoch 2/3 | train_loss=0.0420 | val_acc=0.8624 val_f1=0.8623 threshold=0.51 | 639s                                                                                                 
  ✓ 新最优模型已保存 → /outputs/checkpoints/biencoder_triplet_mean_best.pt  (val_f1=0.8623)
Epoch 3/3 | train_loss=0.0266 | val_acc=0.8695 val_f1=0.8694 threshold=0.52 | 621s                                                                                                 
  ✓ 新最优模型已保存 → /outputs/checkpoints/biencoder_triplet_mean_best.pt  (val_f1=0.8694)

训练完成。最优 val_f1=0.8694
训练日志 → /outputs/logs/biencoder_triplet_log.json
最优 checkpoint → /outputs/checkpoints/biencoder_triplet_mean_best.pt

## 评估 TripletLoss

设备: mps
加载 checkpoint: /outputs/checkpoints/biencoder_triplet_mean_best.pt
训练信息: {'bert_path': '/Users/zhangzhiqiang/Dev/project/Python/study/pretrain_models/chinese-roberta-wwm-ext', 'data_dir': '/Users/zhangzhiqiang/Dev/project/Python/study/第九周/文本匹配项目_自练习_roberta/data/bq_corpus', 'loss': 'triplet', 'pool': 'mean', 'num_hidden_layers': 4, 'epochs': 3, 'batch_size': 16, 'max_length': 64, 'lr': 2e-05, 'head_lr_mult': 5.0, 'warmup_ratio': 0.1, 'grad_accum': 1, 'margin': 0.3}
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

==================================================
BiEncoder 评估结果（validation，8620 条）
  最优阈值: 0.52
  Accuracy: 0.8695
  F1      : 0.8694
  AUC     : 0.9383
图表已保存 → /outputs/figures/biencoder_validation_sim_dist.png

              precision    recall  f1-score   support

         不相似       0.89      0.85      0.87      4291
          相似       0.85      0.89      0.87      4329

    accuracy                           0.87      8620
   macro avg       0.87      0.87      0.87      8620
weighted avg       0.87      0.87      0.87      8620

## 训练 CrossEncoder

设备: mps
BERT 层数: 4  Epochs: 3  Batch size: 32

DataLoader 构建中...
  train :  68,960 条,  2155 batch
  val   :   8,620 条,   270 batch
  test  :   8,620 条,   270 batch

构建模型...
模型: CrossEncoder (layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)
总训练步数: 6465  Warmup 步数: 646
Epoch 1/3 | train_loss=0.5228 train_acc=0.7253 | val_acc=0.8172 val_f1=0.8167 | 753s                                                                                               
  ✓ 新最优模型已保存 → /outputs/checkpoints/crossencoder_best.pt  (val_f1=0.8167)
Epoch 2/3 | train_loss=0.3709 train_acc=0.8362 | val_acc=0.8515 val_f1=0.8512 | 750s                                                                                               
  ✓ 新最优模型已保存 → /outputs/checkpoints/crossencoder_best.pt  (val_f1=0.8512)
Epoch 3/3 | train_loss=0.3014 train_acc=0.8733 | val_acc=0.8611 val_f1=0.8610 | 742s                                                                                               
  ✓ 新最优模型已保存 → /outputs/checkpoints/crossencoder_best.pt  (val_f1=0.8610)

训练完成。最优 val_f1=0.8610
训练日志 → /outputs/logs/crossencoder_log.json
最优 checkpoint → /outputs/checkpoints/crossencoder_best.pt

## 评估 CrossEncoder

设备: mps
加载 checkpoint: /outputs/checkpoints/crossencoder_best.pt
训练信息: {'bert_path': '/Users/zhangzhiqiang/Dev/project/Python/study/pretrain_models/chinese-roberta-wwm-ext', 'data_dir': '/Users/zhangzhiqiang/Dev/project/Python/study/第九周/文本匹配项目_自练习_roberta/data/bq_corpus', 'num_hidden_layers': 4, 'epochs': 3, 'batch_size': 32, 'max_length': 128, 'lr': 2e-05, 'head_lr_mult': 5.0, 'warmup_ratio': 0.1, 'grad_accum': 1}
模型: CrossEncoder (layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

==================================================
CrossEncoder 评估结果（validation，8620 条）
  Accuracy: 0.8611
  F1      : 0.8610

              precision    recall  f1-score   support

         不相似       0.88      0.83      0.86      4291
          相似       0.84      0.89      0.87      4329

    accuracy                           0.86      8620
   macro avg       0.86      0.86      0.86      8620
weighted avg       0.86      0.86      0.86      8620

## 七、方法对比（三种训练方式）

设备: mps  评估集: validation

=======================================================
加载 biencoder_cosine ...
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

=======================================================
加载 biencoder_triplet ...
模型: BiEncoder (pool=mean, layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

=======================================================
加载 crossencoder ...
模型: CrossEncoder (layers=4)
参数量: 45.6M  (BERT 骨干: 45.6M)

=================================================================
方法                              Accuracy  F1(weighted)            额外信息
-----------------------------------------------------------------
  biencoder_cosine                0.8669        0.8667  threshold=0.64
  biencoder_triplet               0.8675        0.8675  threshold=0.55
  crossencoder                    0.8611        0.8610          argmax

─────────────────────────────────────────────────────────────────
结论速览：
  最高 Accuracy : biencoder_triplet (0.8675)
  最高 F1       : biencoder_triplet  (0.8675)

  Cosine vs Triplet (Δ):
    Accuracy: +0.0006  F1: +0.0008
    → 两种 Loss 差距不大（1 epoch + 少量三元组限制了 Triplet 的优势）

对比日志 → /outputs/logs/method_comparison.json
  图表已保存 → /outputs/figures/method_comparison_bar.png
  图表已保存 → /outputs/figures/biencoder_sim_distributions.png

## 使用 ROBERTA ---------------结束------- 将之前的模型换成Bert的模型变体 ROBERTA

---

## 八、BERT vs RoBERTa 对比分析

> 对比基准：同一数据集 `bq_corpus`、同一训练配置（4 层 Transformer、3 epoch、lr=2e-5）、同一设备 MPS。  
> 预训练模型：BERT = `bert-base-chinese`，RoBERTa = `chinese-roberta-wwm-ext`。

### 8.1 实验设置对照

| 项目 | BERT | RoBERTa |
|------|------|---------|
| 预训练模型 | bert-base-chinese | chinese-roberta-wwm-ext |
| 参数量 | 45.6M（4 层） | 45.6M（4 层） |
| 数据 | bq_corpus 68,960 train / 8,620 val | 相同 |
| BiEncoder 配置 | mean pool, max_length=64, bs=16 | 相同 |
| CrossEncoder 配置 | max_length=128, bs=32 | 相同 |

数据分布完全一致（正负样本各约 50%，句子长度均值 ~14 字），因此性能差异主要来自**预训练模型本身**及**架构与下游任务的匹配度**。

### 8.2 验证集核心指标对比

| 方法 | BERT Acc | BERT F1 | RoBERTa Acc | RoBERTa F1 | Δ Acc | Δ F1 |
|------|----------|---------|-------------|------------|-------|------|
| BiEncoder + Cosine | 0.8675 | 0.8672 | 0.8693 | 0.8691 | **+0.0018** | **+0.0019** |
| BiEncoder + Triplet | 0.8686 | 0.8685 | 0.8695 | 0.8694 | **+0.0009** | **+0.0009** |
| CrossEncoder | 0.8807 | 0.8807 | 0.8611 | 0.8610 | **-0.0196** | **-0.0197** |

BiEncoder 额外指标（AUC）：

| 方法 | BERT AUC | RoBERTa AUC | Δ AUC |
|------|----------|-------------|-------|
| Cosine | 0.9308 | 0.9281 | -0.0027 |
| Triplet | 0.9361 | 0.9383 | +0.0022 |

**关键发现：两种 backbone 在 BiEncoder 上几乎打平，但在 CrossEncoder 上差距最大（约 2 个百分点）。**

### 8.3 最优方法排序变化（重要）

| 排名 | BERT（validation） | RoBERTa（validation） |
|------|-------------------|---------------------|
| 第 1 | CrossEncoder **0.8807** | BiEncoder Triplet **0.8675** |
| 第 2 | BiEncoder Triplet 0.8685 | BiEncoder Cosine 0.8667 |
| 第 3 | BiEncoder Cosine 0.8672 | CrossEncoder 0.8610 |

- **BERT**：CrossEncoder 明显优于 BiEncoder（+1.2% ~ +1.3% F1），符合「交互式编码在句对匹配上更强」的常规结论。
- **RoBERTa**：CrossEncoder 反而垫底，BiEncoder 两种 Loss 略优。  
  → **换 backbone 后，方法优劣关系发生了反转**，说明不能默认「CrossEncoder 一定更好」，需结合具体预训练模型验证。

### 8.4 训练过程对比

#### BiEncoder + Cosine（3 epoch 末）

| 指标 | BERT | RoBERTa |
|------|------|---------|
| train_loss | 0.1425 | 0.1447 |
| val_f1 | 0.8672 | 0.8691 |
| 最优阈值 | 0.63 | 0.70 |
| 单 epoch 耗时 | ~840s | ~850s |

RoBERTa 训练 loss 略高，但 val F1 略高；最优分类阈值从 0.63 上移到 0.70，说明**相似度分数整体分布更「保守」**（正例需更高相似度才判为相似）。

#### BiEncoder + Triplet（3 epoch 末）

| 指标 | BERT | RoBERTa |
|------|------|---------|
| train_loss | 0.0265 | 0.0266 |
| val_f1 | 0.8685 | 0.8694 |
| 最优阈值 | 0.55 | 0.52 |
| 单 epoch 耗时 | ~584–602s | ~606–639s |

两者几乎一致；Triplet 在两种 backbone 下均略优于 Cosine（Δ F1 ≈ 0.001），差距依然很小。

#### CrossEncoder（3 epoch 末）—— 差异最大

| 指标 | BERT | RoBERTa |
|------|------|---------|
| train_loss | 0.2616 | 0.3014 |
| train_acc | 0.8925 | 0.8733 |
| val_f1 | **0.8807** | **0.8610** |
| 单 epoch 耗时 | ~727–733s | ~742–753s |

RoBERTa 的 CrossEncoder：
- 训练 loss 更高、train acc 更低 → **拟合更慢、更难收敛**
- val F1 低约 2 点 → **泛化明显弱于 BERT**

Epoch 1→3 的 val F1 提升：

| 模型 | Epoch1 | Epoch2 | Epoch3 | 总提升 |
|------|--------|--------|--------|--------|
| BERT CrossEncoder | 0.8324 | 0.8661 | 0.8807 | +0.0483 |
| RoBERTa CrossEncoder | 0.8167 | 0.8512 | 0.8610 | +0.0443 |

RoBERTa 起点更低、终点更低，全程落后 BERT 约 1.5~2 个百分点。

### 8.5 分类行为差异（Precision / Recall）

#### BiEncoder Cosine

| 类别 | BERT P/R | RoBERTa P/R |
|------|----------|-------------|
| 不相似 | 0.90 / 0.82 | 0.89 / 0.84 |
| 相似 | 0.84 / 0.91 | 0.85 / 0.90 |

- BERT：更倾向判「相似」（相似类 recall 0.91，不相似 recall 仅 0.82）
- RoBERTa：两类 recall 更均衡（0.84 vs 0.90），**误杀相似句略少**

#### CrossEncoder

| 类别 | BERT P/R | RoBERTa P/R |
|------|----------|-------------|
| 不相似 | 0.87 / **0.89** | 0.88 / **0.83** |
| 相似 | 0.89 / **0.87** | 0.84 / **0.89** |

- BERT CrossEncoder：两类较均衡，整体 F1 最高
- RoBERTa CrossEncoder：**不相似 recall 明显偏低（0.83）**，更多负样本被误判为相似 → 这是 F1 下降的主要原因之一

### 8.6 原因分析（结合模型特性）

1. **BiEncoder 场景：两者接近**
   - 两种模型都是「分别编码 + 向量相似度」，对预训练差异不敏感
   - RoBERTa（全词掩码 WWM）在字/词级表示上可能略优，带来 +0.1~0.2% 的微弱提升
   - 差异在统计误差范围内，**可认为 BiEncoder 下两者等价**

2. **CrossEncoder 场景：BERT 明显更好**
   - CrossEncoder 依赖 `[CLS]` 对「整句拼接输入」的联合建模
   - `bert-base-chinese` 与 `BertModel` 原生对齐，segment embedding、tokenizer 行为一致
   - `chinese-roberta-wwm-ext` 虽可加载为 BertModel，但预训练目标（MLM+NSP vs MLM+WWM）、词表与输入分布与 BERT 不同，**在句对交互任务上未必更优**
   - 4 层截断 + 3 epoch 可能不足以让 RoBERTa 在 CrossEncoder 设定下充分微调

3. **阈值差异反映分数分布不同**
   - RoBERTa Cosine 最优阈值 0.70（BERT 0.63）→ 正例相似度整体偏低或分布更分散
   - RoBERTa Triplet 阈值 0.52（BERT 0.55）→ Triplet 训练后分数尺度略有偏移

4. **AUC 与 F1 不完全一致**
   - RoBERTa Cosine：F1 略高但 AUC 略低 → 固定阈值下分类更好，但**整体排序能力**未必更强
   - RoBERTa Triplet：F1 与 AUC 均略优于 BERT → Triplet + RoBERTa 组合在 BiEncoder 下略占优

### 8.7 结论与建议

| 结论 | 说明 |
|------|------|
| BiEncoder：RoBERTa ≈ BERT | F1 差距 < 0.2%，Triplet 略优于 Cosine，两种 Loss 差距仍很小 |
| CrossEncoder：BERT >> RoBERTa | BERT 领先约 **1.97% F1**，是本实验最大差异点 |
| 最优方案取决于 backbone | BERT 选 CrossEncoder（0.8807）；RoBERTa 选 BiEncoder Triplet（0.8694） |
| 不能盲目替换预训练模型 | 更强/更新的预训练模型在特定下游架构上不一定更好 |

**实践建议：**

1. **当前 bq_corpus + 4 层 + 3 epoch 设定下，优先使用 BERT + CrossEncoder**（验证 F1 最高 0.8807）
2. 若必须用 RoBERTa 或需要向量检索（BiEncoder 部署），选 **BiEncoder + Triplet**
3. 若 RoBERTa CrossEncoder 仍想提升，可尝试：**全 12 层**、**增加 epoch**、**调低 lr** 或 **增大 max_length**
4. Cosine vs Triplet 在两种 backbone 下差距均 < 0.2%，课堂/快速实验任选其一即可

### 8.8 汇总图（文字版）

```
验证集 F1 对比
────────────────────────────────────────────────
CrossEncoder      ████████████████████  BERT 0.8807
                  ██████████████████    RoBERTa 0.8610  (↓1.97%)

BiEncoder Triplet ███████████████████   BERT 0.8685
                  ███████████████████   RoBERTa 0.8694  (↑0.09%)

BiEncoder Cosine  ███████████████████   BERT 0.8672
                  ███████████████████   RoBERTa 0.8691  (↑0.19%)
────────────────────────────────────────────────
```

**一句话总结：** 在本文本匹配任务中，RoBERTa 并未全面超越 BERT——BiEncoder 两者相当，CrossEncoder 则 BERT 显著更强；选模型时应同时考虑**任务架构**与**预训练模型的匹配**，而非只看预训练规模或名气。
