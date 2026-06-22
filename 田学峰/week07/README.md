outputs_pd 目录说明
本目录存放 peoples_daily 数据集（人民日报 NER）的训练产出，与 cluener 的 outputs/ 目录完全独立。

目录结构
outputs_pd/
├── checkpoints/          # 模型权重（需训练后生成）
│   ├── best_pd_linear.pt # BERT+Linear 最优 checkpoint
│   └── best_pd_crf.pt    # BERT+CRF 最优 checkpoint
├── logs/                  # 训练/评估日志
│   ├── train_pd_linear.json         # BERT+Linear 训练日志
│   ├── train_pd_crf.json            # BERT+CRF 训练日志
│   ├── eval_pd_linear_test.json     # BERT+Linear 测试集评估
│   ├── eval_pd_crf_test.json       # BERT+CRF 测试集评估
│   ├── eval_pd_linear_validation.json  # BERT+Linear 验证集评估
│   └── eval_pd_crf_validation.json     # BERT+CRF 验证集评估
└── figures/               # 数据探索可视化
    ├── pd_entity_distribution.png        # 实体类型频次分布
    ├── pd_text_length_distribution.png   # 文本长度分布
    └── pd_entity_length_distribution.png # 实体长度分布
实验结果（3 epoch，bert-base-chinese）
模型	评估集	Precision	Recall	F1	非法序列
BERT + Linear	test	0.9342	0.9261	0.9301	21 条
BERT + CRF	test	0.9412	0.9345	0.9378	0 条
数据集特点
实体类型：3 类（PER 人名 / ORG 组织机构 / LOC 地名）
训练集 20864 条，验证集 2318 条，测试集 4636 条
实体平均长度 3.2 字，CRF 对短实体边界识别优势更明显
