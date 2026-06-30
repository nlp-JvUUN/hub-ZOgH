{
  "task": "中文文本语义匹配",
  "dataset": "bq_corpus",
  "dataset_desc": "银行客服问答语义匹配",
  "model": "bert-base-chinese (BiEncoder)",
  "num_hidden_layers": 1,
  "loss_fn": "CosineEmbeddingLoss",
  "pooling": "mean",
  "device": "CPU (18核)",

  "data_split": {
    "train": 68960,
    "val": 8620,
    "test": 8620
  },

  "train_config": {
    "batch_size": 8,
    "max_length": 32,
    "epochs": 2,
    "lr_bert": 2e-5,
    "lr_head": 1e-4,
    "warmup_ratio": 0.1,
    "margin": 0.3,
    "max_grad_norm": 1.0
  },

  "training_process": [
    {
      "epoch": 1,
      "train_loss": 0.2431,
      "val_acc": 0.7850,
      "val_f1": 0.7844,
      "threshold": 0.64,
      "elapsed_min": 31
    },
    {
      "epoch": 2,
      "train_loss": 0.2039,
      "val_acc": 0.8035,
      "val_f1": 0.8030,
      "threshold": 0.64,
      "elapsed_min": 25
    }
  ],
  "total_train_time_min": 56,

  "test_evaluation": {
    "accuracy": 0.7957,
    "f1": 0.7949,
    "auc": 0.8686,
    "best_threshold": 0.63,
    "classification_report": {
      "0_not_similar": {
        "precision": 0.83,
        "recall": 0.74,
        "f1": 0.78,
        "support": 4238
      },
      "1_similar": {
        "precision": 0.77,
        "recall": 0.85,
        "f1": 0.81,
        "support": 4382
      },
      "weighted_avg": {
        "precision": 0.80,
        "recall": 0.80,
        "f1": 0.79,
        "support": 8620
      }
    }
  },

  "cross_dataset_comparison": {
    "afqmc": {
      "train_size": 34334,
      "f1": 0.676,
      "auc": 0.822,
      "train_time_min": 30
    },
    "bq_corpus": {
      "train_size": 68960,
      "f1": 0.795,
      "auc": 0.869,
      "train_time_min": 56
    },
    "delta": {
      "f1": "+11.9%",
      "auc": "+4.7%",
      "train_size": "+34626 (2x)"
    }
  },

  "output_files": {
    "model": "outputs/checkpoints/biencoder_cosine_best.pt",
    "train_log": "outputs/logs/biencoder_cosine_log.json",
    "report": "outputs/bq_corpus_training_report.md",
    "figures": "outputs/figures/biencoder_test_sim_dist.png"
  }
}
