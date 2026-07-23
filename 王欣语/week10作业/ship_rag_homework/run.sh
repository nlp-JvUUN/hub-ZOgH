#!/bin/bash
# 一键运行脚本：船舶术语 RAG 系统
# 使用方式：bash run.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  船舶术语 RAG 系统 - 一键运行"
echo "=========================================="

# 检查环境变量
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "错误：请设置环境变量 DASHSCOPE_API_KEY"
    echo "  export DASHSCOPE_API_KEY=sk-xxx"
    exit 1
fi

echo ""
echo "【步骤 1/4】数据解析：CSV → JSON"
python src/parse_csv.py

echo ""
echo "【步骤 2/4】文档分块：语义分块策略"
python src/chunk_documents.py

echo ""
echo "【步骤 3/4】构建向量索引"
python src/build_index.py

echo ""
echo "【步骤 4/4】运行评估"
cd evaluation
python evaluate.py --question-ids 1,2,3,4,5
cd ..

echo ""
echo "=========================================="
echo "  全部完成！"
echo "  现在可以运行交互式问答："
echo "    python src/rag_pipeline.py"
echo "=========================================="
