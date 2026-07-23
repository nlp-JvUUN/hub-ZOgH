"""
文档分块模块。
将 raw/ 下的 txt 文件按固定大小（500字）切割，overlap 50 字。
参考课程项目的语义分块思路：保持简单，用固定大小作为 baseline。
"""
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CHUNK_DIR = BASE_DIR / "data" / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500     # 每块字符数
OVERLAP = 50         # 重叠字符数


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    将文本按段落边界优先切分，确保每块不超 chunk_size。
    若某段超过 chunk_size，则用滑动窗口切。
    """
    paragraphs = text.split("\n")
    chunks = []
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果合并后不超长，就累积
        if len(buffer) + len(para) + 1 <= chunk_size:
            buffer = (buffer + "\n" + para).strip() if buffer else para
        else:
            # buffer 先输出
            if buffer:
                chunks.append(buffer)

            # 当前段太长，用滑动窗口切
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    chunks.append(para[start:end])
                    start = end - overlap
                buffer = ""
            else:
                buffer = para

    if buffer:
        chunks.append(buffer)

    return chunks


def main():
    print("=" * 60)
    print("文档分块")
    print(f"策略: 固定大小 chunk_size={CHUNK_SIZE} overlap={OVERLAP}")
    print("=" * 60)

    all_chunks = []
    chunk_id = 0

    for txt_file in sorted(RAW_DIR.glob("*.txt")):
        print(f"处理: {txt_file.name}")
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        # 去掉第一行标题（# ...）
        lines = text.split("\n")
        title_line = lines[0].lstrip("# ").strip() if lines else txt_file.stem
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""

        chunks = split_text(body)
        print(f"  → {len(chunks)} 个 chunks")

        for chunk_text in chunks:
            if len(chunk_text.strip()) < 30:
                continue  # 太短的块跳过
            chunk_id += 1
            all_chunks.append({
                "chunk_id": f"lol_{chunk_id:04d}",
                "content": chunk_text.strip(),
                "metadata": {
                    "source": txt_file.name,
                    "title": title_line,
                    "chunk_index": len(all_chunks),
                    "char_count": len(chunk_text),
                },
            })

    # 保存
    output_path = CHUNK_DIR / "all_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n总计: {chunk_id} 个 chunks")
    print(f"保存到: {output_path}")


if __name__ == "__main__":
    main()
