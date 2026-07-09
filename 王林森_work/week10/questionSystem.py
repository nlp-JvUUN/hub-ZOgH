import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# --------------------------1.本地中草药知识库---------------------------
init_herb_db = {
    "黄连": {
        "性味": "苦寒",
        "归经": "心、肝、胃、大肠经",
        "功效": "清热燥湿，泻火解毒",
        "主治": "湿热痞满，呕吐吞酸，泻痢，黄疸，高热神昏，心火亢盛",
        "禁忌": "脾胃虚寒者慎用"
    },
    "人参": {
        "性味": "甘、微苦，微温",
        "归经": "脾、肺、心经",
        "功效": "大补元气，复脉固脱，补脾益肺，生津安神",
        "主治": "体虚欲脱，肢冷脉微，脾虚食少，肺虚喘咳，津伤口渴",
        "禁忌": "实证、热证而正气不虚者忌服，不宜与藜芦同用"
    },
    "金银花": {
        "性味": "甘寒",
        "归经": "肺、心经",
        "功效": "清热解毒，疏散风热",
        "主治": "痈肿疔疮，喉痹，丹毒，热毒血痢，风热感冒，温病发热",
        "禁忌": "脾胃虚寒、气虚疮疡脓清者忌用"
    },
    "当归": {
        "性味": "甘、辛，温",
        "归经": "肝、心、脾经",
        "功效": "补血活血，调经止痛，润肠通便",
        "主治": "血虚萎黄，眩晕心悸，月经不调，经闭痛经，肠燥便秘",
        "禁忌": "湿盛中满、大便溏泄者慎用"
    }
}
DB_FILE = "herb_database.json"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(init_herb_db, f, ensure_ascii=False, indent=2)
        return init_herb_db

herb_db = load_db()
# 将知识库转为文本片段，用于向量相似度检索
docs = []
doc_texts = []
for name, info in herb_db.items():
    content = f"药材:{name}；性味:{info['性味']}；归经:{info['归经']}；功效:{info['功效']}；主治:{info['主治']}；禁忌:{info['禁忌']}"
    docs.append({"name": name, "content": content})
    doc_texts.append(content)

# --------------------------2.加载向量模型用于RAG检索---------------------------
embed_model = SentenceTransformer("all‑MiniLM‑L6‑v2")
doc_embeddings = embed_model.encode(doc_texts, convert_to_tensor=True)

# --------------------------3.加载基座大模型（本地基座）---------------------------
model_name = "Qwen/Qwen‑7B‑Chat"  # 可替换 deepseek‑ai/deepseek‑7b‑chat、meta‑llama/Llama‑3‑8B‑Instruct
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# --------------------------4.RAG检索函数，找到相关药材信息---------------------------
def get_relevant_context(query, top_k=2):
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, doc_embeddings)[0]
    top_results = torch.topk(scores, k=top_k)
    context = ""
    for idx in top_results.indices:
        context += doc_texts[idx.item()] + "\n"
    return context

# --------------------------5.构造Prompt交给基座模型推理---------------------------
def llm_answer(user_question):
    context = get_relevant_context(user_question)
    system_prompt = f"""你是专业的中医药顾问，根据提供的药材资料回答问题；
参考资料：
{context}
要求：
1. 如果参考资料中有内容优先使用资料作答；资料没有依靠你的知识库客观回答；
2. 回答末尾必须加上提示：本内容仅科普，用药请遵医嘱；
3. 回答简洁条理清晰。
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return response

# --------------------------6.主循环问答系统---------------------------
def main():
    print("=====基于本地基座大模型中草药问答系统=====")
    print("输入quit退出")
    while True:
        user_input = input("\n你的提问：")
        if user_input.strip().lower() == "quit":
            print("程序结束")
            break
        ans = llm_answer(user_input)
        print("\nAI回答：")
        print(ans)

if __name__ == "__main__":
    main()
