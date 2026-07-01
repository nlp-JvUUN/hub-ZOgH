import os, time, sys
os.environ["TMP"] = r"C:\tmp_vllm"
os.environ["TEMP"] = r"C:\tmp_vllm"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from openai import OpenAI
    print("openai OK:", OpenAI.__version__ if hasattr(OpenAI, '__version__') else "loaded")
except ImportError as e:
    print(f"openai FAILED: {e}")
    sys.exit(1)

client = OpenAI(api_key="EMPTY", base_url="http://localhost:11434/v1")
MODEL = "qwen2.5:0.5b"

print(f"Testing Ollama API...")
t0 = time.time()
try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
    )
    elapsed = time.time() - t0
    content = resp.choices[0].message.content
    print(f"SUCCESS: '{content}' ({elapsed:.2f}s)")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
