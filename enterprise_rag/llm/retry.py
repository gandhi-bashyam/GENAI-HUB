# enterprise_rag/llm/retry.py

import time

def retry(func, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    raise Exception("❌ All retries failed")