import os
import sys
import time
import openai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_util import load_config

# 加载配置文件
config = load_config()
generation_config = config.get('generation', {})
max_retries = generation_config.get('max_retries', 3)
retry_delay = generation_config.get('retry_delay', 1)

# 基础调用
def base_call_llm(
    api_key: str,
    base_url: str,
    model_name: str,
    prompt: list,
    max_tokens: int,
    n: int,
    temperature: float = None,   # ← 添加可选参数
):

    # 创建client
    client = openai.Client(
        api_key=api_key, base_url=base_url
    )

    for attempt in range(max_retries):
        try:
            # 构建参数字典
            kwargs = dict(
                model=model_name,
                messages=prompt,
                n=n,
            )
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            # 若指定 temperature 则加入 kwargs
            if temperature is not None:
                kwargs["temperature"] = temperature

            # 发送请求
            response = client.chat.completions.create(**kwargs)

            # 响应后处理
            return [x.message.content for x in response.choices]

        except Exception as e:
            print(f"[Attempt {attempt+1}/{max_retries}] LLM call failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                raise

# 根据配置文件调用  
def call_llm(
    prompt: list,
    model_name: str,
    n: int,
    temperature: float = None,   # ← 添加可选参数
):
    max_n = config['models'][model_name]['max_n']
    is_thinking = config['models'][model_name]['is_thinking']
    token_key = 'thinking_max_tokens' if is_thinking else 'chat_max_tokens'
    max_tokens = generation_config.get(token_key)

    if max_n < n:
        remaining = n
        all_results = []
        while remaining > 0:
            batch_n = min(remaining, max_n)

            batch_results = base_call_llm(
                api_key = config['models'][model_name]['api_key'],
                base_url = config['models'][model_name]['base_url'],
                model_name = config['models'][model_name]['model_name'],
                prompt = prompt,
                max_tokens = max_tokens,
                n = batch_n,
                temperature = temperature     # ← 向下传递
            )

            if isinstance(batch_results, list):
                all_results.extend(batch_results)
            else:
                all_results.append(batch_results)
            remaining -= batch_n
        return all_results

    return base_call_llm(
        api_key = config['models'][model_name]['api_key'],
        base_url = config['models'][model_name]['base_url'],
        model_name = config['models'][model_name]['model_name'],
        prompt = prompt,
        max_tokens = max_tokens,
        n = n,
        temperature = temperature           # ← 向下传递
    )
