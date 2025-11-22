"""
示例：如何在代码的其他地方使用 LLM 客户端

这个文件展示了如何使用单例的 OpenRouterClient 来调用不同的大模型
"""
from llm_client import get_llm_client


def example_translation():
    """示例：翻译任务"""
    client = get_llm_client()
    
    messages = [
        {"role": "system", "content": "You are a translator."},
        {"role": "user", "content": "Translate 'Hello, world!' to Chinese."}
    ]
    
    result = client.chat_completion(
        messages=messages,
        model="google/google/gemini-2.5-flash-lite",
        temperature=0.2
    )
    
    print(f"Translation result: {result}")


def example_summarization():
    """示例：文本摘要任务"""
    client = get_llm_client()
    
    long_text = """
    The quick brown fox jumps over the lazy dog. 
    This is a sample text that needs to be summarized.
    It contains multiple sentences and various information.
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
        {"role": "user", "content": f"Summarize the following text in one sentence:\n\n{long_text}"}
    ]
    
    result = client.chat_completion(
        messages=messages,
        model="anthropic/claude-3.5-sonnet",  # 使用更强大的模型
        temperature=0.3,
        max_tokens=100
    )
    
    print(f"Summary: {result}")


def example_code_analysis():
    """示例：代码分析任务"""
    client = get_llm_client()
    
    code = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """
    
    messages = [
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": f"Review this code and suggest improvements:\n\n{code}"}
    ]
    
    result = client.chat_completion(
        messages=messages,
        model="openai/gpt-4o",  # 使用 GPT-4o
        temperature=0.5
    )
    
    print(f"Code review: {result}")


def example_multi_turn_conversation():
    """示例：多轮对话"""
    client = get_llm_client()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is its population?"}
    ]
    
    result = client.chat_completion(
        messages=messages,
        model="google/google/gemini-2.5-flash-lite",
        temperature=0.7
    )
    
    print(f"Response: {result}")


def example_with_timeout_retry():
    """示例：自定义超时和重试"""
    client = get_llm_client()
    
    messages = [
        {"role": "user", "content": "Tell me a joke."}
    ]
    
    result = client.chat_completion(
        messages=messages,
        model="google/google/gemini-2.5-flash-lite",
        temperature=0.9,
        timeout=10,  # 10秒超时
        max_retries=5,  # 最多重试5次
        sort_by_latency=True  # 优先选择最快的提供商
    )
    
    print(f"Joke: {result}")


if __name__ == "__main__":
    print("=" * 60)
    print("LLM 客户端使用示例")
    print("=" * 60)
    
    # 可以调用不同的示例函数
    example_translation()
    
    # 注意：多个函数都使用同一个 LLM 客户端实例（单例模式）
    # 它们共享长连接，提高效率
