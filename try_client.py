from client import LLMClient

client = LLMClient()
result = client.complete("Say hello in one sentence", max_tokens=50)

print(result.text)
print("input tokens:", result.input_tokens)
print("output tokens:", result.output_tokens)
print(f"latency: {result.latency_ms:.0f} ms")