from llama_cpp import Llama

llm = Llama(
    model_path="./models/deepseek_engine.gguf",
    n_gpu_layers=20, # Try 20
    # verbose=True
)

output = llm("Q: Name the planets in the solar system. A: ", max_tokens=32)
print(output)