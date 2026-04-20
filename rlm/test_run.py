"""Quick test: verify RLM works end-to-end with Anthropic."""
import os
from dotenv import load_dotenv
from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="anthropic",
    backend_kwargs={
        "model_name": "claude-sonnet-4-20250514",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
    },
    environment="local",
    max_depth=1,
    max_iterations=10,
    verbose=True,
    logger=logger,
)

# Simple test with a context that forces the model to use the REPL
context = [
    "Document 1: The capital of France is Paris. It was founded in the 3rd century BC.",
    "Document 2: The capital of Japan is Tokyo. It became the capital in 1868.",
    "Document 3: The capital of Brazil is Brasilia. It was built in 1960.",
    "Document 4: The capital of Australia is Canberra. It became the capital in 1913.",
    "Document 5: The capital of Canada is Ottawa. It was chosen as capital in 1857.",
]

result = rlm.completion(
    prompt=context,
    root_prompt="Which capital city is the newest (most recently founded/became capital)?",
)

print("\n\n=== FINAL RESULT ===")
print(result.response)
print(f"\nExecution time: {result.execution_time:.2f}s")
print(f"Usage: {result.usage_summary.to_dict()}")
