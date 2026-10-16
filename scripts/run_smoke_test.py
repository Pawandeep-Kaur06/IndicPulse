import json, sys, os
# Ensure repository root is on sys.path for imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from src.pipeline import IndicPulsePipeline

pipeline = IndicPulsePipeline()

test_cases = [
    "I love the new update!",  # English
    "tuc kyu ro rhe ho?",     # Hinglish question
    "yeh movie bohot mast thi", # Hinglish statement
    "आज मौसम बहुत सुहाना है",   # Devanagari Hindi
    "",                       # Empty input
]

output_path = os.path.join(os.path.dirname(__file__), "smoke_test_results.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for msg in test_cases:
        result = pipeline.predict(msg, detailed=True)
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
print("Smoke test completed.")
