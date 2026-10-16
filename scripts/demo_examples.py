import json, os, sys

# Ensure repository root is on the import path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Force UTF‑8 output on Windows consoles
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.pipeline import IndicPulsePipeline

pipeline = IndicPulsePipeline()

examples = [
    "I love this product, it is awesome!",
    "tuc kyu ro rhe ho?",  # Hinglish question
    "तुम क्या कर रहे हो?",      # Devanagari question
    "Yeh service bahut slow hai and I am annoyed.",
    "Thanks for the quick help!",
]

for msg in examples:
    result = pipeline.predict(msg, detailed=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
