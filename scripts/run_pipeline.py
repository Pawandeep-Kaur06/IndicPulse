"""
CLI runner for the IndicPulse multi-module text understanding pipeline.

Usage:
    # Single message
    python scripts/run_pipeline.py --message "Mera refund abhi tak nahi aaya, bahut gussa hai."

    # Interactive mode
    python scripts/run_pipeline.py --interactive

    # Run default demo cases (English, Hinglish, Punglish)
    python scripts/run_pipeline.py --demo
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure workspace root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import IndicPulsePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="IndicPulse Unified Multilingual Pipeline (Language + Emotion + Intent)"
    )
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        help="Input text message to analyze",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive console loop",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo test cases across English, Hinglish, and Punglish",
    )
    return parser.parse_args()


DEMO_MESSAGES = [
    "Mera refund abhi tak nahi aaya, bahut gussa hai.",
    "Could you please reschedule my appointment to Saturday?",
    "Eh product bilkul bekaar aa, paise wapas karo!",
    "Thanks a lot for the quick support, really appreciate it!",
    "Kal exam kinne vaje shuru hona aa?",
    "Yeh design sach me bahut badhiya lag raha hai.",
    "Meeting kal subah 10 baje scheduled hai.",
]


def main():
    args = parse_args()
    print("Loading IndicPulse pipeline modules (Language, Emotion, Intent)...")
    pipeline = IndicPulsePipeline()
    print("Pipeline ready!\n")

    if args.message:
        result = pipeline.predict(args.message)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.interactive:
        print("--- IndicPulse Interactive Mode ---")
        print("Type a message in English, Hinglish, or Punglish (or 'exit' to quit):\n")
        while True:
            try:
                user_msg = input("\nEnter message: ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ("exit", "quit", "q"):
                    print("Exiting.")
                    break
                result = pipeline.predict(user_msg)
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
        return

    # Default or --demo: run test cases
    print("Running demo cases across English, Hinglish, and Punglish:\n" + "=" * 60)
    for msg in DEMO_MESSAGES:
        res = pipeline.predict(msg)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        print("-" * 60)


if __name__ == "__main__":
    main()
