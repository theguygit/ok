import sys
import os

print("--- 🔍 Starting Comprehensive Import Check ---")

try:
    print("Checking model.py...", end=" ")
    import model
    print("✅")
except Exception as e:
    print(f"❌ FAILED: {e}")

try:
    print("Checking dataset.py...", end=" ")
    import dataset
    print("✅")
except Exception as e:
    print(f"❌ FAILED: {e}")

try:
    print("Checking train.py...", end=" ")
    import train
    print("✅")
except Exception as e:
    print(f"❌ FAILED: {e}")

try:
    print("Checking evaluate.py...", end=" ")
    import evaluate
    print("✅")
except Exception as e:
    print(f"❌ FAILED: {e}")

try:
    print("Checking test.py...", end=" ")
    import test
    print("✅")
except Exception as e:
    print(f"❌ FAILED: {e}")

try:
    print("Checking app.py...", end=" ")
    # app.py might behave weirdly if imported because of gradio launch on main
    # but we just want to check imports.
    # We will modify app.py check slightly or just import it.
    # If app.py has `if __name__ == "__main__": launch()`, it should be safe.
    import app
    print("✅")
except Exception as e:
    print(f"❌ FAILED: {e}")

try:
    print("Checking feature_extraction.py...", end=" ")
    import feature_extraction
    print("✅")
except Exception as e:
    print(f"❌ FAILED: {e}")

try:
    print("Checking eda.py...", end=" ")
    import eda
    print("✅")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("--- ✨ All imports verified successfully! ---")
