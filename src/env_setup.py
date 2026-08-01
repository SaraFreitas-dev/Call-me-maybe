"""
Configures environment variables before any ML libraries are imported.
Must be imported before llm_sdk / transformers / huggingface_hub.
"""
import os

sgoinfre_dir = f"/sgoinfre/{os.environ.get('USER', 'default')}/huggingface"
try:
    os.makedirs(sgoinfre_dir, exist_ok=True)
    os.environ["HF_HOME"] = sgoinfre_dir
except OSError:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
