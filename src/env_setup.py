"""
Configures environment variables before any ML libraries are imported.
Must be imported before llm_sdk / transformers / huggingface_hub.
"""
import os

if os.path.isdir("/sgoinfre") and os.access("/sgoinfre", os.W_OK):
    os.environ["HF_HOME"] = (f"/sgoinfre/{os.environ.get('USER', 'default')}"
                             f"/huggingface")
else:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
