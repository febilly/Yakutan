import os
import sys

os.environ["YAKUTAN_LOCAL_ASR_UI"] = "1"

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    from local_asr.model_manager import apply_cache_env, prepare_qwen_llama_runtime_env

    apply_cache_env()
    prepare_qwen_llama_runtime_env()
