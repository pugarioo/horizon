import os
import sqlite3
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DB_DIR = BASE_DIR / "app" / "db"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
TARGET_DIR = MODELS_DIR / "all-MiniLM-L6-v2"

MODELS = {
    "phi_engine.gguf": "https://huggingface.co/bartowski/phi-4-mini-instruct-GGUF/resolve/main/phi-4-mini-instruct-Q4_K_M.gguf",
    "qwen_engine.gguf": "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "llama_engine.gguf": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "monolith_engine.gguf": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
}


def run_command(command: str, env: dict | None = None) -> None:
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    if process.stdout:
        for line in process.stdout:
            print(line, end="")
    process.wait()
    if process.returncode != 0:
        sys.exit(1)


def install_dependencies() -> None:
    run_command(f"pip install -r {REQUIREMENTS_FILE}")

    custom_env = os.environ.copy()
    custom_env["CUDACXX"] = "/usr/local/cuda/bin/nvcc"

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        sys.exit(1)

    custom_env["CC"] = f"{conda_prefix}/bin/x86_64-conda-linux-gnu-gcc"
    custom_env["CXX"] = f"{conda_prefix}/bin/x86_64-conda-linux-gnu-g++"
    custom_env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
    custom_env["FORCE_CMAKE"] = "1"

    run_command(
        "pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir",
        env=custom_env,
    )


def download_models() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for filename, url in MODELS.items():
        target_path = MODELS_DIR / filename
        if not target_path.exists():
            run_command(f"wget -O {target_path} {url}")


def download_embedding_model() -> None:
    try:
        import huggingface_hub
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"]
        )

    from huggingface_hub import snapshot_download

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=str(TARGET_DIR),
    )


def init_databases() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DB_DIR / "chat_history.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            name TEXT,
            timestamp INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            sent_by TEXT,
            content TEXT,
            timestamp INTEGER,
            FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
        )
    """)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    install_dependencies()
    download_models()
    download_embedding_model()
    init_databases()
