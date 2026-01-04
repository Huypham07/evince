import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from evince folder or project root
EVINCE_DIR = Path(__file__).resolve().parent.parent  # evince/
PROJECT_ROOT = EVINCE_DIR.parent  # esg_pipeline/

# Try evince/.env first, then src/.env, then project root
ENV_PATHS = [
    EVINCE_DIR / '.env',
    PROJECT_ROOT / 'src' / '.env',
    PROJECT_ROOT / '.env'
]

for env_path in ENV_PATHS:
    if env_path.exists():
        load_dotenv(env_path)
        break

class Config:
    # Google Gemini
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    REQUEST_INTERVAL = float(os.getenv("REQUEST_INTERVAL", "10.0"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

    # Qwen (self-hosted)
    QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "")
    QWEN_AUTH_USERNAME = os.getenv("QWEN_AUTH_USERNAME", "")
    QWEN_AUTH_PASSWORD = os.getenv("QWEN_AUTH_PASSWORD", "")
    QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen3-14B")
    QWEN_MAX_TOKENS = 16000
    QWEN_REQUEST_INTERVAL = 3.0

    # AWS Bedrock
    AWS_BEDROCK_REGION = os.getenv("AWS_BEDROCK_REGION", "us-east-1")
    BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "claude-haiku")
    BEDROCK_MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "4096"))
    
    # LLM Provider selection: "qwen", "bedrock", "gemini"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")

    @staticmethod
    def validate():
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

