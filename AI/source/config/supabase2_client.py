import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

BASE_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

SUPABASE2_URL      = os.getenv("SUPABASE2_URL")
SUPABASE2_ANON_KEY = os.getenv("SUPABASE2_ANON_KEY")

if not SUPABASE2_URL or not SUPABASE2_ANON_KEY:
    raise ValueError(
        ".env 파일에 SUPABASE2_URL 또는 SUPABASE2_ANON_KEY 가 없습니다."
    )

supabase2 = create_client(SUPABASE2_URL, SUPABASE2_ANON_KEY)
