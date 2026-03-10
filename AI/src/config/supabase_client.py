import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

print("ENV PATH:", ENV_PATH)
print("SUPABASE_URL:", SUPABASE_URL)
print("SUPABASE_URL loaded:", bool(SUPABASE_URL))
print("SUPABASE_ANON_KEY loaded:", bool(SUPABASE_ANON_KEY))

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError(".env 파일에서 Supabase 환경변수를 읽지 못했습니다.")

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)