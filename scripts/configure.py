"""Generate local dev bootstrap/signing secrets without displaying them."""
from pathlib import Path
import secrets
from dotenv import dotenv_values
from app.config import ROOT

path = ROOT / '.env'
values = dict(dotenv_values(path if path.exists() else ROOT / '.env.example'))
for key in ['DEV_API_KEY', 'SESSION_SIGNING_KEY']:
    values[key] = values.get(key) or secrets.token_urlsafe(48)
path.write_text('\n'.join(k + '=' + (v or '') for k, v in values.items()) + '\n')
path.chmod(0o600)
print('本地 .env 已准备；请填写模型和 Jev 凭证。')
