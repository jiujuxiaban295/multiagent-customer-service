"""Local development bootstrap + signed session credential; no self-reported identity."""
import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from fastapi import HTTPException


@dataclass(frozen=True)
class Principal:
    owner: str
    scope: str
    conv_id: str


def issue_token(principal, key, ttl=86400):
    payload = {**principal.__dict__, 'expires': int(time.time()) + ttl}
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(',', ':')).encode()).decode().rstrip('=')
    signature = hmac.new(key.encode(), body.encode(), hashlib.sha256).hexdigest()
    return body + '.' + signature


def verify_token(authorization, key):
    try:
        scheme, token = authorization.split(' ', 1)
        body, signature = token.split('.')
        expected = hmac.new(key.encode(), body.encode(), hashlib.sha256).hexdigest()
        if scheme.lower() != 'bearer' or not key or not hmac.compare_digest(signature, expected):
            raise ValueError('invalid credential')
        payload = json.loads(base64.urlsafe_b64decode(body + '=' * (-len(body) % 4)))
        if payload['expires'] <= time.time():
            raise ValueError('expired')
        return Principal(payload['owner'], payload['scope'], payload['conv_id'])
    except (ValueError, KeyError, TypeError):
        raise HTTPException(401, '需要有效的会话凭证') from None
