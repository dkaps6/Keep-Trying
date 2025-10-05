# scripts/retry_http.py
from __future__ import annotations
import time, requests

class HTTPError(Exception): ...

def get(url, *, params=None, headers=None, timeout=15, max_attempts=5, backoff=0.75):
    attempts = 0
    while True:
        attempts += 1
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429,) or r.status_code >= 500:
                raise HTTPError(f"status {r.status_code}")
            return r
        except Exception as e:
            if attempts >= max_attempts:
                raise
            sleep = backoff * (2 ** (attempts - 1))
            print(f"[retry_http] attempt {attempts} failed: {e}; sleeping {sleep:.2f}s")
            time.sleep(sleep)
