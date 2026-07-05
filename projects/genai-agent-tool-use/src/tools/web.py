"""Web fetch tool with allowlist."""

from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


ALLOWLIST = {"es.wikipedia.org", "en.wikipedia.org"}
MAX_CHARS = 8000


def web_fetch(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed.")
    if parsed.netloc not in ALLOWLIST:
        raise PermissionError(f"Domain {parsed.netloc} not in allowlist.")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text[:MAX_CHARS]
