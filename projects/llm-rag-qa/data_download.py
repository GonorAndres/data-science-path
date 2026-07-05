"""Download a small Spanish Wikipedia corpus for the RAG demo.

Targets pages relevant to Mexican / Spanish constitutional and political topics.
Edit TITLES to broaden or narrow the corpus.
"""

import json
from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent / "data"
TITLES = [
    "Constitucion Politica de los Estados Unidos Mexicanos",
    "Division de poderes",
    "Congreso de la Union (Mexico)",
    "Presidente de Mexico",
    "Suprema Corte de Justicia de la Nacion",
    "Derechos humanos en Mexico",
    "Federalismo en Mexico",
    "Sistema electoral de Mexico",
    "Tratado entre Mexico, Estados Unidos y Canada",
    "Historia constitucional de Mexico",
]


def fetch_wikipedia(title: str, lang: str = "es") -> str | None:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "true",
        "titles": title,
        "format": "json",
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        if "extract" in page and page["extract"]:
            return page["extract"]
    return None


def download() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    corpus = []
    for title in TITLES:
        print(f"fetching {title}")
        text = fetch_wikipedia(title)
        if text:
            corpus.append({"title": title, "text": text})
    out = DATA_DIR / "corpus.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for entry in corpus:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"wrote {len(corpus)} documents to {out}")


if __name__ == "__main__":
    download()
