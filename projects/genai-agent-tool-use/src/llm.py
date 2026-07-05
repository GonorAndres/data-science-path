"""Thin Ollama client."""

import os

import requests


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")


def chat(messages: list[dict], temperature: float = 0.1, max_tokens: int = 1024) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]
