"""Ollama client wrapper for the generator."""

import os

import requests


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")


def generate(prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"]
