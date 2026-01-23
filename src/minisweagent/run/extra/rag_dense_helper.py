import os
import requests

NVIDIA_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
NVIDIA_EMBED_MODEL = "nvidia/nv-embed-v1"

def nvidia_embed(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("Missing NVIDIA_API_KEY env var")

    prefix = "query: " if input_type == "query" else "passage: "
    inputs = [prefix + t for t in texts]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": NVIDIA_EMBED_MODEL,
        "input": inputs,
        "encoding_format": "float",
    }

    r = requests.post(NVIDIA_EMBED_URL, headers=headers, json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()
    # OpenAI-style response: {"data":[{"embedding":[...], ...}, ...]}
    return [row["embedding"] for row in data["data"]]
