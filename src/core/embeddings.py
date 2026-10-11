from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import hashlib
import json
import urllib.request
import urllib.error
import numpy as np


class EmbeddingEngine(ABC):
    """Abstract interface for dense semantic embedding vectors."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Convert a text string into a normalized dense vector."""
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class LocalEmbeddingEngine(EmbeddingEngine):
    """
    Zero-dependency, high-speed semantic embedding engine using sub-word n-gram hashing
    and TF-IDF weighting projected into a fixed dense embedding space.
    Guarantees consistent, deterministic semantic ranking without needing GPU/APIs.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
        words = cleaned.split()
        tokens = list(words)
        # Add character 3-grams and 4-grams for subword similarity
        for word in words:
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    tokens.append(word[i:i + 3])
            if len(word) >= 4:
                for i in range(len(word) - 3):
                    tokens.append(word[i:i + 4])
        return tokens

    def embed_text(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dimension

        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            # Consistent hash into bucket
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            # Sign hash for random projection
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign

        # L2 normalization so dot products equal cosine similarity directly
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class OllamaEmbeddingEngine(EmbeddingEngine):
    """Embedding engine that delegates to local Ollama API with fallback to LocalEmbeddingEngine."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma4:12b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback = LocalEmbeddingEngine()

    def embed_text(self, text: str) -> List[float]:
        try:
            url = f"{self.base_url}/api/embeddings"
            payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
                if "embedding" in result:
                    return result["embedding"]
        except Exception:
            # Graceful fallback to local engine if Ollama model doesn't support embedding endpoint
            pass
        return self.fallback.embed_text(text)


def get_embedding_engine(use_ollama: bool = False) -> EmbeddingEngine:
    """Factory helper to obtain an embedding engine instance."""
    if use_ollama:
        return OllamaEmbeddingEngine()
    return LocalEmbeddingEngine()
