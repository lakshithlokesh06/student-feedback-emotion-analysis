"""Lazy local model loading; no torch/transformers import during app startup."""
import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from src.config import MAX_MODEL_TOKENS, MODEL_CACHE_DIR, MODEL_NAME, MODEL_REVISION
from src.emotion.labels import EmotionError, label_names


@dataclass
class EmotionModel:
    tokenizer: Any
    model: Any
    metadata: dict
    lock: Any = field(default_factory=Lock)

    def predict(self, texts: list[str]) -> list[dict]:
        return self._predict(texts, detailed=False)

    def predict_detailed(self, texts: list[str]) -> list[dict]:
        return self._predict(texts, detailed=True)

    def _predict(self, texts: list[str], detailed: bool) -> list[dict]:
        import torch
        with self.lock, torch.inference_mode():
            inputs = self.tokenizer(texts, padding=True, truncation=True,
                                    max_length=self.metadata['max_tokens'], return_tensors='pt')
            probabilities = torch.softmax(self.model(**inputs).logits, dim=-1)
            confidence, indices = probabilities.max(dim=-1)
            output = [{'label': self.metadata['labels'][index], 'score': score}
                      for index, score in zip(indices.tolist(), confidence.tolist())]
            if detailed:
                lengths = self.tokenizer(texts, truncation=False, add_special_tokens=True,
                                         return_length=True, verbose=False)['length']
                for row, scores, length in zip(output, probabilities.tolist(), lengths):
                    row['scores'] = dict(zip(self.metadata['labels'], scores))
                    row['token_length'] = length
                    row['was_truncated'] = length > self.metadata['max_tokens']
            return output


def load_model() -> EmotionModel:
    # Keep download-related caches in the project as well as model weights.
    os.environ.setdefault('HF_HOME', str(MODEL_CACHE_DIR))
    os.environ.setdefault('HF_XET_CACHE', str(MODEL_CACHE_DIR / 'xet'))
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except (ImportError, OSError) as exc:
        raise EmotionError('Model dependencies are unavailable. Install requirements.txt in your active Python environment and restart the app.') from exc
    try:
        kwargs = dict(revision=MODEL_REVISION, cache_dir=str(MODEL_CACHE_DIR), trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **kwargs)
        model.to('cpu').eval()
        labels = label_names(model.config.id2label)
        max_tokens = min(MAX_MODEL_TOKENS, tokenizer.model_max_length)
        metadata = {
            'name': MODEL_NAME, 'source': f'https://huggingface.co/{MODEL_NAME}',
            'provider': 'Jochen Hartmann / Hugging Face', 'task': 'text-classification',
            'labels': labels, 'device': 'cpu', 'max_tokens': max_tokens,
            'truncation': True, 'revision': getattr(model.config, '_commit_hash', None) or MODEL_REVISION,
            'confidence': 'Top-class softmax probability; not calibrated accuracy.',
        }
        return EmotionModel(tokenizer, model, metadata)
    except EmotionError:
        raise
    except Exception as exc:
        raise EmotionError('Could not load the emotion model. Check your internet connection, available disk space, and access to Hugging Face, then retry. Cached model files are stored in .model_cache inside this project.') from exc
