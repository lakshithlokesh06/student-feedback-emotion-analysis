"""Opaque deterministic IDs, stable within a source snapshot and across filtering."""
import hashlib
import json
import pandas as pd


def dataset_identity(data: pd.DataFrame, source: str, revision: str = '') -> str:
    digest = hashlib.sha256(json.dumps([source, revision, list(data.columns)], ensure_ascii=False).encode('utf-8'))
    digest.update(pd.util.hash_pandas_object(data, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def row_identifiers(data: pd.DataFrame, source: str, identity: str, feedback_column: str) -> list[str]:
    hashes = pd.util.hash_pandas_object(data, index=False).tolist()
    seen, output = {}, []
    identifiers = data['feedback_id'].tolist() if 'feedback_id' in data else [None] * len(data)
    for row_hash, identifier, feedback in zip(hashes, identifiers, data[feedback_column]):
        occurrence = seen.get(row_hash, 0)
        seen[row_hash] = occurrence + 1
        value = [source, identity, str(identifier), str(feedback), row_hash, occurrence]
        output.append(hashlib.sha256(json.dumps(value, ensure_ascii=False).encode('utf-8')).hexdigest())
    return output
