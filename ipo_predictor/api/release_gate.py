"""Require an explicit reviewed release tied to exact local model artifacts."""
import hashlib
import json
from pathlib import Path


def verify_release(directory, stage):
    if stage not in {'pre_demand', 'post_demand'}:
        raise ValueError('unsupported_stage')
    root = Path(directory)
    content = (root / 'production_release.json').read_bytes()
    manifest = json.loads(content)
    if not isinstance(manifest, dict) or manifest.get('approved') is not True or manifest.get('prediction_stage') != stage:
        raise ValueError('release_not_approved')
    for key in ('release_id', 'reviewed_by', 'validation_report'):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError('missing_review_evidence')
    expected = {f'{stage}_{target}_v1{suffix}' for target in ('open', 'close')
                for suffix in ('.pkl', '_meta.json')}
    hashes = manifest.get('artifact_sha256', {})
    if not isinstance(hashes, dict) or set(hashes) != expected:
        raise ValueError('artifact_set_mismatch')
    for name in sorted(expected):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError('artifact_hash_mismatch')
    return hashlib.sha256(content).hexdigest()
