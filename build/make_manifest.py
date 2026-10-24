"""Refresh local hashes without overwriting externally verified submission state.

CIERRE_AIMS.json is the factual source for platform receipts and settings.
This script never infers publication, acceptance or a score from local files.
"""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'CIERRE_AIMS.json'
manifest = json.loads(path.read_text(encoding='utf-8'))
manifest['generated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
manifest['repository']['local_head_at_generation'] = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
for artifact in manifest['artifacts'].values():
    if not isinstance(artifact, dict) or 'path' not in artifact:
        continue
    local_path = (ROOT / artifact['path']).resolve()
    if not local_path.is_relative_to(ROOT) or not local_path.is_file():
        raise ValueError('Artifact must be an existing file inside the repository')
    if 'sha256' in artifact:
        artifact['sha256'] = hashlib.sha256(local_path.read_bytes()).hexdigest()
    if 'bytes' in artifact:
        artifact['bytes'] = local_path.stat().st_size
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Local hashes refreshed; external state preserved.')
