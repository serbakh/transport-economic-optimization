"""Check the project, its links and unchanged experiment artifacts."""
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

def verify():
    errors = []
    checked = 0
    output = ROOT/'results/conditions_contract_ml_2026_11_07'
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8-sig'))
    for folder, entries in [(ROOT,manifest['source_sha256']), (output,manifest['output_sha256'])]:
        for name,digest in entries.items():
            path = folder/name.replace('\\','/')
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append(f'Artifact mismatch: {name}')
            checked += 1
    for line in (ROOT/'baseline/SHA256SUMS.txt').read_text(encoding='utf-8-sig').splitlines():
        digest,name = line.split(maxsplit=1)
        path = ROOT/'baseline'/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f'Baseline mismatch: {name}')
        checked += 1
    for folder in ['src','tests','tools','baseline/src']:
        for path in (ROOT/folder).rglob('*.py'):
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'))
            except SyntaxError as exc:
                errors.append(f'{path.relative_to(ROOT)}: {exc}')
    links = 0
    for path in [*ROOT.glob('*.md'), *(ROOT/'docs').glob('*.md'), ROOT/'baseline/README.md', ROOT/'demo/README.md']:
        for href in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8-sig')):
            href = href.strip('<>').split('#')[0]
            if not href or re.match(r'[a-zA-Z]+:',href):
                continue
            links += 1
            if not (path.parent/unquote(href)).exists():
                errors.append(f'Broken link: {path.relative_to(ROOT)} -> {href}')
    return dict(artifact_hashes_checked=checked,markdown_links_checked=links,errors=errors,passed=not errors)

if __name__ == '__main__':
    report = verify()
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(not report['passed'])
