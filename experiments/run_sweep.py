import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with open(args.manifest, encoding='utf-8-sig', newline='') as handle:
        entries = list(csv.DictReader(handle))
    results = []
    for entry in entries:
        result_path = Path(entry['result'])
        command = [sys.executable, str(root / 'train.py'), '--config', entry['config']]
        if not result_path.is_file():
            print(subprocess.list2cmdline(command), flush=True)
            if args.execute:
                subprocess.run(command, cwd=root, check=True)
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding='utf-8'))
            result.pop('counts', None)
            results.append(result)
    if results:
        output = Path(args.results)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    print(f'Collected {len(results)} of {len(entries)} results.')
