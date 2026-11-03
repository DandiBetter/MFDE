import argparse
import copy
import csv
import json
from pathlib import Path


AXES = {'num_prototypes': [8, 16, 32, 64, 128], 'routing_temperature': [0.025, 0.05, 0.1, 0.2, 0.4], 'lambda_reg_effective': [0, 0.25, 0.5, 1, 2]}
BASE = {'num_prototypes': 32, 'routing_temperature': 0.1, 'lambda_reg_effective': 1}


def generate(config, output, seeds):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    combinations = {tuple(BASE.values()): 'baseline'}
    for axis, values in AXES.items():
        for value in values:
            setting = {**BASE, axis: value}
            combinations.setdefault(tuple(setting.values()), f'{axis}_{value}')
    rows = []
    for values, setting_id in combinations.items():
        for seed in seeds:
            current = copy.deepcopy(config)
            current['num_prototypes'] = values[0]
            current['objective']['routing_temperature'] = values[1]
            current['objective']['lambda_reg_effective'] = values[2]
            current['seed'] = seed
            current['output_dir'] = str(output / 'runs' / f'{setting_id}_seed{seed}')
            path = output / f'{setting_id}_seed{seed}.json'
            path.write_text(json.dumps(current, indent=2), encoding='utf-8')
            rows.append({'config': str(path), 'result': str(Path(current['output_dir']) / 'result.json'), 'setting': setting_id, 'seed': seed})
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    args = parser.parse_args()
    rows = generate(json.loads(Path(args.config).read_text(encoding='utf-8')), args.output, args.seeds)
    print(f'Generated {len(rows)} runs. No training was started.')
