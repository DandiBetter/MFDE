import argparse
import itertools
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AXES = {'num_prototypes': [8, 16, 32, 64, 128], 'routing_temperature': [0.025, 0.05, 0.1, 0.2, 0.4], 'lambda_reg_effective': [0, 0.25, 0.5, 1, 2]}
BASE = {'num_prototypes': 32, 'routing_temperature': 0.1, 'lambda_reg_effective': 1}
LABELS = {'num_prototypes': 'Prototype count K', 'routing_temperature': 'Routing-target temperature tau', 'lambda_reg_effective': 'Effective regularization weight'}
DATASETS = ['Camelyon', 'PANDA', 'HER2']
TASK_COUNTS = {'Camelyon': 3, 'PANDA': 2, 'HER2': 2}


def validate(frame):
    required = {'dataset', 'task', 'seed', 'source_val_auc', 'target_auc', 'split_hash', 'protocol_hash', 'implementation', 'selection', 'pool', *AXES}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError('Results are incomplete. Every required field must contain a measured value or its provenance.')
    if not set(frame['dataset']).issubset(DATASETS):
        raise ValueError('Dataset names must be Camelyon, PANDA or HER2.')
    if set(frame['selection']) != {'source_validation'}:
        raise ValueError('Only source-validation-selected results are accepted.')
    if not frame[['source_val_auc', 'target_auc']].apply(lambda col: col.between(0, 1).all()).all():
        raise ValueError('AUC must be expressed on the zero-to-one scale.')
    if frame.duplicated(['dataset', 'task', 'seed', *AXES]).any():
        raise ValueError('Duplicate run records were found.')
    changed = sum((~np.isclose(frame[key], default)).astype(int) for key, default in BASE.items())
    if (changed > 1).any():
        raise ValueError('Only one parameter may differ from the baseline in each run.')
    for dataset, group in frame.groupby('dataset'):
        if group['task'].nunique() != TASK_COUNTS[dataset] or group['seed'].nunique() != 3:
            raise ValueError(f'{dataset} requires all source tasks and exactly three common seeds.')
        if group['implementation'].nunique() != 1 or group['pool'].nunique() != 1:
            raise ValueError('Do not mix implementations or model pools within a sensitivity study.')
        for _, runs in group.groupby(['task', 'seed']):
            if runs['split_hash'].nunique() != 1 or runs['protocol_hash'].nunique() != 1:
                raise ValueError('Fixed factors or split assignments change across sweep settings.')
        expected = set(itertools.product(group['task'].unique(), group['seed'].unique()))
        for axis, values in AXES.items():
            part = group
            for other, default in BASE.items():
                if other != axis:
                    part = part[np.isclose(part[other], default)]
            if not set(np.round(part[axis], 8)).issubset(set(np.round(values, 8))):
                raise ValueError('Unexpected sweep values.')
            for value in values:
                sample = part[np.isclose(part[axis], value)]
                if set(zip(sample['task'], sample['seed'])) != expected:
                    raise ValueError(f'Missing matched runs: {dataset}, {axis}={value}.')


def summarize(frame):
    validate(frame)
    records = []
    for dataset, group in frame.groupby('dataset'):
        for axis, values in AXES.items():
            part = group
            for other, default in BASE.items():
                if other != axis:
                    part = part[np.isclose(part[other], default)]
            for value in values:
                selected = part[np.isclose(part[axis], value)]
                by_seed = selected.groupby('seed')[['target_auc', 'source_val_auc']].mean()
                records.append({'dataset': dataset, 'parameter': axis, 'value': value, 'mean_auc': float(by_seed['target_auc'].mean()), 'sd_auc': float(by_seed['target_auc'].std(ddof=1)), 'mean_source_val_auc': float(by_seed['source_val_auc'].mean()), 'sd_source_val_auc': float(by_seed['source_val_auc'].std(ddof=1)), 'n_seeds': len(by_seed), 'n_source_tasks': selected['task'].nunique()})
    return pd.DataFrame(records)


def draw(summary, output, template=False, source_validation=False):
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.4), constrained_layout=True)
    colors = ['#236A9D', '#B2433C', '#2C805B']
    for row, dataset in enumerate(DATASETS):
        for column, (parameter, values) in enumerate(AXES.items()):
            ax = axes[row, column]
            ax.set_title(f'({chr(97 + row * 3 + column)}) {dataset}', fontsize=11)
            ax.set_xlabel(LABELS[parameter], fontsize=10)
            ax.set_ylabel('Source validation AUC' if source_validation else 'Target AUC at ID checkpoint', fontsize=9)
            if parameter != 'lambda_reg_effective':
                ax.set_xscale('log', base=2)
            ax.set_xticks(values, [str(value) for value in values])
            ax.grid(alpha=0.22)
            ax.axvline(BASE[parameter], linestyle=':', color='#777777', linewidth=1, label='Default setting')
            part = summary[(summary['dataset'] == dataset) & (summary['parameter'] == parameter)] if summary is not None else None
            if part is None or part.empty:
                ax.set_ylim(0, 1)
                ax.text(0.5, 0.5, 'AWAITING MEASURED RESULTS\nNo experimental curve', transform=ax.transAxes, ha='center', va='center', fontsize=10, color='#666666')
            else:
                part = part.sort_values('value')
                mean = 'mean_source_val_auc' if source_validation else 'mean_auc'
                sd = 'sd_source_val_auc' if source_validation else 'sd_auc'
                ax.errorbar(part['value'], part[mean], yerr=part[sd], fmt='o-', color=colors[row], capsize=3, linewidth=1.6, markersize=4, label='MFDE mean ± sample SD')
                row_data = summary[summary['dataset'] == dataset]
                lo, hi = float((row_data[mean] - row_data[sd]).min()), float((row_data[mean] + row_data[sd]).max())
                pad = max(0.005, (hi - lo) * 0.15)
                ax.set_ylim(max(0, lo - pad), min(1, hi + pad))
                ax.legend(fontsize=7, loc='best')
    if template:
        fig.suptitle('Sensitivity study layout — pending experiments', fontsize=14)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output', required=True)
    parser.add_argument('--template', action='store_true')
    args = parser.parse_args()
    if args.template:
        draw(None, args.output, template=True)
    else:
        if not args.input:
            parser.error('--input is required for measured plots.')
        summary = summarize(pd.read_csv(args.input))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output.with_name(output.name + '_summary.csv'), index=False)
        draw(summary, output)
        draw(summary, output.with_name(output.name + '_source_validation'), source_validation=True)
        analysis = []
        for (dataset, parameter), part in summary.groupby(['dataset', 'parameter']):
            default = part[np.isclose(part['value'], BASE[parameter])].iloc[0]
            analysis.append({'dataset': dataset, 'parameter': parameter, 'target_auc_range_percentage_points': float((part['mean_auc'].max() - part['mean_auc'].min()) * 100), 'default_mean_auc': float(default['mean_auc']), 'default_sample_sd': float(default['sd_auc']), 'best_source_validation_value': float(part.loc[part['mean_source_val_auc'].idxmax(), 'value'])})
        output.with_name(output.name + '_descriptive_analysis.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
