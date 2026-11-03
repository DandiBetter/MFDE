import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from mfde import MFDE, MFDEEngine
from mfde.data import FrozenFeatureBags, read_splits


def build_model(config, sample):
    model = MFDE(instance_dim=next(iter(sample.values())).shape[-1], embed_instance_dim=config['embed_dim'], bag_dim=config['embed_dim'], n_classes=config['num_classes'], num_attention_heads=config['attention_heads'], attention_dim=config['attention_dim'], gated=False, bias=False, dropout=config['attention_dropout'], act='gelu', num_prototypes=config['num_prototypes'])
    model.get_submodules(sample)
    return model


def evaluate(model, loader, device, num_classes):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for features, target in loader:
            output = model({key: value.to(device) for key, value in features.items()})
            probabilities.append(output['logits'].softmax(-1).cpu().numpy())
            labels.extend(target.tolist())
    probabilities = np.concatenate(probabilities)
    if set(labels) != set(range(num_classes)):
        raise ValueError('AUC is undefined: the evaluation split does not contain every configured class.')
    if num_classes == 2:
        return float(roc_auc_score(labels, probabilities[:, 1]))
    return float(roc_auc_score(labels, probabilities, multi_class='ovr', average='macro', labels=list(range(num_classes))))


def expand_paths(value):
    if isinstance(value, dict):
        return {key: expand_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_paths(item) for item in value]
    if isinstance(value, str):
        result = os.path.expandvars(value)
        if '${' in result:
            raise ValueError('An input environment variable has not been set.')
        return result
    return value


def run(config):
    config = expand_paths(config)
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select an appropriate runtime or set device to cpu.')
    parts, split_hash = read_splits(config['split_csv'], config['num_classes'])
    datasets = {key: FrozenFeatureBags(frame, config['feature_dirs']) for key, frame in parts.items()}
    generator = torch.Generator().manual_seed(seed)
    loaders = {key: DataLoader(data, batch_size=1, shuffle=key == 'train', num_workers=0, generator=generator if key == 'train' else None) for key, data in datasets.items()}
    sample, _ = datasets['train'][0]
    model = build_model(config, sample).to(device)
    if config['optimizer'] != 'Adam':
        raise ValueError('This focused package implements the documented Adam configuration.')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0)
    virtual_batch = config['virtual_batch_size']
    if virtual_batch < 1 or config['epochs'] < 1:
        raise ValueError('Epoch count and virtual batch size must be positive.')
    protocol = copy.deepcopy(config)
    for key in ('split_csv', 'feature_dirs', 'output_dir', 'seed', 'num_prototypes'):
        protocol.pop(key, None)
    protocol['encoder_names'] = list(config['feature_dirs'])
    for key in ('routing_temperature', 'lambda_reg_effective'):
        protocol['objective'].pop(key, None)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    engine = MFDEEngine(model, optimizer, scheduler, config, output)
    training = engine.learning(loaders['train'], loaders['val'], loaders['test'])
    best_score, best_epoch = training['source_val_auc'], training['selected_epoch']
    target_auc = training['target_auc']
    result = {'dataset': config['dataset'], 'task': config['task'], 'seed': seed, 'num_prototypes': config['num_prototypes'], 'routing_temperature': config['objective']['routing_temperature'], 'lambda_reg_effective': config['objective']['lambda_reg_effective'], 'source_val_auc': best_score, 'target_auc': target_auc, 'selected_epoch': best_epoch, 'split_hash': split_hash, 'protocol_hash': protocol_hash, 'implementation': config['implementation'], 'selection': 'source_validation', 'pool': '+'.join(config['feature_dirs']), 'counts': {key: len(frame) for key, frame in parts.items()}}
    if 'oracle' in training:
        result['oracle_target_auc'] = training['oracle']['target_auc']
        result['oracle_selected_epoch'] = training['oracle']['selected_epoch']
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--oracle-reporting', action='store_true')
    arguments = parser.parse_args()
    config = json.loads(Path(arguments.config).read_text(encoding='utf-8'))
    if arguments.oracle_reporting:
        config.setdefault('engine', {})['oracle_reporting'] = True
    print(json.dumps(run(config), indent=2))
