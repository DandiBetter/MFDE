from pathlib import Path
import hashlib
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_splits(path, num_classes):
    frame = pd.read_csv(path, dtype={'Case': str, 'ID': str, 'DomainLabel': str})
    required = {'Case', 'ID', 'Label', 'DomainLabel', 'Split'}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError('Complete Case, ID, Label, DomainLabel and Split columns are required.')
    frame['Split'] = frame['Split'].str.lower().str.strip()
    if set(frame['Split']) != {'train', 'val', 'test'}:
        raise ValueError('Explicit nonempty train, val and test partitions are required.')
    if frame['ID'].duplicated().any():
        raise ValueError('Slide identifiers must be unique across partitions.')
    labels = pd.to_numeric(frame['Label'], errors='raise')
    if ((labels % 1) != 0).any() or not labels.between(0, num_classes - 1).all():
        raise ValueError('Labels must be integers in the configured class range.')
    frame['Label'] = labels.astype(int)
    parts = {key: frame[frame['Split'] == key].reset_index(drop=True) for key in ('train', 'val', 'test')}
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(parts[left]['Case']) & set(parts[right]['Case']):
            raise ValueError(f'Patient overlap between {left} and {right}.')
    source = set(parts['train']['DomainLabel'])
    if len(source) != 1 or set(parts['val']['DomainLabel']) != source:
        raise ValueError('Train and validation must belong to the same single source domain.')
    if source & set(parts['test']['DomainLabel']):
        raise ValueError('Target domains must be disjoint from the source domain.')
    return parts, hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenFeatureBags(Dataset):
    def __init__(self, frame, feature_dirs):
        self.frame = frame
        self.paths = []
        for slide_id in frame['ID']:
            if Path(slide_id).name != slide_id or '/' in slide_id or '\\' in slide_id:
                raise ValueError('Slide identifiers must be basenames.')
            paths = {}
            for name, root in feature_dirs.items():
                root = Path(root)
                options = [root / f'{slide_id}.pt', root / 'pt_files' / f'{slide_id}.pt']
                available = [path for path in options if path.is_file()]
                if len(available) != 1:
                    raise ValueError('Each slide must have exactly one feature tensor per encoder.')
                paths[name] = available[0]
            self.paths.append(paths)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        features = {}
        for name, path in self.paths[index].items():
            tensor = torch.load(path, map_location='cpu', weights_only=True)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[0] == 0:
                raise ValueError('Features must be nonempty [patches, channels] tensors.')
            if not torch.isfinite(tensor).all():
                raise ValueError('Nonfinite features are not allowed.')
            features[name] = tensor.float()
        if len({value.shape[0] for value in features.values()}) != 1:
            raise ValueError('Encoder patch counts do not agree.')
        return features, int(self.frame.iloc[index]['Label'])
