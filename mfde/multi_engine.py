import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import SequentialSampler
from .objective import compute_objective
from .prior_preservation import PriorPreservationTeacher


def unpack_batch(batch):
    if len(batch) == 2:
        features, labels = batch
        return features, labels, None, None
    if len(batch) == 6:
        _, slide_ids, features, labels, coordinates, _ = batch
        return features, labels, slide_ids, coordinates
    raise ValueError('A batch must contain features/labels or six legacy slide fields.')


def classification_metrics(labels, probabilities, num_classes, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(labels), num_classes) or not np.isfinite(probabilities).all():
        raise ValueError('Evaluation probabilities must be finite and match the configured classes.')
    if set(labels.tolist()) != set(range(num_classes)):
        raise ValueError('AUC is undefined: the evaluation split must contain every configured class.')
    if num_classes == 2:
        auc = roc_auc_score(labels, probabilities[:, 1])
        ap = average_precision_score(labels, probabilities[:, 1])
        predictions = (probabilities[:, 1] >= threshold).astype(np.int64)
        average = 'binary'
    else:
        auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro', labels=list(range(num_classes)))
        ap = average_precision_score(np.eye(num_classes)[labels], probabilities, average='macro')
        predictions = probabilities.argmax(axis=1)
        average = 'macro'
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average=average, zero_division=0)
    metrics = {'score': float(auc), 'roc_auc': float(auc), 'average_precision': float(ap), 'accuracy': float(accuracy_score(labels, predictions)), 'precision': float(precision), 'recall': float(recall), 'f1': float(fscore)}
    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        metrics['ppv'] = float(tp / (tp + fp)) if tp + fp else 0.0
        metrics['npv'] = float(tn / (tn + fn)) if tn + fn else 0.0
    return metrics


class MFDEEngine:
    def __init__(self, model, optimizer, scheduler, config, output_dir, logger=None, objective=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.options = config.get('engine', {})
        self.output_dir = Path(output_dir)
        self.device = next(model.parameters()).device
        self.logger = logger
        self.objective = objective or compute_objective
        self.virtual_batch_size = config['virtual_batch_size']
        self.epochs = config['epochs']
        self.num_classes = config['num_classes']
        self.oracle_reporting = self.options.get('oracle_reporting', False)
        self.patience = self.options.get('early_stopping_patience')
        self.threshold = self.options.get('binary_threshold', 0.5)
        self.averaging_start = self.options.get('parameter_averaging_start_update')
        if self.virtual_batch_size < 1 or self.epochs < 1:
            raise ValueError('Epoch count and virtual batch size must be positive.')
        if self.patience is not None and self.patience < 1:
            raise ValueError('Early-stopping patience must be positive or null.')
        if self.averaging_start is not None and self.averaging_start < 1:
            raise ValueError('Parameter averaging starts at a positive optimizer update.')
        if not 0 < self.threshold < 1:
            raise ValueError('The binary decision threshold must be between zero and one.')
        if not isinstance(self.oracle_reporting, bool):
            raise ValueError('oracle_reporting must be a boolean.')
        self.teacher = PriorPreservationTeacher(model, decay=config['ema_decay']).to(self.device)
        self.averaged_model = AveragedModel(model).to(self.device) if self.averaging_start is not None else None
        self.optimizer_updates = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, train_loader):
        if len(train_loader) == 0:
            raise ValueError('The source training loader must be nonempty.')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pending, pending_count, total_count = [], 0, 0
        totals = {}
        for index, batch in enumerate(train_loader):
            features, labels, _, _ = unpack_batch(batch)
            features = {name: tensor.to(self.device) for name, tensor in features.items()}
            labels = labels.to(self.device).long().reshape(-1)
            student = self.model(features)
            teacher = self.teacher(features)
            loss, metrics = self.objective(student, teacher, labels, self.model, self.config['objective'])
            if not torch.isfinite(loss):
                raise ValueError('The training objective is nonfinite.')
            count = len(labels)
            pending.append(loss * count)
            pending_count += count
            total_count += count
            for key, value in {'total': float(loss.detach()), **metrics}.items():
                totals[key] = totals.get(key, 0.0) + float(value) * count
            if len(pending) == self.virtual_batch_size or index + 1 == len(train_loader):
                (torch.stack(pending).sum() / pending_count).backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.teacher.update(self.model)
                self.optimizer_updates += 1
                if self.averaged_model is not None and self.optimizer_updates >= self.averaging_start:
                    self.averaged_model.update_parameters(self.model)
                pending, pending_count = [], 0
        return {key: value / total_count for key, value in totals.items()}

    def evaluation_model(self):
        if self.averaged_model is not None and int(self.averaged_model.n_averaged) > 0:
            return self.averaged_model
        return self.model

    def _evaluate(self, data_loader, model, collect=False):
        model.eval()
        all_probabilities, all_labels, features, weights, attention, slide_ids = [], [], [], [], [], []
        total_loss, count = 0.0, 0
        ordered_names = None
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels, identifiers, coordinates = unpack_batch(batch)
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                labels = labels.to(self.device).long().reshape(-1)
                result = model(inputs)
                batch_size = len(labels)
                all_probabilities.append(result['logits'].softmax(-1).cpu())
                all_labels.append(labels.cpu())
                total_loss += float(F.cross_entropy(result['logits'], labels, reduction='sum'))
                if collect:
                    names = list(result['ordered_names'])
                    if ordered_names is not None and names != ordered_names:
                        raise ValueError('The encoder order changed during evaluation.')
                    ordered_names = names
                    features.append(result['bag_feat'].cpu())
                    weights.append(result['weights'].cpu())
                    if identifiers is None:
                        frame = getattr(data_loader.dataset, 'frame', None)
                        if frame is not None and not isinstance(data_loader.sampler, SequentialSampler):
                            raise ValueError('Prediction export requires sequential evaluation or explicit slide identifiers.')
                        identifiers = frame.iloc[count:count + batch_size]['ID'].tolist() if frame is not None else [str(i) for i in range(count, count + batch_size)]
                    if isinstance(identifiers, str):
                        identifiers = [identifiers]
                    slide_ids.extend(str(value) for value in identifiers)
                    if self.options.get('export_attention', False):
                        branches = len(names)
                        effective = result['attn'].reshape(batch_size, branches, -1)
                        raw = result['attn_ori'].reshape(batch_size, branches, -1, effective.shape[-1])
                        for i in range(batch_size):
                            item = {'instance_weights': effective[i].cpu(), 'head_scores': raw[i].cpu(), 'head_weights': result['head_weights'][i].cpu(), 'routing_weights': result['weights'][i].cpu()}
                            if coordinates is not None:
                                item['coordinates'] = torch.as_tensor(coordinates[i]).cpu()
                            attention.append(item)
                count += batch_size
        if count == 0:
            raise ValueError('The evaluation loader must be nonempty.')
        probabilities = torch.cat(all_probabilities)
        labels = torch.cat(all_labels)
        metrics = classification_metrics(labels.numpy(), probabilities.numpy(), self.num_classes, self.threshold)
        metrics['loss'] = total_loss / count
        outputs = None
        if collect:
            outputs = {'slide_ids': slide_ids, 'labels': labels, 'probabilities': probabilities, 'bag_features': torch.cat(features), 'routing_weights': torch.cat(weights), 'encoder_names': ordered_names, 'metrics': metrics, 'attention': attention}
        return metrics, outputs

    def validate(self, data_loader, model=None):
        return self._evaluate(data_loader, self.model if model is None else model)[0]

    def testing(self, data_loader, model=None, output_name='target_id_outputs.pt'):
        collect = self.options.get('export_predictions', True)
        metrics, outputs = self._evaluate(data_loader, self.model if model is None else model, collect=collect)
        if outputs is not None:
            torch.save(outputs, self.output_dir / output_name)
        return metrics

    def _save_model(self, model, filename):
        module = model.module if isinstance(model, AveragedModel) else model
        torch.save(module.state_dict(), self.output_dir / filename)

    def learning(self, train_loader, val_loader, test_loader):
        history, best_source, best_oracle = [], -float('inf'), -float('inf')
        source_epoch, oracle_epoch, stale_epochs = -1, -1, 0
        best_source_metrics = None
        for epoch in range(self.epochs):
            start = time.perf_counter()
            train_metrics = self.train(train_loader)
            elapsed = time.perf_counter() - start
            evaluation_model = self.evaluation_model()
            validation = self.validate(val_loader, evaluation_model)
            if validation['roc_auc'] > best_source:
                best_source, source_epoch = validation['roc_auc'], epoch
                best_source_metrics = validation
                stale_epochs = 0
                self._save_model(evaluation_model, 'best_source_validation.pt')
            else:
                stale_epochs += 1
            record = {'epoch': epoch, 'source_validation_auc': validation['roc_auc'], 'source_validation_metrics': validation, 'train_loss': train_metrics['total'], 'train_metrics': train_metrics, 'training_seconds': elapsed, 'optimizer_updates': self.optimizer_updates}
            if self.oracle_reporting:
                target = self.validate(test_loader, evaluation_model)
                record['oracle_target_auc'] = target['roc_auc']
                if target['roc_auc'] > best_oracle:
                    best_oracle, oracle_epoch = target['roc_auc'], epoch
                    self._save_model(evaluation_model, 'best_oracle.pt')
            history.append(record)
            (self.output_dir / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False), encoding='utf-8')
            if self.logger is None:
                print(json.dumps(record, allow_nan=False), flush=True)
            else:
                self.logger(record)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.patience is not None and stale_epochs >= self.patience:
                break
        self.model.load_state_dict(torch.load(self.output_dir / 'best_source_validation.pt', map_location=self.device, weights_only=True))
        target_metrics = self.testing(test_loader)
        result = {'source_val_auc': best_source, 'selected_epoch': source_epoch, 'target_auc': target_metrics['roc_auc'], 'source_validation_metrics': best_source_metrics, 'target_metrics': target_metrics, 'epochs_completed': len(history), 'optimizer_updates': self.optimizer_updates, 'selection': 'source_validation'}
        if self.oracle_reporting:
            self.model.load_state_dict(torch.load(self.output_dir / 'best_oracle.pt', map_location=self.device, weights_only=True))
            oracle_metrics = self.testing(test_loader, output_name='target_oracle_outputs.pt')
            result['oracle'] = {'selection': 'target_oracle', 'selected_epoch': oracle_epoch, 'selection_auc': best_oracle, 'target_auc': oracle_metrics['roc_auc'], 'target_metrics': oracle_metrics}
            self.model.load_state_dict(torch.load(self.output_dir / 'best_source_validation.pt', map_location=self.device, weights_only=True))
        (self.output_dir / 'engine_result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        return result
