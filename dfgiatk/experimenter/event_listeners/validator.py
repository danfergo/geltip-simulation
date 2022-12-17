from ..experimenter import e
from ...train import predict_batch

import torch

class Stats:
    def __init__(self, n_metrics):
        self.n_metrics = n_metrics
        self.data = {
            'epoch': 0,
            'train': {
                'loss': 0.0,
                'metrics': [],
                'all_losses': [],
                'all_metrics': [[] for _ in range(n_metrics)]
            },
            'val': {
                'loss': 0.0,
                'metrics': [],
                'all_losses': [],
                'all_metrics': [[] for _ in range(n_metrics)]
            }
        }

    def set_current_epoch(self, epoch):
        self['epoch'] = epoch

    def reset_running_stats(self, phase):
        self[phase]['loss'] = 0.0
        self[phase]['metrics'] = [0.0 for _ in range(self.n_metrics)]

    def update_running_stats(self, phase, loss, metrics):
        self[phase]['loss'] += loss
        self[phase]['metrics'] = [self[phase]['metrics'][m] + metrics[m] for m in range(len(metrics))]

    def normalize_running_stats(self, phase, n_batches):
        self[phase]['loss'] /= n_batches

        def normalize(i, n):
            self[phase]['metrics'][i] /= n

        [normalize(i, n_batches) for i in range(self.n_metrics)]

        self[phase]['all_losses'].append(self[phase]['loss'])
        [self[phase]['all_metrics'][i].append(self[phase]['metrics'][i]) for i in range(self.n_metrics)]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]


class Validator:

    def __init__(self):
        self.stats = Stats(len(e.metrics))
        self.metrics = e.metrics
        self.loss_fn = e.loss
        self.loader = e.val_loader
        self.n_val_batches = e.n_val_batches

    def on_train_epoch_start(self, ev):
        self.stats.set_current_epoch(ev['epoch'])
        self.stats.reset_running_stats('train')

    def on_train_batch_end(self, ev):
        x, y_true = ev['batch']
        y_pred = ev['y_pred']
        self.stats.update_running_stats('train', ev['loss'], [m(y_pred, y_true).item() for m in self.metrics])

    def on_train_epoch_end(self, ev):
        self.stats.normalize_running_stats('train', ev['n_used_batches'])

        self.stats.reset_running_stats('val')

        with torch.inference_mode():
        # with torch.no_grad():
            for batch in iter(self.loader):
                x, y_true = batch
                batch_size = x.size()[0]
                loss, y_pred = predict_batch(batch, feed_size=e.val_feed_size)

                self.stats.update_running_stats('val', loss, [m(y_pred, y_true).item() for m in self.metrics])

        self.stats.normalize_running_stats('val', self.n_val_batches)

        e.emit('validation_end', {'history': self.stats})
