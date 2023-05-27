import torch

from ..experimenter import e


class ModelSaver:

    def __init__(self):
        self.model = e.model
        self.latest_loss = None

    def save_model(self, name):
        torch.save(self.model.state_dict(), e.out(name))

    def on_epoch_end(self):
        pass

    def on_validation_end(self, ev):
        loss = ev['history']['val']['loss']
        if self.latest_loss is None or loss < self.latest_loss:
            self.latest_loss = loss
            self.save_model('best_model')

        self.save_model('latest_model')
