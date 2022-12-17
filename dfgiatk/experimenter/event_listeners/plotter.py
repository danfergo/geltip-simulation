import matplotlib.pyplot as plt
import torch

from ..experimenter import e


class Plotter:
    """
        Class used to save the plots graphs during training
    """

    def __init__(self, file_path='plot.png'):
        self.file_path = file_path
        self.metrics_names = e['metrics_names']

    def on_validation_end(self, ev):
        history = ev['history']

        plt.clf()

        for i in range(1 + len(self.metrics_names)):
            train_values = history['train']['all_losses'] if i == 0 else history['train']['all_metrics'][i - 1]
            val_values = history['val']['all_losses'] if i == 0 else history['val']['all_metrics'][i - 1]
            description = 'Loss ' if i == 0 else self.metrics_names[i - 1]

            if i > 0:
                train_values = [x.cpu() if torch.is_tensor(x) else x for x in train_values]
                val_values = [x.cpu() if torch.is_tensor(x) else x for x in val_values]

            plt.subplot(1 + len(self.metrics_names), 1, i + 1)
            plt.plot(train_values, label='Train ' + description)
            plt.plot(val_values, label='Val ' + description)
            plt.ylabel(description)
            plt.legend()

        plt.savefig(e.out(self.file_path), dpi=150)
