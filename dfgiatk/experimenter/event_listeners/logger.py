import yaml

from ..experimenter import e


class Logger:
    """
        Class used to save the plots graphs during training
    """

    def __init__(self, metrics_names=None):
        self.metrics_names = e['metrics_names']

    def on_validation_end(self, ev):
        history = ev['history']

        # Console log
        print('')
        print('')
        print('Done epoch ' + str(history['epoch']) + '.')
        for i in range(1 + len(self.metrics_names)):
            train_value = history['train']['loss'] if i == 0 else history['train']['metrics'][i - 1]
            val_value = history['val']['loss'] if i == 0 else history['val']['metrics'][i - 1]
            description = 'Loss' if i == 0 else self.metrics_names[i - 1]
            print("\t{description} \t\t train {train:.2f} \tval {val:.2f}".format(description=description,
                                                                                  train=train_value,
                                                                                  val=val_value))
        # File log
        yaml.dump(ev['history'].data, open(e.out('stats.yaml'), 'w'))
