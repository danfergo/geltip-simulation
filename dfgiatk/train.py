"""
Main class that is used to train the NN i.e. run the optimization loop. Samples batches from the train data,
feeds to the NN, computes gradients and updates the network weights. Then computes training metrics. Per epoch,
after training, runs the validation loop by sampling batches from the validation data, and computes validation
metrics. At each relevant moment, calls the corresponding method of the History object, then calls the callbacks
passing them the history.
"""

from .experimenter import e
import torch

torch.cuda.empty_cache()


def feed_chunk(chunk, loss_factor, update_weights=True, zero_grad=True, step=True):
    model, loss_fn = e['model', 'loss']
    optimizer = e.optimizer if update_weights else None
    x, y_true = chunk
    if update_weights:
        # zero the parameter gradients
        if zero_grad:
            optimizer.zero_grad()

    # get predictions and computes loss
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)
    loss = loss * loss_factor
    if update_weights:

        # compute gradients and
        loss.backward()

        # perform optimization step
        if step:
            optimizer.step()

    return loss, y_pred


def feed_batch(batch, feed_size=None, update_weights=True):
    batch_size = batch[0].size()[0]
    feed_size = feed_size or e.feed_size or batch_size
    batch_loss = 0
    batch_pred = []

    for c in range(0, batch_size, feed_size):
        c0 = c
        c1 = min(c + feed_size, batch_size)
        loss_factor = (c1 - c0) / batch_size
        c_loss, c_pred = feed_chunk(
            tuple([x[c0:c1, ...] for x in list(batch)]),
            loss_factor=loss_factor,
            update_weights=update_weights,
            zero_grad=c == 0,  # first step
            step=c1 == batch_size  # last step
        )
        batch_loss += c_loss.item()
        batch_pred = (batch_pred or []) + [c_pred]

    batch_pred = torch.cat(tuple(batch_pred))

    return batch_loss, batch_pred


def fit_to_batch(batch):
    return feed_batch(batch, update_weights=True)


def predict_batch(batch, feed_size=None):
    return feed_batch(batch, feed_size, update_weights=False)


def fit_to_dataset():
    """
    The optimization loop per se
    :return:
    """

    epochs, batch_size, batches_per_epoch, data_loader, train_device, model = e[
        'epochs',
        'batch_size',
        'batches_per_epoch',
        'data_loader',
        'train_device',
        'model'
    ]

    model.to(train_device)

    for epoch in range(epochs):
        e.emit('train_epoch_start', {'epoch': epoch})

        # Train some batches
        for batch in iter(data_loader):
            e.emit('train_batch_start')

            # batch = next(x)
            batch_loss, batch_pred = fit_to_batch(batch)

            # save running stats
            e.emit('train_batch_end', {'batch': batch, 'y_pred': batch_pred, 'loss': batch_loss})

        e.emit('train_epoch_end', {'n_used_batches': batches_per_epoch})
