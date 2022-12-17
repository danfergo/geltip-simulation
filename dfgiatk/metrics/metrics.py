import torch


def accuracy(y_pred, y_true):
    # acc = torch.sum(y_pred == y_true)
    # return acc / y_true.shape[0]
    # print(y_pred.cpu().numpy(), y_true.cpu()
    pred = torch.argmax(y_pred, dim=1)
    labels = y_true
    # print(pred.cpu().numpy(), labels.cpu().numpy())
    correct_predictions = sum([pred[i] == labels[i] for i in range(len(pred))])
    return correct_predictions / len(pred)
