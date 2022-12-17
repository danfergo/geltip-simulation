import os
import random

import torch
import torchvision
import itertools
from torchvision.datasets.folder import make_dataset
from torchvision import transforms as t
from os import path
import yaml
import cv2
import numpy as np


def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_samples(root, extensions=(".mp4", ".avi", ".webm")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)


# def to_categorical():
#     batch_size = 5
#     nb_digits = 10
#     # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
#     y = torch.LongTensor(batch_size, 1).random_() % nb_digits
#     # One hot encoding buffer that you create out of the loop and just keep reusing
#     y_onehot = torch.FloatTensor(batch_size, nb_digits)
#
#     # In your for loop
#     y_onehot.zero_()
#     y_onehot.scatter_(1, y, 1)
#
#     print(y)
#     print(y_onehot)


class ImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, base_path,
                 train_split,
                 batch_size=32
                 # epoch_size=None,
                 # frame_transform=None,
                 # video_transform=None,
                 # step_frames=1,
                 # clip_len=16,
                 # labels=None
                 ):
        super(ImageDataset).__init__()

        def get_cls(s):
            return s.split('/')[-2]

        self.batch_size = batch_size
        self.classes = list({get_cls(s): 1 for s in train_split}.keys())
        self.classes.sort()

        def get_target(s):
            idx = self.classes.index(get_cls(s))
            target = np.zeros((len(self.classes, )))
            target[idx] = 1
            return target

        self.samples = [[path.join(base_path, s), get_target(s)] for s in train_split]

        self.base_path = base_path
        # if epoch_size is None:
        #     epoch_size = len(self.samples)
        self.epoch_size = 1000
        self.n = 0

        # self.step_frames = step_frames
        # self.class_names = tuple(_find_classes(root)[0])
        # # # Allow for temporal jittering
        #
        # self.clip_len = clip_len
        # self.frame_transform = frame_transform
        # self.video_transform = video_transform
        # self.length = int(1e4)  # if self.dataset is None else self.dataset.shape[1]
        # self.clip_slice_len = self.clip_len + (self.clip_len - 1) * self.step_frames
        # self.labels = labels
        # self.y_true_one_hot = torch.IntTensor(1, len(self.class_names))

    def sample_video_frames(self, vid):
        metadata = vid.get_metadata()

        while True:  # quick hack to ensure all batches have clip_len frames
            video_frames = []  # video frame buffer

            # Seek and return frames
            max_seek = metadata["video"]['duration'][0] - (self.clip_slice_len / metadata["video"]['fps'][0])
            start = random.uniform(0., max_seek)

            for frame in itertools.islice(vid.seek(start), 0, self.clip_len, self.step_frames - 1):
                video_frames.append(self.frame_transform(frame['data']))
                current_pts = frame['pts']

            # quick hack to ensure all batches have clip_len frames, always
            if len(video_frames) == self.clip_len:
                return video_frames
            else:
                print('Failed to sample batch. clip length: ' + str(self.clip_len)
                      + ' start: ' + str(start) + ' current pts: ' + str(current_pts) + ' metadata: ', metadata)

    def sample_one(self):
        # Get random sample
        path, target = random.choice(self.samples)

        # print(path)
        img = cv2.imread(path)

        x = torch.from_numpy(img)
        y_true = torch.from_numpy(target)

        return x, y_true
        # return x.to('cuda'), y_true.to('cuda')

    def __iter__(self):
        return self

    def __next__(self):
        if self.n > self.epoch_size:
            raise StopIteration
        else:

            batch = list(zip(*[self.sample_one() for _ in range(self.batch_size)]))

            # print(batch)
            x = torch.stack(batch[0])
            y_true = torch.stack(batch[1])

            return x.to('cuda'), y_true.to('cuda')

        # torch.stack()
        # # Get random sample
        # path, target = random.choice(self.samples)
        #
        # # print(path)
        # img = cv2.imread(path)
        #
        # x = torch.from_numpy(img)
        # y_true = torch.from_numpy(target)
        # for i in range(self.epoch_size):
        # Get video object
        # vid = torchvision.io.VideoReader(path, "video")
        # video_frames = self.sample_video_frames(vid)
        # video = torch.stack(video_frames, 0)

        # if self.video_transform:
        #     video = self.video_transform(video)
        #
        # x, y_true = video, target  # next(enumerate(self.train_data))[1]
        # if self.labels is not None:
        #     y_true = self.labels[y_true]
        # x = torch.swapaxes(x, 0, 1).float()

        # y_true = torch.tensor(y_true)
        # y_true = torch.nn.functional.one_hot(y_true)
        # print(y_true,  self.y_true_one_hot)
        # self.y_true_one_hot.zero_().scatter_(1, y_true, 1)

        # y_true = torch.nn.functional.one_hot(y_true, num_classes=len(self.class_names))
        #

    # def __len__(self):
    #     return 1000


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    base = '/home/danfergo/Projects/PhD/geltip_simulation/geltip_dataset/dataset/'

    loader = ImageDataset(
        path.join(base + 'real_rgb'),
        yaml.load(open(path.join(base, 'train_split.yaml'))),
    )

    # batch_size=32
    # data = {"video": [], 'start': [], 'end': [], 'tensorsize': []}
    for batch in loader:
        print(batch[0].size())

        # print(batch)
        # for i in range(len(batch['path'])):
        # data['video'].append(batch['path'][i])
        # data['start'].append(batch['start'][i].item())
        # data['end'].append(batch['end'][i].item())
        # data['tensorsize'].append(batch['video'][i].size())

# import matplotlib.pylab as plt
#
# plt.figure(figsize=(12, 12))
# for batch in loader:
#     for j in range(10):
#         plt.subplot(4, 4, j + 1)
#         # print(batch.shape)
#         for i in range(16):
#             plt.imshow(batch["video"][j, i, ...].permute(1, 2, 0))
#             # plt.axis("off")
#             print('xx', batch["video"].shape)
#         plt.show()

# ===============================================================================
#
# ===============================================================================


#
# import torch
# # Choose the `slow_r50` model2
# model2 = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
# print(model2)
#
# # Set to GPU or CPU
# device = "gpu"
# model2 = model2.eval()
# model2 = model2.to(device)
#
#
#
#
# # Pass the input clip through the model2
# preds = model2(inputs[None, ...])
#
# # Get the predicted classes
# post_act = torch.nn.Softmax(dim=1)
# preds = post_act(preds)
# pred_classes = preds.topk(k=5).indices[0]
#
# # Map the predicted classes to the label names
# pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
# print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))


# Author: Robert Guthrie
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
# torch.manual_seed(1)
#
# # the network
#
# lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
# inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
#
# # initialize the hidden state.
# hidden = (torch.randn(1, 1, 3),
#           torch.randn(1, 1, 3))
# for i in inputs:
#     # Step through the sequence one element at a time.
#     # after each step, hidden contains the hidden state.
#     out, hidden = lstm(i.view(1, 1, -1), hidden)
#
# # alternatively, we can do the entire sequence all at once.
# # the first value returned by LSTM is all of the hidden states throughout
# # the sequence. the second is just the most recent hidden state
# # (compare the last slice of "out" with "hidden" below, they are the same)
# # The reason for this is that:
# # "out" will give you access to all hidden states in the sequence
# # "hidden" will allow you to continue the sequence and backpropagate,
# # by passing it as an argument  to the lstm at a later time
# # Add the extra 2nd dimension
# inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
# out, hidden = lstm(inputs, hidden)
# print(out)
# print(hidden)


# class LSTMTagger(nn.Module):
#
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
#         super(LSTMTagger, self).__init__()
#         self.hidden_dim = hidden_dim
#
#         self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
#
#         # The LSTM takes word embeddings as inputs, and outputs hidden states
#         # with dimensionality hidden_dim.
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim)
#
#         # The linear layer that maps from hidden state space to tag space
#         self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
#
#     def forward(self, sentence):
#         embeds = self.word_embeddings(sentence)
#         lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
#         tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
#         tag_scores = F.log_softmax(tag_space, dim=1)
#         return tag_scores
#
#
#
#
#
# model2 = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
# loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model2.parameters(), lr=0.1)
#
# # See what the scores are before training
# # Note that element i,j of the output is the score for tag j for word i.
# # Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model2(inputs)
#     print(tag_scores)
#
# for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
#     for sentence, tags in training_data:
#         # Step 1. Remember that Pytorch accumulates gradients.
#         # We need to clear them out before each instance
#         model2.zero_grad()
#
#         # Step 2. Get our inputs ready for the network, that is, turn them into
#         # Tensors of word indices.
#         sentence_in = prepare_sequence(sentence, word_to_ix)
#         targets = prepare_sequence(tags, tag_to_ix)
#
#         # Step 3. Run our forward pass.
#         tag_scores = model2(sentence_in)
#
#         # Step 4. Compute the loss, gradients, and update the parameters by
#         #  calling optimizer.step()
#         loss = loss_function(tag_scores, targets)
#         loss.backward()
#         optimizer.step()
#
# # See what the scores are after training
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model2(inputs)
#
#     # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#     # for word i. The predicted tag is the maximum scoring tag.
#     # Here, we can see the predicted sequence below is 0 1 2 0 1
#     # since 0 is index of the maximum value of row 1,
#     # 1 is the index of maximum value of row 2, etc.
#     # Which is DET NOUN VERB DET NOUN, the correct sequence!
#     print(tag_scores)
