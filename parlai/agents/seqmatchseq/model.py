# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .compare_aggregate_reader import CompareAggregateReader
from .utils import load_embeddings, AverageMeter

logger = logging.getLogger('DrQA')


class SeqMatchSeqModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, word_dict, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.word_dict = word_dict
        self.updates = 0
        self.train_loss = AverageMeter()

        # Building network.
        self.network = CompareAggregateReader(opt)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if not k in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        # criterion
        self.criterion = nn.KLDivLoss()

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=opt['learning_rate'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])

    def set_embeddings(self):
        # Read word embeddings.
        if not self.opt.get('embedding_file'):
            logger.warning('[ WARNING: No embeddings provided. '
                           'Keeping random initialization. ]')
            return
        logger.info('[ Loading pre-trained embeddings ]')
        embeddings = load_embeddings(self.opt, self.word_dict)
        logger.info('[ Num embeddings = %d ]' % embeddings.size(0))

        # Sanity check dimensions
        new_size = embeddings.size()
        old_size = self.network.embedding.weight.size()
        if new_size[1] != old_size[1]:
            raise RuntimeError('Embedding dimensions do not match.')
        if new_size[0] != old_size[0]:
            logger.warning(
                '[ WARNING: Number of embeddings changed (%d->%d) ]' %
                (old_size[0], new_size[0])
            )

        # Swap weights
        self.network.embedding.weight.data = embeddings

        # If partially tuning the embeddings, keep the old values
        if self.opt['tune_partial'] > 0:
            if self.opt['tune_partial'] + 2 < embeddings.size(0):
                fixed_embedding = embeddings[self.opt['tune_partial'] + 2:]
                self.network.fixed_embedding = fixed_embedding

    def update(self, batch: list):
        # Train mode
        self.network.train()

        batch_loss = 0.
        for ex in batch:
            # Transfer to GPU
            if self.opt['cuda']:
                data_q = Variable(ex[0].cuda(async=True), requires_grad=False)
                data_as = Variable(ex[1].cuda(async=True), requires_grad=False)
                data_as_len = ex[2].cuda(async=True)
                target = Variable(ex[3].cuda(async=True), requires_grad=False)
            else:
                data_q = Variable(ex[0], requires_grad=False)
                data_as = Variable(ex[1], requires_grad=False)
                data_as_len = ex[2]
                target = Variable(ex[3], requires_grad=False)

            # Run forward
            soft_output = self.network(data_q, data_as, data_as_len)

            # Compute loss and accuracies
            example_loss = self.criterion(soft_output, target)
            batch_loss += example_loss

        loss = batch_loss / len(batch)
        self.train_loss.update(loss.data[0], len(batch))

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def predict(self, batch):
        # Eval mode
        self.network.eval()

        predictions = []

        # Transfer to GPU
        for ex in batch:
            # Transfer to GPU
            if self.opt['cuda']:
                data_q = Variable(ex[0].cuda(async=True), requires_grad=False)
                data_as = Variable(ex[1].cuda(async=True), requires_grad=False)
                data_as_len = ex[2].cuda(async=True)
            else:
                data_q = Variable(ex[0], requires_grad=False)
                data_as = Variable(ex[1], requires_grad=False)
                data_as_len = ex[2]

            # Get argmax text spans
            candidates = ex[-1]

            # Run forward
            soft_output = self.network.forward(data_q, data_as, data_as_len)

            # Transfer to CPU/normal tensors for numpy ops
            soft_output = soft_output.data.cpu().tolist()

            assert len(candidates) == len(soft_output)
            text_candidates = [candidates[x] for x in
                               np.argsort(soft_output)[::-1]]  # sort candiates from highest score to lowest
            predictions.append(text_candidates)
        return predictions

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial'] + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'word_dict': self.word_dict,
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()
