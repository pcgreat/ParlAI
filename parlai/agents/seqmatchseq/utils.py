# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import unicodedata

import torch


# ------------------------------------------------------------------------------
# Data/model utilities.
# ------------------------------------------------------------------------------


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def load_embeddings(opt, word_dict):
    """Initialize embeddings from file of pretrained vectors.
    The trick here is to let non-glove words have as less impact as possible"""
    embeddings = torch.randn(len(word_dict), opt['embedding_dim']) * 0.05

    # Fill in embeddings
    if not opt.get('embedding_file'):
        raise RuntimeError('Tried to load embeddings with no embedding file.')
    with open(opt['embedding_file']) as f:
        existing = 0
        for line in f:
            parsed = line.rstrip().split(' ')
            assert (len(parsed) == opt['embedding_dim'] + 1)
            w = normalize_text(parsed[0])
            if w in word_dict:
                vec = torch.Tensor([float(i) for i in parsed[1:]])
                embeddings[word_dict[w]].copy_(vec)
                existing += 1

    # Zero NULL token
    embeddings[word_dict[word_dict.default_null]].fill_(0)
    existing += 1
    return embeddings


def build_feature_dict(opt):
    """Make mapping of feature option to feature index."""
    feature_dict = {}
    if opt['use_in_question']:
        feature_dict['in_question'] = len(feature_dict)
        feature_dict['in_question_uncased'] = len(feature_dict)
    if opt['use_tf']:
        feature_dict['tf'] = len(feature_dict)
    if opt['use_time'] > 0:
        for i in range(opt['use_time'] - 1):
            feature_dict['time=T%d' % (i + 1)] = len(feature_dict)
        feature_dict['time>=T%d' % opt['use_time']] = len(feature_dict)
    return feature_dict


# ------------------------------------------------------------------------------
# Torchified input utilities.
# ------------------------------------------------------------------------------


def vectorize(ex, word_dict):
    """Turn tokenized text inputs into feature vectors."""

    # Index words
    question = torch.LongTensor([word_dict[w] for w in ex['question']])
    cands = torch.LongTensor([word_dict[w] for w in ex['cands']])
    cands_as_len = torch.LongTensor(ex["cands_as_len"])

    if ex['labels'] is not None:
        labels = torch.FloatTensor(ex["labels"])
        return question, cands, cands_as_len, labels
    else:
        return question, cands, cands_as_len


# ------------------------------------------------------------------------------
# General logging utilities.
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
