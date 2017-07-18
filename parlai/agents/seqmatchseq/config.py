# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import os

import torch


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_cmdline_args(parser):
    # Runtime environment
    agent = parser.add_argument_group('SeqMatchSeq Arguments')
    agent.add_argument('--cuda', type='bool', default=False)
    agent.add_argument('--random_seed', type=int, default=528)

    # Basics
    agent.add_argument('--embedding_file', type=str, default=None,
                       help='File of space separated embeddings: w e1 ... ed')
    agent.add_argument('--pretrained_model', type=str, default=None,
                       help='Load dict/features/weights/opts from this file')
    agent.add_argument('--log_file', type=str, default=None)

    # Model details
    agent.add_argument('--fix_embeddings', type='bool', default=True)
    agent.add_argument('--tune_partial', type=int, default=0,
                       help='Train the K most frequent word embeddings')
    agent.add_argument('--embedding_dim', type=int, default=300,
                       help=('Default embedding size if '
                             'embedding_file is not given'))

    # Optimization details
    agent.add_argument('--display_iter', type=int, default=10,
                       help='Print train error after every \
                              <display_iter> epoches (default 10)')
    agent.add_argument('--optimizer', type=str, default='adam',
                       help='Optimizer: sgd, adamax, adam (default)')
    agent.add_argument('--learning_rate', '-lr', type=float, default=0.004,
                       help='Learning rate for SGD (default 0.1)')

    # Model-specific
    agent.add_argument('--mem_dim', type=int, default=150, help='state dimension')
    agent.add_argument('--cov_dim', type=int, default=150, help='conv dimension')
    agent.add_argument('--window_sizes', type=list, default=[1, 2, 3, 4, 5], help='window sizes')

    agent.add_argument('--dropoutP', type=float, default=0.04, help='dropout ratio')
    agent.add_argument('--comp_type', type=str, choices=['mul'], default="mul", help='w-by-w type')


def set_defaults(opt):
    # Set random seed
    torch.manual_seed(opt["random_seed"])

    # Embeddings options
    if opt.get('embedding_file'):
        if not os.path.isfile(opt['embedding_file']):
            raise IOError('No such file: %s' % opt['embedding_file'])
        with open(opt['embedding_file']) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        opt['embedding_dim'] = dim
    elif not opt.get('embedding_dim'):
        raise RuntimeError(('Either embedding_file or embedding_dim '
                            'needs to be specified.'))

    # Make sure tune_partial and fix_embeddings are consistent
    if opt['tune_partial'] > 0 and opt['fix_embeddings']:
        print('Setting fix_embeddings to False as tune_partial > 0.')
        opt['fix_embeddings'] = False

    # Make sure fix_embeddings and embedding_file are consistent
    if opt['fix_embeddings']:
        if not opt.get('embedding_file') and not opt.get('pretrained_model'):
            print('Setting fix_embeddings to False as embeddings are random.')
            opt['fix_embeddings'] = False


def override_args(opt, override_opt):
    # Major model args are reset to the values in override_opt.
    # Non-architecture args (like dropout) are kept.
    args = set(['embedding_file', 'embedding_dim', 'hidden_size', 'mem_dim', 'cov_dim', 'window_sizes',
                'vocab_size', 'tune_partial'])
    for k, v in override_opt.items():
        if k in args:
            opt[k] = v
