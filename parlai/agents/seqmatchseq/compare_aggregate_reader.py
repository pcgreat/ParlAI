# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn

from .layers import LinearLogSoftmax, \
    ConvolutionDMax, FullyConnected, SimMul, SoftAttention


class CompareAggregateReader(nn.Module):
    def __init__(self, opt, padding_idx=0):
        super(CompareAggregateReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(opt['vocab_size'],
                                      opt['embedding_dim'],
                                      padding_idx=padding_idx)

        # ...(maybe) keep them fixed
        if opt['fix_embeddings']:
            for p in self.embedding.parameters():
                p.requires_grad = False

        # Register a buffer to (maybe) fill later for keeping *some* fixed
        if opt['tune_partial'] > 0:
            buffer_size = torch.Size((
                opt['vocab_size'] - opt['tune_partial'] - 2,
                opt['embedding_dim']
            ))
            self.register_buffer('fixed_embedding', torch.Tensor(buffer_size))

        # Define our layers
        self.att_module_master = SoftAttention()

        if opt["comp_type"] == "mul":
            self.sim_sg_module = SimMul()
        else:
            Exception("The word matching method is not provided")

        self.conv_module = ConvolutionDMax(opt["window_sizes"], opt["cov_dim"], opt["mem_dim"], opt["cuda"])
        self.soft_module = LinearLogSoftmax(opt["mem_dim"])
        self.proj_modules = FullyConnected(opt['embedding_dim'], opt["mem_dim"])
        self.dropout_modules = nn.Dropout(opt["dropoutP"])

    def forward(self, data_q, data_as, data_as_len):
        inputs_q_emb = self.embedding(data_q)
        inputs_a_emb = self.embedding(data_as)

        inputs_a_emb = self.dropout_modules(inputs_a_emb)
        inputs_q_emb = self.dropout_modules(inputs_q_emb)

        projs_a_emb = self.proj_modules(inputs_a_emb)
        projs_q_emb = self.proj_modules(inputs_q_emb)

        if data_q.size()[0] == 1:
            projs_q_emb = projs_q_emb.resize(1, self.mem_dim)

        att_output = self.att_module_master(projs_q_emb, projs_a_emb)
        sim_output = self.sim_sg_module(projs_a_emb, att_output)

        conv_output = self.conv_module(sim_output, data_as_len)
        soft_output = self.soft_module(conv_output)

        return soft_output
