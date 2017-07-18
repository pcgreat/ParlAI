# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Simple IR baselines.
# We plan to implement the following variants:
# Given an input message, either:
# (i) find the most similar message in the (training) dataset and output the response from that exchange; or
# (ii) find the most similar response to the input directly.
# (iii) if label_candidates are provided, simply ranks them according to their similarity to the input message.
# Currently only (iii) is used.
#
# Additonally, TFIDF is either used (requires building a dictionary) or not,
# depending on whether you train on the train set first, or not.
import copy
import os

import torch

from parlai.core.agents import Agent
from . import config
from .dict_agent import SimpleDictionaryAgent
from .model import SeqMatchSeqModel
from .utils import vectorize


class SeqmatchseqAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        SeqmatchseqAgent.dictionary_class().add_cmdline_args(argparser)

    @staticmethod
    def dictionary_class():
        return SimpleDictionaryAgent

    def __init__(self, opt, shared=None):
        super(SeqmatchseqAgent, self).__init__(opt)
        if opt['numthreads'] > 1:
            raise RuntimeError("numthreads > 1 not supported for this model.")

        # Load dict.
        if not shared:
            word_dict = SeqmatchseqAgent.dictionary_class()(opt)
        # All agents keep track of the episode (for multiple questions)
        self.episode_done = True

        # Only create an empty dummy class when sharing
        if shared is not None:
            self.is_shared = True
            return

        # Set up params/logging/dicts
        self.is_shared = False
        self.id = self.__class__.__name__
        self.word_dict = word_dict
        self.opt = copy.deepcopy(opt)
        config.set_defaults(self.opt)

        if self.opt.get('model_file') and os.path.isfile(opt['model_file']):
            self._init_from_saved(opt['model_file'])
        else:
            if self.opt.get('pretrained_model'):
                self._init_from_saved(opt['pretrained_model'])
            else:
                self._init_from_scratch()
        self.opt['cuda'] = self.opt['cuda'] and torch.cuda.is_available()
        if self.opt['cuda']:
            print('[ Using CUDA (GPU %d) ]' % opt['gpu'])
            torch.cuda.set_device(opt['gpu'])
            self.model.cuda()
        self.n_examples = 0

    def _init_from_scratch(self):
        self.opt['vocab_size'] = len(self.word_dict)

        print('[ Initializing model from scratch ]')
        self.model = SeqMatchSeqModel(self.opt, self.word_dict)
        self.model.set_embeddings()

    def _init_from_saved(self, fname):
        print('[ Loading model %s ]' % fname)
        saved_params = torch.load(fname,
                                  map_location=lambda storage, loc: storage
                                  )

        # TODO expand dict and embeddings for new data
        self.word_dict = saved_params['word_dict']
        self.state_dict = saved_params['state_dict']
        config.override_args(self.opt, saved_params['config'])
        self.model = SeqMatchSeqModel(self.opt, self.word_dict, self.state_dict)

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            dialogue = self.observation['text'].split('\n')[:-1]
            dialogue.extend(observation['text'].split('\n'))
            observation['text'] = '\n'.join(dialogue)
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        obs = self.observation
        example = self._build_ex(obs)
        batch = [example]

        reply = {}
        reply['id'] = self.getID()

        # Either train or predict
        if 'labels' in obs:
            self.n_examples += 1
            self.model.update(batch)
        else:
            predictions = self.model.predict(batch)
            text_candidates = predictions[0]
            reply['text'] = text_candidates[0]
            reply['text_candidates'] = text_candidates

        return reply

    def batch_act(self, observations):
        """Update or predict on a batch of examples.
        More efficient than act().
        """
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # Some examples will be None (no answer found). Filter them.
        examples = [self._build_ex(obs) for obs in observations]
        examples = [ex for ex in examples if
                    ex is not None]  # ex = (Tensor(q), [Tensor(a1), Tensor(a2)], Tensor(labels))

        # If all examples are invalid, return an empty batch.
        if len(examples) == 0:
            return batch_reply

        # No need to batchify for seqmatchseq
        batch = examples

        # Either train or predict
        if 'labels' in observations[0]:
            self.n_examples += len(examples)
            self.model.update(batch)
        else:
            predictions = self.model.predict(batch)

            # len(predictions) might not equal to len(batch_reply), as batch_reply contain empty observation
            for i, text_candidates in enumerate(predictions):
                batch_reply[i]['text'] = text_candidates[0]
                batch_reply[i]['text_candidates'] = text_candidates

        return batch_reply

    def _build_ex(self, ex):
        """Find the token span of the answer in the context for this example.
        If a token span cannot be found, return None. Otherwise, torchify.
        """
        # Check if empty input (end of epoch)
        if not 'text' in ex:
            return

        if "label_candidates" not in ex:
            raise RuntimeError('Invalid input. Is task a sentence selection task?')

        # Extract question, cands and labels
        question = ex["text"]
        cands, labels = [], []

        for cand in set(ex["label_candidates"] + ex.get("labels", ())):
            cands.append(cand)
            if "labels" in ex:  # train
                labels.append(int(cand in ex["labels"]))

        inputs = {}
        inputs["question"] = [w for w in self.word_dict.tokenize(question)]
        inputs["cands"] = [[w for w in self.word_dict.tokenize(cand)] for cand in cands]
        inputs["labels"] = labels if "labels" in ex else None

        # padding short cand to window large
        window_large = max(self.opt["window_sizes"])
        for i, cand_toks in enumerate(inputs["cands"]):
            if len(cand_toks) < window_large:
                as_tmp = [self.word_dict.null_token] * window_large  # paddings
                inputs["cands"][i] = as_tmp

        if inputs["labels"] is not None:
            assert len(inputs["labels"]) == len(inputs["cands"])
        inputs["cands_as_len"] = [len(x) for x in inputs["cands"]]
        inputs["cands"] = sum(inputs["cands"], [])

        tinputs = vectorize(inputs, self.word_dict)

        # Return inputs with original text + spans (keep for prediction)
        return tinputs + (cands,)

    def save(self, fname=None):
        """Save the parameters of the agent to a file."""
        fname = self.opt.get('model_file', None) if fname is None else fname
        if fname:
            print("[ saving model: " + fname + " ]")
            self.model.save(fname)

    def report(self):
        return {
            "updates": self.model.updates,
            "total": self.n_examples,
            "loss": self.model.train_loss.avg
        }
