# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import os

from parlai.core.fbdialog_teacher import FbDialogTeacher
from .build import build


def _path(task, opt, dt=''):
    # Build the data if it doesn't exist.
    build(opt)
    if dt == '':
        dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'InsuranceQA', task, dt + '.txt')


# V1 InsuranceQA task
class V1Teacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = _path("V1", opt)
        super().__init__(opt, shared)


# V2 InsuranceQA task
class V2Teacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = _path("V2", opt)
        super().__init__(opt, shared)


class DefaultTeacher(V1Teacher):
    pass
