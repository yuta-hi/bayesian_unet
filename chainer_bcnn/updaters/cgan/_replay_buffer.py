from __future__ import absolute_import

from chainer import backend
from chainer import Variable
import numpy as np


class ReplayBuffer(object):
    """ Buffer for handling the experience replay.
    Args:
        size (int): buffer size
        p (float): probability to evoke the past experience
        return_variable (bool): if True, return chainer's variable

    See also:
     https://arxiv.org/pdf/1612.07828.pdf
     https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, size, p=0.5, return_variable=True):
        self.size = size
        self.p = p
        self.return_variable = return_variable
        self._buffer = []


    @property
    def buffer(self):
        if len(self._buffer) == 0:
            return None
        return self._buffer

    def _preprocess(self, x):
        if isinstance(x, Variable):
            x = x.array
        return x

    def _postprocess(self, x):
        if not self.return_variable:
            return x
        return Variable(x)

    def __call__(self, samples):

        samples = self._preprocess(samples)

        xp = backend.get_array_module(samples)

        n_samples = len(samples)

        if self.size == 0:
            pass
        elif len(self._buffer) == 0:
            self._buffer = samples
        elif len(self._buffer) < self.size:
            self._buffer = xp.vstack((self._buffer, samples))
        else:
            # evoke the memory
            random_bool = np.random.rand(n_samples) < self.p
            replay_indices = np.random.randint(0, len(self._buffer), size=n_samples)[random_bool]
            sample_indices = np.random.randint(0, n_samples, size=n_samples)[random_bool]

            self._buffer[replay_indices], samples[sample_indices] \
                = samples[sample_indices], self._buffer[replay_indices] # swap

        return self._postprocess(samples)
