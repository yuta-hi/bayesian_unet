from __future__ import absolute_import

import os
import numpy as np
import contextlib
import datetime
import warnings

@contextlib.contextmanager
def fixed_seed(seed, strict=False):
    """Fix random seed to improve the reproducibility.

    Args:
        seed (float): Random seed
        strict (bool, optional): If True, cuDNN works under deterministic mode.
            Defaults to False.

    TODO: Even if `strict` is set to True, the reproducibility cannot be guaranteed under the `MultiprocessIterator`.
          If your dataset has stochastic behavior, such as data augmentation, you should use the `SerialIterator` or `MultithreadIterator`.
    """

    import random
    import chainer

    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)

    if strict:
        warnings.warn('Even if `strict` is set to True, the reproducibility cannot be guaranteed under the `MultiprocessIterator`. \
          If your dataset has stochastic behavior such as data augmentation, you should use the `SerialIterator` or `MultithreadIterator`.')

    with chainer.using_config('cudnn_deterministic', strict):
        yield

    pass

def find_latest_snapshot(fmt, path, return_fullpath=True):
    '''Alias of :func:`_find_latest_snapshot`
    '''
    from chainer.training.extensions._snapshot \
        import _find_latest_snapshot

    ret = _find_latest_snapshot(fmt, path)

    if ret is None:
        raise FileNotFoundError('cannot find snapshot for <%s>' %
                                    os.path.join(path, fmt))

    if return_fullpath:
        return os.path.join(path, ret)

    return ret


def get_git_revision(base_path='./', short=True):
    # https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    git_dir = os.path.join(base_path, '.git')
    with open(os.path.join(git_dir, 'HEAD'), 'r') as head:
        ref = head.readline().split(' ')[-1].strip()

    with open(os.path.join(git_dir, ref), 'r') as git_hash:
        revision = git_hash.readline().strip()

    if short:
        return revision[:7]

    return revision


def get_logstamp(delimiter='_'):

    now = datetime.datetime.now()

    logstamp = []
    logstamp.append(now.strftime('%y%m%d'))
    logstamp.append(now.strftime('%H%M%S'))

    try:
        logstamp.append(get_git_revision())
    except Exception:
        pass

    return delimiter.join(logstamp)
