from chainer_bcnn.updaters.cgan._replay_buffer import ReplayBuffer

if __name__ == '__main__':

    import cupy
    from chainer import Variable

    buffer = ReplayBuffer(10)
    print(buffer.buffer)

    for i in range(20):
        a = buffer(Variable(cupy.zeros((2,3,4,5)) + i))
        print(i, a)
    print(a.shape)
    print(a.__class__)

    print(buffer.buffer)
    print(len(buffer.buffer))
