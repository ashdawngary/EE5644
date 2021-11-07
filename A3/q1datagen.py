import numpy as np

# class means
mu = np.asarray([[1, 1, 1],
                 [-1, -1, 1],
                 [-1, 1, -1],
                 [1, -1, -1]])

covall = np.eye(3)


## assume equal priors

def generate_data(nsamples):
    global mu, covall
    y = np.random.randint(0, high=4, size=nsamples)
    x = np.zeros([nsamples, 3])

    counts = [0, 0, 0, 0]
    arrs = [None, None, None, None]
    for i in range(0, 4):
        counts[i] = np.count_nonzero(y == i)
        arrs[i] = np.random.multivariate_normal(mu[i, :], covall, counts[i])

    class_used = [0, 0, 0, 0]
    for ix in range(0, len(y)):
        x[ix, :] = arrs[y[ix]][class_used[y[ix]], :]
        class_used[y[ix]] += 1

    return x.T, y.T, np.eye(4)[y].T


for train_size in [100, 200, 500, 1000, 2000, 5000]:
    xtrain, ytrain, one_hot = generate_data(train_size)
    with open("q1data/trainset_%s.npy" % train_size, 'wb') as f:
        np.save(f, xtrain)
        np.save(f, ytrain)
        np.save(f, one_hot)
        f.close()

xtest, ytest, one_hot = generate_data(100000)
with open("q1data/test.npy", 'wb') as f:
    np.save(f, xtest)
    np.save(f, ytest)
    np.save(f, one_hot)
    f.close()
