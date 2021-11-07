import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_data(nsamples, plot_data=False):
    global mu, covall

    priors = [0.15, 0.35, 0.2, 0.3]

    make_cov = lambda varx, vary: np.asarray([[varx, 0], [0, vary]])

    scale = 1 / 4

    cov_1 = scale * make_cov(1, 2)
    cov_2 = scale * make_cov(4, 4)
    cov_3 = scale * make_cov(2, 4)
    cov_4 = scale * make_cov(3, 1)

    mu_1 = np.asarray([-2, -1])
    mu_2 = np.asarray([2, 2])
    mu_3 = np.asarray([0, -2])
    mu_4 = np.asarray([-2, 2])

    cov = np.asarray([cov_1, cov_2, cov_3, cov_4])
    mu = np.asarray([mu_1, mu_2, mu_3, mu_4])
    y = np.random.choice(len(priors), size=nsamples, replace=True, p=priors)
    x = np.zeros([nsamples, 2])
    for samp in range(0, nsamples):
        mu_target = mu[y[samp]]
        cov_target = cov[y[samp]]
        x[samp] = np.random.multivariate_normal(mu_target, cov_target, 1)

    colors = ['red', 'green', 'blue', 'purple']
    if plot_data:
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('figures/q2/data_%s.png' % nsamples)

    return x, y


if __name__ == '__main__':
    for exp in tqdm(range(0, 30), desc="making data for experiments"):
        for train_size in [10, 100, 1000, 10000]:
            xtrain, ytrain, = generate_data(train_size, plot_data=(exp == 0))
            with open("q2data/trainset_%s_%s.npy" % (exp, train_size), 'wb') as f:
                np.save(f, xtrain)
                np.save(f, ytrain)
                f.close()

    xtest, ytest, = generate_data(100000)
    with open("q2data/test.npy", 'wb') as f:
        np.save(f, xtest)
        np.save(f, ytest)
        f.close()
