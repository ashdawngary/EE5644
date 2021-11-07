from typing import Tuple

import numpy as np
import scipy.stats

# class means
mu = np.asarray([[1, 1, 1],
                 [-1, -1, 1],
                 [-1, 1, -1],
                 [1, -1, -1]])

covall = np.eye(3)


def min_perror(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    global mu, covall

    pdf0 = scipy.stats.multivariate_normal.pdf(x, mu[0, :], covall)
    pdf1 = scipy.stats.multivariate_normal.pdf(x, mu[1, :], covall)
    pdf2 = scipy.stats.multivariate_normal.pdf(x, mu[2, :], covall)
    pdf3 = scipy.stats.multivariate_normal.pdf(x, mu[3, :], covall)

    dist = [0, 0]
    confusion_matrix = np.zeros([4, 4])
    for i in range(0, y.shape[0]):
        true_label = y[i]
        pdfs = np.asarray([pdf0[i], pdf1[i], pdf2[i], pdf3[i]])
        classified = np.argmax(pdfs)

        if true_label == classified:
            dist[0] += 1
        else:
            dist[1] += 1
        confusion_matrix[classified, true_label] += 1

    return dist[1] / sum(dist), confusion_matrix


if __name__ == '__main__':
    with open("q1data/test.npy", 'rb') as f:
        xtest = np.load(f).T
        ytest = np.load(f).T
        f.close()

    perror, confusion = min_perror(xtest, ytest)
    print("min perror of test set: %s" % perror)
