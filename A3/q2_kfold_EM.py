from typing import List, Tuple

import sklearn.mixture as mixture
import numpy as np
import numpy.linalg as linalg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm


def plot_model_against_data(model, data, outfile=None):
    if outfile is None:
        raise AssertionError("outfile cannot be none!")
    fig, axes = plt.subplots()

    # display predicted scores by the model as a contour plot
    x = np.linspace(-10, 10)
    y = np.linspace(-10, 10)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -model.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    )

    CB = plt.colorbar(CS, shrink=0.8, extend="both")
    axes.scatter(data[0, :], data[1, :], 0.8)
    axes.title.set_text("Negative log-likelihood contours of model")

    axes.axis("tight")

    axes.figure.savefig(outfile)

    plt.close(fig)


def fit_model_to_k(data: np.ndarray, k: int, num_starts: int = 4) -> mixture.GaussianMixture:
    # fit model to k gaussian components
    mix = mixture.GaussianMixture(n_components=k,
                                  covariance_type='diag',
                                  n_init=num_starts).fit(data.T)
    return mix


def eval_gm_loglikeli(model: mixture.GaussianMixture, test: np.ndarray) -> float:
    llh_set = model.score_samples(test.T)
    return -np.sum(llh_set, axis=None)


def kfold_exp(k_compoonents: int, data: np.ndarray, nfolds=5) -> float:
    data = data.T  # oop

    samples = data.shape[0]
    data_ixs = np.arange(samples)
    np.random.shuffle(data_ixs)

    folds = np.array_split(data_ixs, nfolds)
    exp = []
    for experiment in range(0, nfolds):
        validateFold = folds[experiment]

        xValidate = data[validateFold]

        train_mask = np.ones(samples, bool)
        train_mask[validateFold] = False

        xTrain = data[train_mask]

        model = fit_model_to_k(xTrain.T, k_compoonents)

        llh_sum = eval_gm_loglikeli(model, xValidate.T)

        exp.append([validateFold.shape[0], llh_sum])

    sum_score = 0
    for experiment in exp:  # weighted average
        sum_score += (experiment[0] / samples) * experiment[1]
    return sum_score


# output best component
def component_selection(components_to_check: List[int], data: np.ndarray, test_data: np.ndarray, plot_selections=False) \
        -> Tuple[int, mixture.GaussianMixture]:
    comp_to_score = {}
    comp_to_testLLH = {}

    for component in components_to_check:
        comp_score = kfold_exp(component, data, nfolds=10)
        comp_to_score[component] = comp_score

        component_model = fit_model_to_k(data, component)
        comp_to_testLLH[component] = eval_gm_loglikeli(component_model,
                                                       test_data)
        if plot_selections:
            plot_model_against_data(component_model, data,
                                    outfile='figures/q2/modelfits/%s/%s_merged_pdf.png' % (data.shape[1], component))
    scores = list(map(lambda x: comp_to_score[x], components_to_check))
    testLLH = list(map(lambda x: comp_to_testLLH[x], components_to_check))
    if plot_selections:
        fig, ax = plt.subplots()
        ax.plot(np.asarray(components_to_check), np.asarray(scores))

        ax.figure.savefig("figures/q2/component_selection_fold_performance/%s.png" % data.shape[1])

        fig, ax = plt.subplots()
        ax.plot(np.asarray(components_to_check), np.asarray(testLLH))
        ax.figure.savefig("figures/q2/component_selection_test_performance/%s.png" % data.shape[1])

    minix = np.argmin(scores)
    n_comp: int = components_to_check[minix]
    component_model: mixture.GaussianMixture = fit_model_to_k(data, component)

    return n_comp, component_model, comp_to_testLLH[n_comp]


if __name__ == '__main__':

    with open("q2data/test.npy", 'rb') as f:
        xtest = np.load(f).T
        ltest = np.load(f).T
        f.close()

    conf_matrix = np.zeros([4, 6])

    for exp_trial in tqdm(range(0, 30), desc="model selection experiment", position=0):  # run 30 experiements
        test_sizes = [10, 100, 1000, 10000]
        test_results = []
        best_components = []
        for ix, size in enumerate(test_sizes):
            with open("q2data/trainset_%s_%s.npy" % (exp_trial, size), 'rb') as f:
                xtrain = np.load(f).T
                train_labels = np.load(f).T
                f.close()
            (best_ncomp, best_model, test_score) = component_selection([1, 2, 3, 4, 5, 6], xtrain, xtest,
                                                                       plot_selections=exp_trial == 0)
            test_results.append(test_score)
            best_components.append(best_ncomp)
            conf_matrix[ix][best_ncomp - 1] += 1
        if exp_trial == 0:
            fig, ax = plt.subplots()
            ax.scatter(np.asarray(test_sizes), np.asarray(best_components))
            ax.figure.savefig("figures/q2/components_selected_per_test.png")

    print("Model selection matrix: ")
    print(conf_matrix)
    for (ix, size) in enumerate([10, 100, 1000, 10000]):
        fig, ax = plt.subplots()
        ax.bar([1, 2, 3, 4, 5, 6], conf_matrix[ix, :])
        ax.title.set_text('Model Selection over 30 experiments for %s Train set' % size)
        ax.figure.savefig("figures/q2/model_selec_%s.png" % size)
        plt.close(fig)

