import numpy as np
import keras
import keras.layers
import keras.losses
import q1_minperror
import matplotlib.pyplot as plt


def construct_pmodel(percep):
    model = keras.Sequential(
        layers=[
            keras.layers.Dense(percep, activation='elu'),
            keras.layers.Dense(4, activation='softmax')  # output layer
        ]
    )
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


def model_order_selection(xData, yData, max_percep=20, nfold=10, epoch_per_part=10):
    prior_errors = []
    serie = {}
    for npercep in range(1, max_percep + 1):
        print("kfold with %s percep" % npercep)
        score_n = kfold_evaluate(npercep, xData, yData, nfolds=nfold, epoch_per_part=epoch_per_part)
        prior_errors.append(score_n)
        serie[npercep] = score_n
        if len(prior_errors) > 5:
            print("prior scores: ", prior_errors[-5:])
            print("decrease factor: %s" % (prior_errors[-1] / prior_errors[-5]))

            if prior_errors[-1] > prior_errors[-5] * 0.95:  # 2% dec at least per 5 epochs
                print(np.asarray(prior_errors))
                amin = np.argmin(np.asarray(prior_errors)) + 1

                tpercep = amin

                print("selected %s perceptrons" % tpercep)

                t_model = construct_pmodel(tpercep)
                t_model.fit(xData, yData, epochs=epoch_per_part)
                return serie, t_model

    t_model = construct_pmodel(max_percep)
    t_model.fit(xData, yData)
    return serie, t_model


def model_select_and_test(xTrain, yTrain, train_labels, xTest, yTest, pref='dataset'):
    points, model = model_order_selection(xTrain, yTrain)

    mpe, _ = q1_minperror.min_perror(xTrain, train_labels)

    triplet = [[], [], []]
    for (percep, perror) in points.items():
        triplet[0].append(percep)
        triplet[1].append(perror)
        triplet[2].append(mpe)

    x_bounds = np.asarray(triplet[0])
    perrors = np.asarray(triplet[1])
    mpe_bound = np.asarray(triplet[2])

    loss, accuracy = model.evaluate(xTest, yTest)

    print("final test accuracy: ", accuracy)
    print("final test loss: ", loss)

    plt.plot(x_bounds, perrors)
    plt.plot(x_bounds, mpe_bound)
    plt.xlabel('number perceptrons')
    plt.ylabel('p-error')
    plt.savefig('figures/q1/q1kfold_%s.png' % pref)
    return 1 - accuracy


def kfold_evaluate(p, xData, yData, nfolds=10, epoch_per_part=10):
    samples = xData.shape[0]
    data_ixs = np.arange(samples)
    np.random.shuffle(data_ixs)

    folds = np.array_split(data_ixs, nfolds)
    exp = []
    for experiment in range(0, nfolds):
        validateFold = folds[experiment]

        xValidate = xData[validateFold]
        yValidate = yData[validateFold]

        train_mask = np.ones(samples, bool)
        train_mask[validateFold] = False

        xTrain = xData[train_mask]
        yTrain = yData[train_mask]

        model = construct_pmodel(p)
        # print("datax: %s validatex: %s" % (xTrain.shape[0], xValidate.shape[0]))
        model.fit(xTrain, yTrain, epochs=epoch_per_part)

        loss, accuracy = model.evaluate(xValidate, yValidate)
        exp.append([validateFold.shape[0], 1 - accuracy])

    sum_score = 0
    for experiment in exp:  # weighted average
        sum_score += (experiment[0] / samples) * experiment[1]
    return sum_score


if __name__ == '__main__':

    with open("q1data/test.npy", 'rb') as f:
        xtest = np.load(f).T
        ltest = np.load(f).T
        otest = np.load(f).T
        f.close()

    test_sizes = [100, 200, 500, 1000, 2000, 5000]
    test_results = []
    min_perror, _ = q1_minperror.min_perror(xtest, ltest)
    test_min_perror = [min_perror] * len(test_sizes)

    for size in test_sizes:
        with open("q1data/trainset_%s.npy" % size, 'rb') as f:
            xtrain = np.load(f).T
            train_labels = np.load(f).T
            otrain = np.load(f).T
            f.close()
        score = model_select_and_test(xtrain, otrain, train_labels, xtest, otest, pref=str(size))
        test_results.append(score)

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.plot(np.asarray(test_sizes), np.asarray(test_results))
    ax.plot(np.asarray(test_sizes), np.asarray(test_min_perror))
    ax.figure.savefig("figures/q1/data_minperrors.png")
