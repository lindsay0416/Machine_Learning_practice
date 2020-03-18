#python3 steven
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score

def getData():
    path = r'Iris.xlsx' #r'../../Iris.xlsx'  # should change the path accordingly
    rawdata = pd.read_excel(path)  # pip install xlrd
    #print ("data summary")
    #print (rawdata.describe().transpose())
    nrow, ncol = rawdata.shape
    print (nrow, ncol)
    return rawdata

def splitRowData(data):
    nRow, nCol = data.shape
    predictors = data.iloc[:, :nCol - 1]
    # print(predictors)
    target = data.iloc[:, -1]
    # print(target)

    pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=0.3, random_state=42)
    print('pred_train.shape = ', pred_train.shape)
    print('tar_train.shape = ', tar_train.shape)
    print('pred_test.shape = ', pred_test.shape)
    print('tar_test.shape = ', tar_test.shape)
    print(pred_train[0:5])

    return (pred_train, tar_train), (pred_test, tar_test)

def getModel():
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                        solver='adam', verbose=0, tol=1e-8, random_state=1,
                        learning_rate_init=.01)
    return mlp

def plotResult(scores_train,scores_test,loss):
    fig, ax = plt.subplots(3, sharex=True, sharey=True)
    ax[0].plot(scores_train)
    ax[0].set_title('Train mean accuracy')
    ax[1].plot(scores_test)
    ax[1].set_title('Test mean accuracy')
    ax[2].plot(loss)
    ax[2].set_title('Loss')
    fig.suptitle("Accuracy &Loss over epochs", fontsize=14)
    plt.show()

def train(x_train, y_train, x_test, y_test):
    mlp = getModel()

    N_EPOCHS = 25
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []

    epoch = 0
    while epoch < N_EPOCHS:

        mlp.partial_fit(x_train, np.ravel(y_train, order='C'), classes=N_CLASSES)

        accTrain = mlp.score(x_train, y_train)
        accTest = mlp.score(x_test, y_test)
        scores_train.append(accTrain) # SCORE TRAIN
        scores_test.append(accTest) # SCORE TEST
        print('epoch: ', epoch,'accTrain=',accTrain,'accTest=',accTest,'loss=',mlp.loss_)

        epoch += 1

    """ Plot """
    plotResult(scores_train,scores_test,mlp.loss_curve_)
    pass

def trainMiniBatch(x_train, y_train, x_test, y_test):
    mlp = getModel()

    N_TRAIN_SAMPLES = x_train.shape[0]
    N_EPOCHS = 25
    N_BATCH = 20
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            mlp.partial_fit(x_train[mini_batch_index:mini_batch_index + N_BATCH], np.ravel(y_train[mini_batch_index:mini_batch_index + N_BATCH], order='C'), classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        scores_train.append(mlp.score(x_train, y_train)) # SCORE TRAIN
        scores_test.append(mlp.score(x_test, y_test))  # SCORE TEST
        epoch += 1

    """ Plot """
    plotResult(scores_train,scores_test,mlp.loss_curve_)
    pass

def main():
    data  = getData()
    (x_train, y_train), (x_test, y_test) = splitRowData(data)

    #trainMiniBatch(x_train, y_train, x_test, y_test)
    train(x_train, y_train, x_test, y_test)

if __name__=='__main__':
    main()

