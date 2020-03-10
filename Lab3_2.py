import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score,auc
from sklearn.metrics import roc_curve,roc_auc_score, plot_roc_curve


def getData():
    path = r'../../Iris.xlsx'  # should change the path accordingly
    rawdata = pd.read_excel(path)  # pip install xlrd
    print ("data summary")
    print (rawdata.describe())
    nrow, ncol = rawdata.shape

    print (nrow, ncol)
    return rawdata

def showCorr(data):
    print ("\n correlation Matrix")
    print (data.corr())
    data.hist()
    plt.show()

def scatter_matrix(data):
    pd.plotting.scatter_matrix(data, figsize=[8, 8])
    plt.show()

def boxplot(data):
    # boxplot
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(data.values)
    ax.set_xticklabels(['Petal Length', 'Petal Width', 'Sepal Length', 'Sepal Width', 'Class'])
    plt.show()

def train(data):
    nRow, nCol = data.shape
    predictors = data.iloc[:, :nCol - 1]
    #print(predictors)
    target = data.iloc[:, -1]
    #print(target)

    pred_train, pred_test, tar_train, tar_test = train_test_split(predictors,target,test_size=0.3, random_state=42)
    # print(pred_train.shape)
    # print(tar_train.shape)
    # print(pred_test.shape)
    # print(tar_test.shape)

    split_threshold = 3
    fpr = dict()  # store false positive rate in a dictionary object
    tpr = dict()  # likewise, store the true positive rate
    roc_auc = dict()
    print('*'*60)
    for i in range(2, split_threshold):
        classifier = DecisionTreeClassifier(criterion='entropy',min_samples_split=2)  # configure the classifier
        classifier = classifier.fit(pred_train, tar_train)  # train a decision tree model
        predictions = classifier.predict(pred_test)  # deploy model and make predictions on test set
        prob = classifier.predict_proba(pred_test)  # obtain probability scores for each sample in test set

        print("Accuracy score of our model with Decision Tree:", i, accuracy_score(tar_test, predictions))
        precision = precision_score(y_true=tar_test, y_pred=predictions, average='micro')
        print("Precision score of our model with Decision Tree :", precision)

        recall = recall_score(y_true=tar_test, y_pred=predictions, average='micro')
        print("Recall score of our model with Decision Tree :", recall)

    for x in range(3):
        fpr[x], tpr[x], _ = roc_curve(tar_test[:], prob[:, x], pos_label=x)
        roc_auc[x] = auc(fpr[x], tpr[x])
        print("AUC values of the decision tree", roc_auc[x])
        plt.plot(fpr[x], tpr[x], color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc[x])
        plt.show()


    pass


def main():
    rawdata = getData()
    #showCorr(rawdata)
    #scatter_matrix(rawdata)
    #boxplot(rawdata)
    train(rawdata)



if __name__ == "__main__":
    main()
