import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR


def getData():
    path = r'./dataBase/pima-indians-diabetes.data.csv'  # should change the path accordingly
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    rawdata = pd.read_csv(path, names=names)  # pip install xlrd
    #print("data summary")
    #print (rawdata.describe())
    print('rawdata.shape = ', rawdata.shape)
    return rawdata

def MLPClassifierModel(pred_train, pred_test, tar_train, tar_test):
    #clf = MLPClassifier(activation='logistic', learning_rate_init=0.1)
    clf = MLPClassifier(activation='logistic', learning_rate_init=0.1, solver='sgd', alpha=1e-5, hidden_layer_sizes=(2,2), random_state=1)
    clf.fit(pred_train, np.ravel(tar_train, order='C'))
    predictions = clf.predict(pred_test)
    #print("Accuracy score of our model with MLP :", accuracy_score(tar_test, predictions))
    scores = cross_val_score(clf, pred_test, tar_test, cv=10)
    #print("Accuracy score of our model with MLP under cross validation :", scores.mean())
    return accuracy_score(tar_test, predictions)

def get_accuracy(target_train, target_test, predicted_test,predicted_train):
    return MLPClassifierModel(predicted_train,predicted_test,target_train,target_test)

def FeatureExtractChiSquare(X,Y, N=3):
    print("*"*50,FeatureExtractChiSquare.__name__)
    test = SelectKBest(score_func=chi2,k=N)
    fit = test.fit(X, Y)
    #print('X.shape = ',X.shape)
    #print('Y.shape = ', Y.shape)

    np.set_printoptions(precision=3)
    print('scores = ', fit.scores_)

    features = fit.transform(X)
    print('after chi square,features.shape = ',features.shape)
    return features,Y

def FeatureExtract_RFE(X,Y,N=3):
    print("*" * 50, FeatureExtract_RFE.__name__)
    #estimator = SVR(kernel="linear")
    estimator = SVC(kernel="linear", C=1)
    rfe = RFE(estimator, n_features_to_select=N, step=1)

    fit = rfe.fit(X, Y)
    features = fit.transform(X)
    # print(fit.classes_)
    print("Num Features: %d" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))
    #print(fit.support_)
    print('after FRE features.shape = ',features.shape)
    return features, Y

def FeatureExtract_PCA(X,Y,N=3):
    print("*" * 50, FeatureExtract_PCA.__name__)
    rfe = PCA(n_components=N)
    fit = rfe.fit(X, Y)
    features = fit.transform(X)

    print("Explained Variance: %s" % (fit.explained_variance_))
    print("Explained Variance ratio: %s" % (fit.explained_variance_ratio_))
    #print(fit.components_)
    print('after PCA features.shape = ', features.shape)
    return features, Y

def FeatureExtract_ETC(X,Y):
    print("*" * 50, FeatureExtract_ETC.__name__)
    model = ExtraTreesClassifier(max_depth=3, min_samples_leaf=2)
    fit = model.fit(X, Y)
    print('Feature importance:', model.feature_importances_)
    t = SelectFromModel(fit, prefit=True)  # extra step required as we are using an ensemble classifier here
    features = t.transform(X)

    print('after ETC features.shape = ', features.shape)
    return features, Y

def train(data):
    nRow, nCol = data.shape
    predictors = data.iloc[:, :nCol - 1]
    target = data.iloc[:, -1]
    #print(predictors[:5])
    #print(target[:5])

    pred_train, pred_test, tar_train, tar_test = train_test_split(predictors,target,test_size=0.3, random_state=4)
    accuracy = get_accuracy(tar_train,tar_test, pred_test, pred_train)
    print("Accuracy score with MLP :", accuracy)

    X = data.values[:, 0:nCol - 1]
    Y = data.values[:, -1]
    #print(X)
    #print(Y)

    FeatureExtractChiSquareAcc(X, Y,4)
    FeatureExtract_RFEAcc(X, Y, 3)
    FeatureExtract_PCAAcc(X, Y, 3)
    FeatureExtract_ImportanceETC_Acc(X,Y)

def FeatureExtractChiSquareAcc(X,Y,N=3):
    X, Y = FeatureExtractChiSquare(X, Y, N)  # after feature selection, features from 8-->3
    pred_train, pred_test, tar_train, tar_test = train_test_split(X, Y, test_size=0.3, random_state=4)
    accuracy = get_accuracy(tar_train, tar_test, pred_test, pred_train)
    print("Accuracy score with MLP after Chi Square :", accuracy)

def FeatureExtract_RFEAcc(X,Y,N=3):
    X, Y = FeatureExtract_RFE(X, Y, N)  # after feature selection, features from 8-->3
    pred_train, pred_test, tar_train, tar_test = train_test_split(X, Y, test_size=0.3, random_state=4)
    accuracy = get_accuracy(tar_train, tar_test, pred_test, pred_train)
    print("Accuracy score with MLP after RFE :", accuracy)

def FeatureExtract_PCAAcc(X,Y,N=3):
    X, Y = FeatureExtract_PCA(X, Y, N)  # after feature selection, features from 8-->N
    pred_train, pred_test, tar_train, tar_test = train_test_split(X, Y, test_size=0.3, random_state=4)
    accuracy = get_accuracy(tar_train, tar_test, pred_test, pred_train)
    print("Accuracy score with MLP after PCA :", accuracy)

def FeatureExtract_ImportanceETC_Acc(X,Y):
    X, Y = FeatureExtract_ETC(X, Y)
    pred_train, pred_test, tar_train, tar_test = train_test_split(X, Y, test_size=0.3, random_state=4)
    accuracy = get_accuracy(tar_train, tar_test, pred_test, pred_train)
    print("Accuracy score with MLP after Importance ETC :", accuracy)


def main():
    data = getData()
    train(data)

if __name__=='__main__':
    main()
