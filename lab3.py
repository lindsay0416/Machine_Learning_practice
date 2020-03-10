import pandas as pd
from scipy.io import arff
from sklearn.linear_model import LogisticRegression

def getData(file):
    #data = arff.loadarff(file)
    #df = pd.DataFrame(data[0])
    df = pd.read_csv(file,encoding = "utf-8")  #ISO-8859-1
    #print(df.head())
    return df

def main():
    data = r'../../iris.csv'  #r'../../data/iris.arff'
    df = getData(data)

    X = df.iloc[:, :4]
    y = df.iloc[:, 4]
    test = df.iloc[98:103, :4]

    print('X=',X)
    print('y=',y)
    print('test=',test)

    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X, y)
    print(LR.predict(test))


if __name__ == "__main__":
    main()
