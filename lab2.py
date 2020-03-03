import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def getData():
    csv_data = \
        ''' A, B, C, D
            1.0, 2.0, 3.0, 4.0
            5.0, 6.0,, 8.0
            10.0, 11.0, 12.0,
            15, 16, 17, 18'''

    df = pd.read_csv(StringIO(csv_data))
    print(df)
    return df

def test1():
    df = getData()
    print(df.isnull().sum())
    print('*' * 50)
    print(df.dropna(axis=0))
    print('*' * 50)
    print(df.dropna(axis=1))
    print('*' * 50)
    print(df.dropna(how='all'))
    print('*' * 50)
    print(df.dropna(thresh=5))

    print('*' * 50)
    #print(df.dropna(subset=['D']))  #error
    df = df.dropna()
    print(df)

    return df

def test2():
    csv_data = \
        ''' A, B, C, D
            1.0, 2.0, ,
            5.0, 6.0, ,
            NaN, NaN, NaN,'''

    df = pd.read_csv(StringIO(csv_data))
    print(df)
    print('*' * 50)
    print(df.dropna())
    #print(df.dropna(axis=1))
    print(df.dropna(how='all'))
    pass


def test3(): #input values when nan
    df = getData()
    print('*' * 50, 'start test3----------------')
    print(df)
    imr = SimpleImputer(missing_values=np.nan, strategy='most_frequent')  #most_frequent  #mean
    imr = imr.fit(df.values)
    input_data = imr.transform(df.values)
    print(input_data)
    pass

def test4():
    df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']])

    print(df)
    print('columns=',df.columns)
    print('index=', df.index)

    df.columns = ['color', 'size', 'price', 'class']  #settting columns
    print(df)

    size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1}

    df['size'] = df['size'].map(size_mapping)
    print(df)
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['class'].values)
    print(y)

    X = df[['color', 'size', 'price']].values
    print(X)

    ohe = ColumnTransformer([('anyname', OneHotEncoder(), [0])], remainder='passthrough')
    print(ohe.fit_transform(X))
    pass

def OhotTest():
    X = [['a', 1, 100.1],
         ['b', 2, 100.2]]
    print(X)
    ohe = ColumnTransformer([('anyname', OneHotEncoder(), [0])], remainder='passthrough')
    print(ohe.fit_transform(X))
    pass

def test5():
    #df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine = pd.read_csv(r'H:\Data\wine.data')

    print(df_wine)
    df_wine.columns = ['Class label', 'Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']

    print('Class labels', np.unique(df_wine['Class label']))
    print(df_wine.head())

    X, y = df_wine.iloc[:5, 1:].values, df_wine.iloc[:5, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size = 0.3,
    random_state = 0,
    stratify = y)

    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)
    print(X_train)
    print(X_train_norm)
    pass

def scatterplot(x_data, y_data, x_label="", y_label="", title="", color="r", yscale_log=False):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s=10, color=color, alpha=0.75)

    if yscale_log == True:
        ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def lineplot(x_data, y_data, x_label="", y_label="", title=""):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x_data, y_data, lw = 2, color = '#539caf', alpha = 1)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def histogram(x_data, n_bins, cumulative=False, x_label = "", y_label = "", title = ""):
    _, ax = plt.subplots()
    ax.hist(x_data, n_bins, cumulative = cumulative, color = '#539caf')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)

def test6Plot():
    # createdata
    N = 100
    x_data = np.random.rand(N)
    y_data = np.random.rand(N)
    #plt.show(scatterplot(x_data, y_data,yscale_log=False))

    #plt.show(lineplot(x_data, y_data, x_label = "x_label", y_label="y_label", title="title"))
    plt.show(histogram(x_data, n_bins=10))
    pass


if __name__ == "__main__":
    #test1()
    #test2()
    #test3()
    #test4()
    #OhotTest()
    #test5()
    test6Plot()