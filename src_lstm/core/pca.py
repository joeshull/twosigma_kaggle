import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler


def scalethis(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def PCA_this(data):
    pca = PCA()
    pca.fit(data)
    return pca

def plot_2d(pca_object, data, labels):
    reduced = pca_object.fit_transform(data)
    exp_var = np.sum(pca_object.explained_variance_ratio_[:2])
    X1 = reduced[:,0]
    X2 = reduced[:,1]
    l = []
    for x1, x2, label in zip(X1, X2, labels):
        trace0 = go.Scatter(
            x = x1,
            y = x2,
            mode = 'markers',
            text = label)
        l.append(trace0)

    layout = go.Layout(
        title = 'PCA for All Stocks (Mean Values), 59% Variance Explained',
        hovermode = 'closest',
        xaxis=dict(
                title = 'PC #1'),
        yaxis=dict(
            title = 'PC #2',), 
        showlegend=False)
    fig = go.Figure(data=l, layout=layout)
    py.iplot(fig)


if __name__ == '__main__':
    
    df = pd.read_pickle('../data/init_train_data.pkl')
    agg = df.groupby('assetName').agg(np.mean)
    labels = agg.index.values
    agg.drop(columns=['time', 'close', 'open', 'universe'], inplace=True)
    scaled = scalethis(agg)
    pca = PCA_this(agg)

    fig, ax = plt.subplots(figsize=(12,12))
    plot_2d(pca, agg, ax)
    plt.show()



