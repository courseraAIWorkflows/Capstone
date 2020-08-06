#!/usr/bin/env python
"""
example performance monitoring script
"""

import os, sys, pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import wasserstein_distance
from cslib import engineer_features

def get_latest_train_data():
    """
    load the data used in the latest training
    """

    data_file = os.path.join("models",'latest-train.pickle')

    if not os.path.exists(data_file):
        raise Exception("cannot find {}-- did you train the model?".format(data_file))

    with open(data_file,'rb') as tmp:
        data = pickle.load(tmp)

    return(data)

    
def get_monitoring_tools(X,y):
    """
    determine outlier and distance thresholds
    return thresholds, outlier model(s) and source distributions for distances

    """
    X_pre = X
    xpipe = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor())])
    xpipe.fit(X_pre,y)
    
    bs_samples = 1000
    outliers_X = np.zeros(bs_samples)
    wasserstein_X = np.zeros(bs_samples)
    wasserstein_y = np.zeros(bs_samples)
    
    for b in range(bs_samples):
        n_samples = int(np.round(0.8 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y_bs=y[mask]
        X_bs=X[mask]
    
        test1 = xpipe.predict(X_bs)
        wasserstein_X[b] = wasserstein_distance(X_pre.values.flatten(),X_bs.values.flatten())
        wasserstein_y[b] = wasserstein_distance(y,y_bs.flatten())
        outliers_X[b] = 100 * (1.0 - (test1[test1==1].size / test1.size))

    ## determine thresholds as a function of the confidence intervals
    outliers_X.sort()
    outlier_X_threshold = outliers_X[int(0.975*bs_samples)] + outliers_X[int(0.025*bs_samples)]

    wasserstein_X.sort()
    wasserstein_X_threshold = wasserstein_X[int(0.975*bs_samples)] + wasserstein_X[int(0.025*bs_samples)]

    wasserstein_y.sort()
    wasserstein_y_threshold = wasserstein_y[int(0.975*bs_samples)] + wasserstein_y[int(0.025*bs_samples)]
    
    to_return = {"outlier_X": np.round(outlier_X_threshold,1),
                 "wasserstein_X":np.round(wasserstein_X_threshold,2),
                 "wasserstein_y":np.round(wasserstein_y_threshold,2),
                 "pipe_X":xpipe,
                 "X_source":X_pre,
                 "y_source":y,
                 "latest_X":X,
                 "latest_y":y}
    return(to_return)


if __name__ == "__main__":

    ## get latest training data
    data = get_latest_train_data()
    y = data['y']
    X = data['X']

    ## get performance monitoring tools
    pm_tools = get_monitoring_tools(X,y)
    print("outlier_X",pm_tools['outlier_X'])
    print("wasserstein_X",pm_tools['wasserstein_X'])
    print("wasserstein_y",pm_tools['wasserstein_y'])
    
    print("done")
    
