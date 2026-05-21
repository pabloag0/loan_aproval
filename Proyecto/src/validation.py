import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 

##############################################################################
# CROSS VALIDATION

def cross_validate(X, y, train: function, predict: function, folds=5):

    for i in range(folds):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    



    pass

# Dijo en clase que sería interesante implementarlo (mas nota)
def gridsearch():
    pass