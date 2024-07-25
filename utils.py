import random

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, f1_score
from sklearn.utils import shuffle
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def plot_permutation_importance(clf, X, y, ax, random_state):
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=random_state, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()

    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X.columns[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")
    return ax

def cls_scores(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    train_macro_f1 = f1_score(y_train, model.predict(X_train), average='macro')
    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    return train_score, test_score, train_macro_f1, test_macro_f1

def reg_scores(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    mape_train = mean_absolute_percentage_error(y_train, model.predict(X_train))
    mape_test = mean_absolute_percentage_error(y_test, y_pred)
    return train_score, test_score, mape_train, mape_test

def reg_all_scores(model, X_train, y_train, X_test, y_test, thres):
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    r2_neg_test = r2_score(y_test[y_test<=thres], y_pred[y_test<=thres])
    r2_pos_test = r2_score(y_test[y_test>thres], y_pred[y_test>thres])
    mape_train = mean_absolute_percentage_error(y_train, model.predict(X_train))
    mape_test = mean_absolute_percentage_error(y_test, y_pred)
    mape_neg_test = mean_absolute_percentage_error(y_test[y_test<=thres], y_pred[y_test<=thres])
    mape_pos_test = mean_absolute_percentage_error(y_test[y_test>thres], y_pred[y_test>thres])
    return train_score, test_score, r2_neg_test, r2_pos_test, mape_train, mape_test, mape_neg_test, mape_pos_test

def cv_scores(tv_df, kf, cls, best_param, thres, random_state, load_model=False):
    kf_scores = []
    neg_idx = tv_df['fnc_ber_com_composite']<=thres
    pos_idx = (1-neg_idx).astype('bool')
    neg_df = shuffle(tv_df[neg_idx],random_state=random_state)
    pos_df = shuffle(tv_df[pos_idx],random_state=random_state)
    neg_l = round(len(neg_df)/kf)
    pos_l = round(len(pos_df)/kf)
    for k in range(kf):
        neg_val_df = neg_df.iloc[neg_l*k:neg_l*(k+1),:]
        neg_tr_df = neg_df.drop(neg_val_df.index)
        pos_val_df = pos_df.iloc[pos_l*k:pos_l*(k+1),:]
        pos_tr_df = pos_df.drop(pos_val_df.index)        

        train_df = pd.concat([neg_tr_df, pos_tr_df])
        val_df = pd.concat([neg_val_df, pos_val_df])

        X_train = train_df.drop(['fnc_ber_com_composite'], axis=1)
        y_train = train_df['fnc_ber_com_composite']

        X_val = val_df.drop(['fnc_ber_com_composite'], axis=1)
        y_val = val_df['fnc_ber_com_composite']
        if cls == 'extratree':
            model = ExtraTreesRegressor(**best_param, random_state=random_state)
        elif cls == 'gradientboost':
            model = GradientBoostingRegressor(**best_param, random_state=random_state)
        elif cls == 'model_selection':
            model = load_model
        model.fit(X_train, y_train)

        train_r2 = model.score(X_train, y_train)
        val_r2 = model.score(X_val, y_val)
        train_mape = mean_absolute_percentage_error(y_train, model.predict(X_train))
        val_mape = mean_absolute_percentage_error(y_val, model.predict(X_val))
        val_mape_neg = mean_absolute_percentage_error(y_val[y_val<=thres], model.predict(X_val)[y_val<=thres])
        val_mape_pos = mean_absolute_percentage_error(y_val[y_val>thres], model.predict(X_val)[y_val>thres])

        if cls == 'model_selection':
            kf_scores.append([str(model)[:str(model).index('(')], k, train_r2, val_r2, train_mape, val_mape, val_mape_neg, val_mape_pos])
            kf_scores_df = pd.DataFrame(kf_scores, columns=['Model', 'kfold', 'Train_R2', 'Val_R2', 'Train_MAPE', 'Val_MAPE','Val_MAPE_NEG', 'Val_MAPE_POS'])
        else:
            kf_scores.append([k, train_r2, val_r2, train_mape, val_mape, val_mape_neg, val_mape_pos])
            kf_scores_df = pd.DataFrame(kf_scores, columns=['kfold', 'Train_R2', 'Val_R2', 'Train_MAPE', 'Val_MAPE','Val_MAPE_NEG', 'Val_MAPE_POS'])
    
    return kf_scores_df


def bayescv(X, y, n_iter, model, random_state, cls='extratree'):
    if cls in ['extratree', 'gradientboost']:
        opt = BayesSearchCV(model,
                    {
                        'max_depth': Integer(3,12),
                        'min_samples_leaf': Integer(1,4),
                        'min_samples_split': Integer(2,5),
                        'n_estimators': Integer(50, 800),
                    },
                    cv=5,
                    n_iter=n_iter,
                    scoring='neg_mean_absolute_percentage_error',
                    random_state=random_state
                )
    elif cls == 'xgboost':
        opt = BayesSearchCV(model,
        {
            'gamma': Real(0.01, 0.35),
            'min_child_weight': Integer(1, 3),
            'max_depth': Integer(3,10),
            'subsample': Real(0.001, 1.0, prior='log-uniform'),
            'colsample_bytree': Real(0.001, 1.0, prior='log-uniform'),
            'n_estimators': Integer(50, 800),
        },
        cv=5,
        n_iter=n_iter,
        scoring='neg_mean_absolute_percentage_error',
        random_state=random_state
        )
    opt.fit(X, y)
    return opt
