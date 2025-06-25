import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline


def read_dataset(input, classifier_type):
    DE_df = pd.read_csv(input, sep = ",", index_col=0).transpose()
    if classifier_type == "healthy":
        labels = {"WT":0, "FTA":1}
    else:
        labels = {"FTC":1, "PTC":0}
    Y = [labels[i.split("_")[-2]] for i in list(DE_df.index)]
    DE_df["Y"] = Y
    return DE_df


def test_model_params(dataset, params, model, logfc_table, n_features = None):
    data_shuffled = dataset.sample(frac = 1)
    labels = data_shuffled["Y"]
    steps = list()
    steps.append(('scaler', MinMaxScaler()))
    data_shuffled = data_shuffled.drop("Y", axis = 1)[logfc_table[:n_features]]
    steps.append(('model', model))
    pipeline = Pipeline(steps=steps)
    search = GridSearchCV(pipeline, param_grid = params, cv = 5, n_jobs = 12)
    search.fit(data_shuffled, labels)
    return search


def test_feature_num(dataset, params, model, feat_range, logfc_table):
    df = pd.DataFrame()
    for feat in feat_range:
        res = test_model_params(dataset, params, model, logfc_table, feat)
        new_df = pd.DataFrame(res.cv_results_["params"])
        new_df["test_error"] = (1-res.cv_results_["mean_test_score"])*100
        new_df["features"] = [feat]*(new_df.shape[0])
        df = pd.concat([df, new_df])
    return df



#WT_FTA = read_dataset("subSVMs_definitivos/WT_FTA.csv", "healthy")
PTC_FTC = read_dataset("subSVMs_definitivos/PTC_FTC.csv", "patho")

DE_PTC_FTC = pd.read_csv("subSVMs_definitivos/DE_PTC_FTC.csv")
rnk = list(DE_PTC_FTC["Row.names"])

#DE_WT_FTA = pd.read_csv("subSVMs_definitivos/DE_WT_FTA.csv")
#rnk = list(DE_WT_FTA["Row.names"])

rango_C = [3.3]
rango_g = [k/100 for k in range(22,34,1)]
params = {"model__C": rango_C, "model__kernel": ["rbf"], "model__gamma": rango_g}

average_results =pd.DataFrame()
for i in range(1000):
    print(i)
    res = test_feature_num(PTC_FTC, params, SVC(), range(145, 161, 1), rnk)
    if i==0:
        average_results = res
    else:
        average_results[f'test_error_{i}'] = res["test_error"]

average_results['average_error'] = average_results.filter(like='test_error').mean(axis=1)
average_results['std_error'] = average_results.filter(like='test_error').std(axis=1)

average_results.to_csv("genes/patho/resultados_svm33.csv")
