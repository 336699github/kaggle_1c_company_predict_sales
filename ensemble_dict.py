
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import model_simple_neural_network 
import model_embedding_neural_network
from xgboost.sklearn import XGBRegressor
from features import non_lr_feats,lr_feats,embedding_feats

non_lr_feats=non_lr_feats
lr_feats=lr_feats
embedding_feats=embedding_feats


lr_params={'normalize':True,'n_jobs':-1}
xgb1_params={
    "n_jobs":-1,
    "n_estimators":500,
    "max_depth":3,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "learning_rate":0.3,
}
xgb2_params={
    "n_jobs":-1,
    "n_estimators":500,
    "max_depth":7,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "learning_rate":0.3,
}
rfr_params={
    "n_jobs":-1,
    "n_estimators":100,
    "max_features":'sqrt',
    "min_samples_leaf":50,
}
knn_params={
    "n_jobs":-1
}
xgb1=XGBRegressor(**xgb1_params)
xgb2=XGBRegressor(**xgb2_params)
lr=LinearRegression(**lr_params)
rfr=RandomForestRegressor(**rfr_params)
knn=KNeighborsRegressor(**knn_params)
simple_nn=model_simple_neural_network._build_keras_model(len(lr_feats))
simple_nn.compile(loss='mean_squared_error', optimizer='adam')
embedding_nn=model_embedding_neural_network._build_keras_model()
embedding_nn.compile(loss='mean_squared_error', optimizer='adam')



ensemble_dict={
    'xgb1':(xgb1, non_lr_feats),
    'xgb2':(xgb2, non_lr_feats),
    'rfr':(rfr,non_lr_feats),
    'lr':(lr,lr_feats),
    'knn':(knn,lr_feats),
    'simple_nn':(simple_nn,lr_feats),
    'embedding_nn':(embedding_nn,embedding_feats)
}
