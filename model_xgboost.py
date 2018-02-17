import pandas as pd 
import numpy as np
import time 
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV,PredefinedSplit, KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import _pickle as cPickle
import xgboost 
def get_rmse(ground_truth, predictions):
    mse=mean_squared_error(ground_truth, predictions)
    return np.sqrt(mse)
RMSE=make_scorer(get_rmse, greater_is_better=False)

def go(data_dict,feats_to_use, params={"seed":0,"silent":False,"n_jobs":-1},
 parameter_tuning=False):
    
    '''
    if with_gpu:
        xgb = XGBRegressor(seed=0, silent=False, tree_method='gpu_hist', n_gpus=-1)
    else:
        xgb = XGBRegressor(seed=0, silent=False, n_jobs=-1)
    '''
    X_train=data_dict['X_train'][feats_to_use].copy()
    y_train=data_dict['y_train'].copy()
    X_test=data_dict['X_test'][feats_to_use].copy()
    X_val=data_dict['X_val'][feats_to_use].copy()
    y_val=data_dict['y_val'].copy()

    
    
    if parameter_tuning:
        fit_params={
        "early_stopping_rounds":10, 
        "eval_metric" : "rmse", 
        "eval_set" : [(X_val,y_val)]}
        xgb=XGBRegressor() 
        train_val_features=pd.concat([X_train,X_val])
        train_val_labels=pd.concat([y_train,y_val])
        test_fold = np.zeros(train_val_features.shape[0])   # initialize all index to 0
        test_fold[:X_train.shape[0]] = -1   # set index of training set to -1, indicating not to use it in validation
        
        ps=PredefinedSplit(test_fold=test_fold)
        X_train=data_dict['X_train'][feats_to_use]
        y_train=data_dict['y_train']
        X_test=data_dict['X_test'][feats_to_use]
        grid=GridSearchCV(xgb,params,fit_params=fit_params,scoring=RMSE , cv=ps, verbose=32, n_jobs=-1)
        start=time.time()
        grid.fit(train_val_features,train_val_labels)
        elapsed=time.time()-start
        print (elapsed)
        print ('best params:',grid.best_params_)
        print ('best score:',grid.best_score_)

        return grid.best_params_, grid.best_estimator_
        
    else:
        xgb=XGBRegressor(**params)
        print (xgb)
    
        print ('start xgboost training')
        start=time.time()
        eval_set=[(X_val,y_val)]
        xgb.fit(X_train,y_train, eval_set=eval_set,eval_metric='rmse',early_stopping_rounds=30)
        elapsed=time.time()-start
        print (elapsed)
        data_dict['y_pred']=np.exp(xgb.predict(X_test))-1

        #generate submission
        data_dict['X_test']['item_cnt_month']=data_dict['y_pred']
        test=pd.read_csv('test.csv')
        submission=pd.merge(test,data_dict['X_test'], 
            on=['shop_id','item_id'],how='left')[['ID','item_cnt_month']]
        return submission, xgb
def get_feature_importances(feats_to_use, xgb):
    feature_importances=list(zip(feats_to_use,xgb.feature_importances_))
    feature_importances.sort(key=lambda x:x[1], reverse=True)
    print (feature_importances)
    with open('../1c_data/xgb_feature_importances.pkl','wb')as handle:
        cPickle.dump(feature_importances,handle) 
        
if __name__ == "__main__":
    with open('../1c_data/data_dict.pkl','rb') as handle:
        data_dict=cPickle.load(handle)
    feats_to_use=data_dict['X_val'].columns
    params={"n_jobs":-1,
    "silent":True,
    "n_estimators":500,
    "max_depth":3,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "learning_rate":0.3,
    "min_child_weight":1,
    "gamma":0}
    submission,xgb=go(data_dict, feats_to_use,params=params)
    submission.to_csv('submission_xgb_%s.csv'%time.strftime("%Y%m%d-%H%M%S"),index=False)
    with open('xgb_depth3.pkl','wb') as handle:
        cPickle.dump(xgb,handle,protocol=-1)
    get_feature_importances(feats_to_use,xgb)
    