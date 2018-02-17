
import numpy as np
import pandas as pd 
import gc 
import _pickle as cPickle
import time
from sklearn.metrics import mean_squared_error
from ensemble_dict import ensemble_dict
from sklearn.linear_model import LinearRegression
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
import features


def RMSE(truth, pred):
    return np.sqrt(mean_squared_error(truth,pred))

def gen_pairwise_diff(X_train_level2,X_test_level2):
    assert X_train_level2.shape[1]==X_test_level2.shape[1]
    train_diff_list=[]
    test_diff_list=[]
    for i, j in combinations(range(X_test_level2.shape[1]), 2):
        train_diff_list.append(X_train_level2[:,i]-X_train_level2[:,j])
        test_diff_list.append(X_test_level2[:,i]-X_test_level2[:,j])
    train_diff=np.column_stack(train_diff_list)
    test_diff=np.column_stack(test_diff_list)
    X_train_level2=np.concatenate((X_train_level2,train_diff),axis=1)
    X_test_level2=np.concatenate((X_test_level2,test_diff),axis=1)
    return X_train_level2,X_test_level2




def gen_ensemble_data(data_dict,date_train,date_train_level2,ensemble_dict,model_list=[]):
    try:
        X_train=pd.concat([data_dict['X_train'],data_dict['X_val']])
        y_train=pd.concat([data_dict['y_train'],data_dict['y_val']])
    except:
        X_train=np.concatenate((data_dict['X_train'],data_dict['X_val']),axis=0)
        y_train=np.concatenate((data_dict['y_train'],data_dict['y_val']),axis=0)
    X_test=data_dict['X_test']
    
    print ('X_train shape',X_train.shape)
    print ('y_train shape',y_train.shape)
                   
    print ('generating test meta features')
    test_pred_lst=[]
    for model in model_list:
        print ('fitting model',model)
        estimator,feats=ensemble_dict[model]
        
        if model=='simple_nn' :
            estimator.fit(X_train,y_train,epochs=11,batch_size=2048,shuffle=True)
        elif model in {'xgb1','xgb2','rfr','lr'}:
            X_train=X_train[feats]
            X_test=X_test[feats]
            estimator.fit(X_train,y_train)
        else:
            estimator.fit(X_train,y_train)
        test_pred_lst.append(estimator.predict(X_test))
                             
    X_test_level2 = np.column_stack(test_pred_lst)
    print ('X_test_level_2 shape is ',X_test_level2.shape)
        
    gc.collect();
        
    print ('generating train meta features')
    

    y_train_level2= y_train[date_train.isin(range(28,34))]
    X_train_level2 = np.zeros([y_train_level2.shape[0],len(model_list)])
    
    for cur_block_num in range(28,34):
        print (cur_block_num)
        X_train_cv=X_train[date_train<cur_block_num]
        y_train_cv=y_train[date_train<cur_block_num]
        X_pred_cv=X_train[date_train==cur_block_num]
        
        for i,model in enumerate(model_list):
            print ('fitting model',model)
            estimator,feats=ensemble_dict[model]            
            if model=='simple_nn' :
                estimator.fit(X_train_cv,y_train_cv,epochs=11,batch_size=2048,shuffle=True)
                pred=np.squeeze(estimator.predict(X_pred_cv) )
            else:
                estimator.fit(X_train_cv,y_train_cv)
                pred=estimator.predict(X_pred_cv) 
            X_train_level2[date_train_level2==cur_block_num,i]=pred
    print ('X_train_level2 shape is ',X_train_level2.shape)

    ensemble_data_dict={
        'X_train_level2': X_train_level2,
        'y_train_level2': y_train_level2,
        'X_test_level2': X_test_level2
    }

                                 
    return ensemble_data_dict

def gen_embedding_ensemble_data(data_dict,date_train,date_train_level2,ensemble_dict,model_list=[]):
    
    X_train=[np.concatenate(t,axis=0) for t in zip(data_dict['X_train'],data_dict['X_val'])]
    y_train=np.concatenate((data_dict['y_train'],data_dict['y_val']),axis=0)
    X_test=data_dict['X_test']
    
    print ('generating test meta features')
    test_pred_lst=[]
    for model in model_list:
        print ('fitting model',model)
        estimator,feats=ensemble_dict[model]
        estimator.fit(X_train,y_train,epochs=11,batch_size=2048,shuffle=True)
        test_pred_lst.append(estimator.predict(X_test))
                             
    X_test_level2 = np.column_stack(test_pred_lst)
    print ('X_test_level_2 shape is ',X_test_level2.shape)
        
    gc.collect();
        
    print ('generating train meta features')
    

    y_train_level2= y_train[date_train.isin(range(28,34))]
    X_train_level2 = np.zeros([y_train_level2.shape[0],len(model_list)])
    
    for cur_block_num in range(28,34):
        print (cur_block_num)
        X_train_cv=[A[date_train<cur_block_num] for A in X_train]
        y_train_cv=y_train[date_train<cur_block_num]
        X_pred_cv=[A[date_train==cur_block_num] for A in X_train]
        
        for i,model in enumerate(model_list):
            print ('fitting model',model)
            estimator,feats=ensemble_dict[model]            
            estimator.fit(X_train_cv,y_train_cv,epochs=11,batch_size=2048,shuffle=True)
            pred=np.squeeze(estimator.predict(X_pred_cv) )
            X_train_level2[date_train_level2==cur_block_num,i]=pred
    print ('X_train_level2 shape is ',X_train_level2.shape)

    ensemble_data_dict={
        'X_train_level2': X_train_level2,
        'y_train_level2': y_train_level2,
        'X_test_level2': X_test_level2
    }

                                 
    return ensemble_data_dict


def simple_mix(X_train_level2,y_train_level2,X_test_level2):
    print ('simple mix')
    # linear simple mixing 
    alphas_to_try = np.linspace(0, 1, 1001)
    best_alpha=None
    rmse_train_simple_mix=100
    for alpha in alphas_to_try:
        mix=alpha*X_train_level2[:,0]+(1-alpha)*X_train_level2[:,1]
        rmse_mix=RMSE(y_train_level2,mix)
        if rmse_mix<rmse_train_simple_mix:
            best_alpha= alpha 
            rmse_train_simple_mix=rmse_mix
    print ('best rmse score is', rmse_train_simple_mix)
    print ('best alpha is',best_alpha)
    return best_alpha*X_test_level2[:,0]+(1-best_alpha)*X_test_level2[:,1]

def stacking(X_train_level2,y_train_level2,X_test_level2):
    print ('stacking') 
    lr=LinearRegression(n_jobs=-1)
    lr.fit(X_train_level2,y_train_level2)
    return lr.predict(X_test_level2)
    
    
    
    
if __name__ == "__main__":
    ensemble_dict= ensemble_dict
    
    test_file_path='../1c_data/test_data_dict.pkl'
  
    with open('../1c_data/data_dict.pkl','rb') as handle:
        data_dict_non_lr = cPickle.load(handle)
    with open('../1c_data/simple_nn_data_dict.pkl','rb') as handle:
        data_dict_lr = cPickle.load(handle)
    with open('../1c_data/embedding_nn_data_dict.pkl','rb') as handle:
        data_dict_embedding = cPickle.load(handle)
  

    date_train=pd.concat([data_dict_non_lr['X_train'],data_dict_non_lr['X_val']]).date_block_num
    date_train_level2=date_train.loc[date_train.isin(range(28,34))]

    ensemble_data_dict_non_lr=gen_ensemble_data(
        data_dict_non_lr,date_train,date_train_level2,ensemble_dict, model_list=['xgb1','rfr','lr'])
    with open('../1c_data/data_dict_ensemble_non_lr', 'wb') as handle:
        cPickle.dump(ensemble_data_dict_non_lr,handle, protocol=-1)
                             
    ensemble_data_dict_lr=gen_ensemble_data(
        data_dict_lr,date_train,date_train_level2,ensemble_dict, model_list=['simple_nn'])
    with open('../1c_data/data_dict_ensemble_lr', 'wb') as handle:
        cPickle.dump(ensemble_data_dict_lr,handle, protocol=-1)

    ensemble_data_dict_embedding=gen_embedding_ensemble_data(
        data_dict_embedding,date_train,date_train_level2,ensemble_dict, model_list=['embedding_nn'])         
    with open('../1c_data/data_dict_ensemble_embedding', 'wb') as handle:
        cPickle.dump(ensemble_data_dict_embedding,handle, protocol=-1)
                             
    assert np.array_equal(ensemble_data_dict_non_lr['y_train_level2'],ensemble_data_dict_embedding['y_train_level2'])
    assert np.array_equal(ensemble_data_dict_embedding['y_train_level2'],ensemble_data_dict_lr['y_train_level2'])
    X_train_level2=np.concatenate((ensemble_data_dict_non_lr['X_train_level2'],
                                ensemble_data_dict_lr['X_train_level2'],ensemble_data_dict_embedding['X_train_level2']),axis=1)
    X_test_level2=np.concatenate((ensemble_data_dict_non_lr['X_test_level2'],
                                ensemble_data_dict_lr['X_test_level2'],ensemble_data_dict_embedding['X_test_level2']),axis=1)
    y_train_level2=ensemble_data_dict_non_lr['y_train_level2']
    X_train_level2, X_test_level2=gen_pairwise_diff( X_train_level2, X_test_level2)
                             
  
    print ('X_train_level2 shape ',X_train_level2.shape)
    print ('X_test_level2 shape',X_test_level2.shape)
    print ('y_train_level2 shape',y_train_level2.shape)

    pred_stacking=stacking(X_train_level2,y_train_level2,X_test_level2)
    print ('stacking prediction shape',pred_stacking.shape)
    data_dict_non_lr['X_test']['item_cnt_month']=pred_stacking
    test=pd.read_csv('test.csv')
    submission=pd.merge(test, data_dict_non_lr['X_test'], on=['shop_id','item_id'],how='left')[['ID','item_cnt_month']]
    submission.to_csv('submission_stacking_%s.csv'%time.strftime("%Y%m%d-%H%M%S"), index=False)
                            
