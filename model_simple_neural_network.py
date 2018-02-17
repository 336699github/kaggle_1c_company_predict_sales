
import numpy as np
import pandas as pd
import _pickle as cPickle
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense,Merge
from keras.layers.core import Dense, Dropout,Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import features
from keras import backend as K
import time


def _build_keras_model(n_feats):
    
    print ('Creating simple nn model')
    model = Sequential()
    model.add(Dense(n_feats, input_dim=n_feats, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.02))
    model.add(Dense(1000, kernel_initializer='uniform',activation='relu'))
    model.add(Dense(500, kernel_initializer='uniform',activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    return model
def preprocess_nn_features_for_simple(data_dict, nn_feats):
    print ('standardize all features')
    nn_data_dict=dict()
    for key in data_dict.keys():
        nn_data_dict[key]=data_dict[key][nn_feats] if 'X' in key else data_dict[key]
    scaler=StandardScaler(copy=True,with_mean=True,with_std=True) 
    nn_data_dict['X_train'][nn_feats]=scaler.fit_transform(nn_data_dict['X_train'][nn_feats])
    nn_data_dict['X_val'][nn_feats]=scaler.transform(nn_data_dict['X_val'][nn_feats])
    nn_data_dict['X_test'][nn_feats]=scaler.transform(nn_data_dict['X_test'][nn_feats])
    
    for key in nn_data_dict.keys():
        nn_data_dict[key]=nn_data_dict[key].as_matrix()
                  
    with open('../1c_data/simple_nn_data_dict.pkl','wb')as handle:
        cPickle.dump(nn_data_dict,handle,protocol=-1)
                  
    with open('../1c_data/simple_nn_X_test.pkl','wb') as handle:
        cPickle.dump(nn_data_dict['X_test'],handle,protocol=-1)
                  
    return nn_data_dict

def run(data_dict,n_feats):
    model=_build_keras_model(n_feats)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    print (model.summary())
    
    early_stopping=EarlyStopping(monitor='val_loss', patience=10)
    check_point = ModelCheckpoint('../1c_data/simple_weights.best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    fit_params={
        'epochs':100,
        'batch_size':2048,
        'validation_data':(data_dict['X_val'],data_dict['y_val']),
        'callbacks':[early_stopping, check_point] ,
        'shuffle':True
    }

    model.fit(data_dict['X_train'],data_dict['y_train'],**fit_params )
    

def make_prediction(test_file_path):
    with open(test_file_path,'rb') as handle:
        X_test=cPickle.load(handle)
    model=_build_keras_model(X_test.shape[1])
    model.load_weights('../1c_data/simple_weights.best.hdf5')
    model.compile(loss='mean_squared_error', optimizer='adam')
    pred= np.squeeze(model.predict(X_test))
    print ('shape of pred:',pred.shape)
    return pred
    
    
if __name__ == "__main__":
    
    file_path='../1c_data/data_dict.pkl'
    print ('getting data from file',file_path)
    #file_path='../1c_data/test_data_dict.pkl'
    with open(file_path,'rb') as handle:
        data_dict = cPickle.load(handle)    
    
    nn_feats=features.lr_feats
    
    nn_data_dict=preprocess_nn_features_for_simple(data_dict, nn_feats)
    
    print ('using %d features'%len(nn_feats))
   
    run(nn_data_dict,len(nn_feats))
    test_file_path='../1c_data/simple_nn_X_test.pkl'
    pred=make_prediction(test_file_path)
   
    
    data_dict['X_test']['item_cnt_month']=pred
    
    test=pd.read_csv('test.csv')
    submission=pd.merge(test,data_dict['X_test'], 
    on=['shop_id','item_id'],how='left')[['ID','item_cnt_month']]
    submission.to_csv('submission_nn_simple_%s.csv'%time.strftime("%Y%m%d-%H%M%S"), index=False)
    print ('saved submission to local')
