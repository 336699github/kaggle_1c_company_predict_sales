# generate data_dict and test_data_dict
import numpy as np
import pickle
import time
import _pickle as cPickle
import pandas as pd
import gc
def go(data, target):

	data_dict=dict()

	print ('spliting dataframe')
	df_train=data[~data.date_block_num.isin([9,21,33,34])]
	df_val=data[data.date_block_num.isin([9,21,33])]
	df_test=data[data.date_block_num==34]
	data_dict={
		'X_train':df_train.drop(target,axis=1),
		'y_train':df_train[target],
		'X_val':df_val.drop(target,axis=1),
		'y_val':df_val[target],
		'X_test':df_test.drop(target,axis=1)
	}

	df_sample_train=df_train.groupby('date_block_num').head(10).reset_index(drop=True)
	df_sample_val=df_val.groupby('date_block_num').head(10).reset_index(drop=True)
	test_data_dict={
	    'X_train': df_sample_train.drop('target',axis=1),
	    'y_train': df_sample_train.target,
	    'X_val': df_sample_val.drop('target',axis=1),
	    'y_val': df_sample_val.target,
	    'X_test': df_test.drop(target,axis=1)[:10]
	}

	print ('saving data_dict and test_data_dict to local')
	with open('../1c_data/data_dict.pkl', 'wb') as handle:  
	    cPickle.dump(data_dict,handle,protocol=-1)

	with open('../1c_data/test_data_dict.pkl', 'wb') as handle:  
	    cPickle.dump(test_data_dict,handle,protocol=-1)

	del df_train,df_val,df_test, df_sample_train,df_sample_val,
	gc.collect();

	return data_dict, test_data_dict