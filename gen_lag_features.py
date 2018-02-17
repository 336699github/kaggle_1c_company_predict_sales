import pandas as pd 
import time
from downcast_dtypes import downcast_dtypes
import gc 
def go(data, lag_feats, lag_list=[1,2,3,4,5,6,12]):

	key_feats=['item_id','shop_id']
	df_all_lag_features=data[key_feats+['date_block_num']]

	for i in lag_list:
		print ('computing lag ', i)
		df_lagged=data[key_feats+lag_feats]
		df_lagged['date_block_num']=data['date_block_num']+i
		df_lagged.columns=[f+'_lag'+str(i) if f in lag_feats else f for f in df_lagged.columns]
		df_all_lag_features=pd.merge(df_all_lag_features,df_lagged,
					  on=key_feats+['date_block_num'],
					  how='left').fillna(0)
   
	df_all_lag_features=downcast_dtypes(df_all_lag_features)
	del df_lagged
	gc.collect();


	print ('df_all_lag_features shape is: ', df_all_lag_features.shape)
	
	

	return df_all_lag_features