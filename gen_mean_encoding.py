# method: expanding mean
# use all the data before current date to encode 


import numpy as np
import pandas as pd 
import time
from downcast_dtypes import downcast_dtypes
import gc 
def go(data,feats_to_encode,target_list ):

	# process target column 
	df_mean_enc=data[['shop_id','item_id','date_block_num']]

	for feat in feats_to_encode:
		print ('calculating encoding for ',feat)
		for target in target_list:
			cumsum=data.groupby(feat)[target].cumsum()-data[target]
			cumcnt=data.groupby(feat).cumcount()
			df_mean_enc[target+'_mean_enc_'+feat]=cumsum/(cumcnt+1)



	df_mean_enc=downcast_dtypes(df_mean_enc)
	gc.collect();
		
	print ('df_mean_enc shape is: ', df_mean_enc.shape)


	return df_mean_enc