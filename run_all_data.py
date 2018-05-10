import extract_clean_data
import gen_mean_encoding
import gen_lag_features
import gen_text_features
import _pickle as cPickle
from downcast_dtypes import downcast_dtypes 
import gc
import pandas as pd
import data_split

df_this_month,cols_to_drop=extract_clean_data.go()


feats_to_enc=['shop_id','item_id','item_category_id','month','year']
target_list=['target','shop_target','item_target','category_target']
df_mean_enc=gen_mean_encoding.go(df_this_month,feats_to_enc,target_list)

lag_feats=['target',
       'shop_item_item_price_median', 'shop_item_sales_sum', 'shop_target',
       'shop_item_price_median', 'shop_sales_sum', 'item_target',
       'item_item_price_median', 'item_sales_sum', 'category_target',
       'category_item_price_median', 'category_sales_sum']
print ('calculating lag for ',lag_feats)
df_lag=gen_lag_features.go(df_this_month,lag_feats)

df_text=gen_text_features.go(df_this_month,['item_name','item_category_name'],dim=10)

for df in [df_this_month,df_lag,df_mean_enc,df_text]:
    print (df.shape)
    
key_feats=['item_id','shop_id','date_block_num']

df_full=df_this_month.copy()
for df in [df_lag,df_text,df_mean_enc]:
    df_full=pd.merge(df_full,df, on=key_feats,how='left').fillna(0)
df_full=downcast_dtypes(df_full)

del df_lag, df_mean_enc, df_text, df_this_month
gc.collect();

print ('dropping cols from df_full')
df_full.drop(cols_to_drop,axis=1, inplace=True)
print (df_full.columns)

data_split.go(df_full,'target')
del df_full
gc.collect();
