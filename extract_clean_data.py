import pandas as pd
import numpy as np
import _pickle as cPickle
from itertools import product
import time
import gc
from downcast_dtypes import downcast_dtypes
import data_transformation 
import warnings
warnings.filterwarnings("ignore")

def go(ROOT_DIR=''):
    # read in dataframe from csv files
    item_cat=pd.read_csv('%sitem_categories.csv'%ROOT_DIR)
    items=pd.read_csv('%sitems.csv'%ROOT_DIR)
    sales=pd.read_csv('%ssales_train.csv'%ROOT_DIR)
    test=pd.read_csv('%stest.csv'%ROOT_DIR)
    shop=pd.read_csv('%sshops.csv'%ROOT_DIR)
    
    #remove outliers from sales
    print ('removing outliers')
    sales=sales[sales['item_price']<100000]
    sales=sales[sales['item_cnt_day']<1500]

    #generate sales column
    sales['sales']=sales['item_cnt_day']*sales['item_price']

    #join items_category table to sales 
    sales=pd.merge(sales,items[['item_id','item_category_id']], on='item_id', how='left')


    #generate full dataframe
    print ('preparing dataframe grid ')
    df_this_month=[]
    index_cols = [ 'shop_id','item_id','date_block_num']
    for block_num in sales.date_block_num.unique():
        cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
        cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
        df_this_month.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
    df_this_month.append(np.array(list(product(*[test.shop_id.unique(),test.item_id.unique(),[34]])),dtype='int32'))
    
    df_this_month = pd.DataFrame(np.vstack(df_this_month), columns = index_cols,dtype=np.int32)

    # add item_category_id columns to dataframe, in order to calculate agg values for item_category 
    df_this_month= pd.merge(df_this_month, items[['item_id','item_category_id']], on='item_id', how='left')
    

    print ('starting computing aggregation stats... ')
    print ('calculating shop-item agg')
    # groupby shop_id,item_id, caculate basic stats feature on item_cnt_day, item_price, and daily sales 
    gb=sales.groupby(['shop_id','item_id','date_block_num'], as_index=False)\
    .agg({
        'item_cnt_day':{'target':'sum'},
        'item_price':{'shop_item_item_price_median':'median'},
        'sales':{'shop_item_sales_sum':'sum'}
        })
    # fix column names 
    gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
    df_this_month=pd.merge(df_this_month,gb, 
        on=['date_block_num','item_id','shop_id'],how='left').fillna(0)
    
    # Same as above but with shop-month aggregates
    print ('calculating shop agg')
    gb = sales.groupby(['shop_id', 'date_block_num'],as_index=False)\
    .agg({
        'item_cnt_day':{'shop_target':'sum'},
        'item_price':{'shop_item_price_median':'median'},
        'sales':{'shop_sales_sum':'sum'}
        })
    gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
    df_this_month = pd.merge(df_this_month, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)
    print ('calculating item agg')
    # Same as above but with item-month aggregates
    gb = sales.groupby(['item_id', 'date_block_num'],as_index=False)\
    .agg({
        'item_cnt_day':{'item_target':'sum'},
        'item_price':{'item_item_price_median':'median'},
        'sales':{'item_sales_sum':'sum'}
        })
    gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
    df_this_month = pd.merge(df_this_month, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

    print ('calculating item_category agg')
    # Same as above but with item_category-month aggregates
    gb = sales.groupby(['item_category_id', 'date_block_num'],as_index=False)\
    .agg({
        'item_cnt_day':{'category_target':'sum'},
        'item_price':{'category_item_price_median':'median'},
        'sales':{'category_sales_sum':'sum'}
        })
    gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
    df_this_month = pd.merge(df_this_month, gb, how='left', on=['item_category_id', 'date_block_num']).fillna(0)


    print ('clipping outliers from target values')
    df_this_month.target=np.clip(df_this_month.target,0,20)
    #df_this_month.shop_target=np.clip(df_this_month.shop_target,0,13000)
    #df_this_month.item_target=np.clip(df_this_month.item_target,0,500)
    #df_this_month.category_target=np.clip(df_this_month.category_target,0,None)


     # join items table and items_category table 
    print ('joining items table and items_category table to grid')
    df_this_month=pd.merge(df_this_month,items[['item_id','item_name']],on='item_id',how='left')
    df_this_month=pd.merge(df_this_month,item_cat,on='item_category_id',how='left')


    print ('month, year features')
    # generate month, year columns 
    df_date=sales.date.str.split('.', expand=True)
    df_date.columns=['day','month','year']
    df_date.month=df_date.month.astype(int)
    df_date.year=df_date.year.astype(int)
    sales=pd.concat([sales,df_date],axis=1)
    df_date= sales[['date_block_num','month','year']].drop_duplicates()
    df_date=df_date.append({'date_block_num':34,'month':11,'year':2015}, ignore_index=True)
    df_this_month=pd.merge(df_this_month,df_date,on=['date_block_num'],how='left')

    # Downcast dtypes from 64 to 32 bit to save memory
    df_this_month = downcast_dtypes(df_this_month)
    del gb , df_date
    gc.collect();
    
    print ('df_this_month shape is: ', df_this_month.shape)
    print ('df_this_month columns: ', df_this_month.columns)

    cols_to_drop=[c for c in df_this_month.columns if c not in 
    ['item_id','shop_id','item_category_id','month','year','target','date_block_num']]
    '''
    with open('../1c_data/df_this_month.pkl','wb') as handle:
        cPickle.dump(df_this_month, handle, protocol=-1)
    
    with open('../1c_data/df_this_month.pkl','rb') as handle:
        df_this_month=cPickle.load(handle)
   
    df_this_month=data_transformation.go(df_this_month)
    df_this_month = downcast_dtypes(df_this_month)
    with open('../1c_data/df_this_month_transformed.pkl','wb') as handle:
        cPickle.dump(df_this_month, handle, protocol=-1)
        '''
    return df_this_month,cols_to_drop
if __name__ == "__main__":

    go()
    