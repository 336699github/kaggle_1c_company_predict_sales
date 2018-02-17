non_lr_feats=[
    'item_id',
    'date_block_num',
    'shop_id',
    'month',
    'item_category_id',
    'category_name_0',
    'category_name_1',
    'category_name_2',
    'category_name_3',
    'category_name_4',
    'item_name_0',
    'item_name_1',
    'item_name_2',
    'item_name_3',
    'item_name_4',
    
    #month mean encoding
    'category_target_mean_enc_month',
    'item_target_mean_enc_month',
    'shop_target_mean_enc_month',
    'target_mean_enc_month',
    # year mean encoding
    'category_target_mean_enc_year',
    'item_target_mean_enc_year',
    'shop_target_mean_enc_year',
    'target_mean_enc_year',
    # item_id mean encoding
    'category_target_mean_enc_item_id',
    'item_target_mean_enc_item_id',
    'shop_target_mean_enc_item_id',
    'target_mean_enc_item_id',
    # shop_id mean encoding
    'category_target_mean_enc_shop_id',
    'item_target_mean_enc_shop_id',
    'shop_target_mean_enc_shop_id',
    'target_mean_enc_shop_id',
    # item_category_id mean encoding
    'category_target_mean_enc_item_category_id',
    'item_target_mean_enc_item_category_id',
    'shop_target_mean_enc_item_category_id',
    'target_mean_enc_item_category_id',
    
    
     # shop item agg lags
    'shop_item_item_price_median_lag1',
    'shop_item_item_price_median_lag12',
    'shop_item_item_price_median_lag2',
    'shop_item_item_price_median_lag3',
    'shop_item_item_price_median_lag6',
  
    'shop_item_sales_sum_lag1',
    'shop_item_sales_sum_lag2',
    'shop_item_sales_sum_lag3',
    'shop_item_sales_sum_lag5',
    
    'target_lag1',
    'target_lag12',
    'target_lag2',
    'target_lag3',
    'target_lag4',
    'target_lag5',
    'target_lag6',
    
    # category agg lags 
    'category_item_price_median_lag1',
    'category_item_price_median_lag12',
    'category_item_price_median_lag2',
    'category_item_price_median_lag3',
    'category_item_price_median_lag6',
    
    'category_sales_sum_lag1',
    'category_sales_sum_lag12',
    'category_sales_sum_lag2',
    'category_sales_sum_lag3',
    'category_sales_sum_lag4',
    'category_sales_sum_lag5',
    
    'category_target_lag1',
    'category_target_lag2',
    'category_target_lag3',
    'category_target_lag4',
    'category_target_lag5',
    'category_target_lag6',
   
    
    # item agg lags
    'item_item_price_median_lag1',
    'item_item_price_median_lag2',
    'item_item_price_median_lag3', 
    'item_sales_sum_lag1',
    'item_sales_sum_lag12',
    'item_sales_sum_lag2',
    'item_sales_sum_lag4',
    'item_target_lag1',
    'item_target_lag12',
    'item_target_lag2',
    'item_target_lag3',
    'item_target_lag4',
    'item_target_lag5',
    'item_target_lag6',
    
    
   
    # shop agg lags
    
    'shop_item_price_median_lag1',
    'shop_item_price_median_lag2',
    'shop_item_price_median_lag3',
    'shop_sales_sum_lag1',
    'shop_sales_sum_lag12',
    'shop_sales_sum_lag2',
    'shop_sales_sum_lag3',
    'shop_sales_sum_lag4',
    
    'shop_target_lag1',
    'shop_target_lag12',
    'shop_target_lag2',
    'shop_target_lag3',
    'shop_target_lag4',
    'shop_target_lag5',
    'shop_target_lag6',
        
]
lr_feats=[f for f in non_lr_feats if f not in {'shop_id', 'item_id', 'date_block_num', 'month', 'item_category_id'}]

embedding_feats=[
    'shop_id',
    'month',
    'item_category_id',
    'category_name_0',
    'category_name_1',
    'category_name_2',
    'category_name_3',
    'category_name_4',
    'item_name_0',
    'item_name_1',
    'item_name_2',
    'item_name_3',
    'item_name_4',
    
    #month mean encoding
    'category_target_mean_enc_month',
    'item_target_mean_enc_month',
    'shop_target_mean_enc_month',
    'target_mean_enc_month',
    # year mean encoding
    'category_target_mean_enc_year',
    'item_target_mean_enc_year',
    'shop_target_mean_enc_year',
    'target_mean_enc_year',
    # item_id mean encoding
    'category_target_mean_enc_item_id',
    'item_target_mean_enc_item_id',
    'shop_target_mean_enc_item_id',
    'target_mean_enc_item_id',
    # shop_id mean encoding
    'category_target_mean_enc_shop_id',
    'item_target_mean_enc_shop_id',
    'shop_target_mean_enc_shop_id',
    'target_mean_enc_shop_id',
    # item_category_id mean encoding
    'category_target_mean_enc_item_category_id',
    'item_target_mean_enc_item_category_id',
    'shop_target_mean_enc_item_category_id',
    'target_mean_enc_item_category_id',
    
    
     # shop item agg lags
    'shop_item_item_price_median_lag1',
    'shop_item_item_price_median_lag12',
    'shop_item_item_price_median_lag2',
    'shop_item_item_price_median_lag3',
    'shop_item_item_price_median_lag6',
  
    'shop_item_sales_sum_lag1',
    'shop_item_sales_sum_lag2',
    'shop_item_sales_sum_lag3',
    'shop_item_sales_sum_lag5',
    
    'target_lag1',
    'target_lag12',
    'target_lag2',
    'target_lag3',
    'target_lag4',
    'target_lag5',
    'target_lag6',
    
    # category agg lags 
    'category_item_price_median_lag1',
    'category_item_price_median_lag12',
    'category_item_price_median_lag2',
    'category_item_price_median_lag3',
    'category_item_price_median_lag6',
    
    'category_sales_sum_lag1',
    'category_sales_sum_lag12',
    'category_sales_sum_lag2',
    'category_sales_sum_lag3',
    'category_sales_sum_lag4',
    'category_sales_sum_lag5',
    
    'category_target_lag1',
    'category_target_lag2',
    'category_target_lag3',
    'category_target_lag4',
    'category_target_lag5',
    'category_target_lag6',
   
    
    # item agg lags
    'item_item_price_median_lag1',
    'item_item_price_median_lag2',
    'item_item_price_median_lag3', 
    'item_sales_sum_lag1',
    'item_sales_sum_lag12',
    'item_sales_sum_lag2',
    'item_sales_sum_lag4',
    'item_target_lag1',
    'item_target_lag12',
    'item_target_lag2',
    'item_target_lag3',
    'item_target_lag4',
    'item_target_lag5',
    'item_target_lag6',
    
    
   
    # shop agg lags
    
    'shop_item_price_median_lag1',
    'shop_item_price_median_lag2',
    'shop_item_price_median_lag3',
    'shop_sales_sum_lag1',
    'shop_sales_sum_lag12',
    'shop_sales_sum_lag2',
    'shop_sales_sum_lag3',
    'shop_sales_sum_lag4',
    
    'shop_target_lag1',
    'shop_target_lag12',
    'shop_target_lag2',
    'shop_target_lag3',
    'shop_target_lag4',
    'shop_target_lag5',
    'shop_target_lag6',
        
]
