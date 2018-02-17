from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from downcast_dtypes import downcast_dtypes
import gc


def go(data,feature_names,dim=10):

	vect = TfidfVectorizer()
	df_text=data[['item_id','shop_id','date_block_num']]
	
	if 'item_category_name' in feature_names:
		print ('fitting item_category_name features ')
		categories=pd.read_csv('item_categories.csv')
		vect.fit(categories.item_category_name)
		print ('transforming item_category_name features')
		category_name_dtm= vect.transform(data.item_category_name)
		#category_name_feats=pd.DataFrame(category_name_dtm.toarray(), columns=vect.get_feature_names()) 
		print ('category_name_feats shape,', category_name_dtm.shape)
		svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
		print ('reducing dimensions for item_category_name features')
		category_name_feats=svd.fit_transform(category_name_dtm)
		category_name_feats=pd.DataFrame(category_name_feats, columns=['category_name_'+str(i) for i in range(dim)])
		df_text=pd.concat([df_text,category_name_feats],axis=1)

		del categories,category_name_dtm,category_name_feats

	if 'item_name' in feature_names:
		items=pd.read_csv('items.csv')
		print ('fitting item_name features ')
		vect.fit(items.item_name)
		print ('transforming item_name features')
		item_name_dtm = vect.transform(data.item_name) # create DTM
		#item_name_feats=pd.DataFrame(item_name_dtm.toarray(), columns=vect.get_feature_names()) 
		print ('item_name_feats shape, ', item_name_dtm.shape)
		svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
		print ('reducing dimensions for item_name features')
		item_name_feats=svd.fit_transform(item_name_dtm)
		item_name_feats=pd.DataFrame(item_name_feats, columns=['item_name_'+str(i) for i in range(dim)])
		df_text=pd.concat([df_text,item_name_feats],axis=1)
		del vect, svd, items,item_name_dtm,item_name_feats 
	
	df_text=downcast_dtypes(df_text)
	gc.collect();

	print ('df_text shape is: ', df_text.shape)

	return df_text