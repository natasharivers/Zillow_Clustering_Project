#################### PREPARE- ZILLOW CLUSTERING PROJECT ####################
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars

############################## REMOVE OUTLIERS  ##############################

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[f'{col}'].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]
        
    return df

############################## NULL VALUES  ##############################

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .5):
    ''' 
    take in a dataframe and a proportion for columns and rows
    return dataframe with columns and rows not meeting proportions dropped
    '''
    # calc column threshold
    col_thresh = int(round(prop_required_column*df.shape[0],0)) 
    # drop columns with non-nulls less than threshold
    df.dropna(axis=1, thresh=col_thresh, inplace=True) 
    # calc row threshhold
    row_thresh = int(round(prop_required_row*df.shape[1],0))  
    # drop columns with non-nulls less than threshold
    df.dropna(axis=0, thresh=row_thresh, inplace=True) 
    
    return df   

############################## IMPUTE MISSING VALUES  ##############################

def impute(df, my_strategy, column_list):
    ''' 
    take in a df strategy and cloumn list
    return df with listed columns imputed using input stratagy
    '''
    #create imputer   
    imputer = SimpleImputer(strategy=my_strategy)
    #fit/transform selected columns
    df[column_list] = imputer.fit_transform(df[column_list])

    return df


############################## PREP ZILLOW  ##############################

def final_prep_zillow(df):
    '''
    This function takes in the zillow df acquired by get_zillow_file,
    then the function removed outliers from bedrooms, bathrooms, value_assessed, and total_sqft
    Returns a cleaned zillow df.
    '''

    #replace blank spaces and special characters
    df = df.replace(r'^\s*$', np.nan, regex=True)

    #filling null values
    #drop using threshold
    df = handle_missing_values(df, prop_required_column = .5, prop_required_row = .5)
    #impute continuous columns using mean
    df = impute(df, 'mean', ['calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount'])
    # imputing descrete columns with most frequent value
    df = impute(df, 'most_frequent', ['bathroomcnt', 'bedroomcnt', 'regionidcity', 'regionidcounty', 'regionidzip', 'yearbuilt', 'fips', 'latitude', 'longitude'])

    #drop duplicate parcelid keeping the latest one by transaction date
    df = df.sort_values('transactiondate').drop_duplicates('parcelid',keep='last')

    #remove outlier
    df = remove_outliers(df, 1.5, ['calculatedfinishedsquarefeet', 'bedroomcnt', 'bathroomcnt'])

    #new column with county names for fips
    df['county'] = df.fips.apply(lambda x: 'orange' if x == 6059.0 else 'los_angeles' if x == 6037.0 else 'ventura')

    #change datatypes
    df.bedroomcnt = df.bedroomcnt.astype(int)
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.astype(int)
    df.fips = df.fips.astype(int)
    df.latitude = df.latitude.astype(int)
    df.longitude = df.longitude.astype(int)
    df.lotsizesquarefeet = df.lotsizesquarefeet.astype(int)
    df.regionidcity = df.regionidcity.astype(int)
    df.regionidcounty = df.regionidcounty.astype(int)
    df.regionidzip = df.regionidzip.astype(int)
    df.yearbuilt = df.yearbuilt.astype(int)
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.astype(int)
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.astype(int)
    df.landtaxvaluedollarcnt = df.landtaxvaluedollarcnt.astype(int)
    df.taxamount = df.taxamount.astype(int)

    #change column names to be more legible
    df = df.rename(columns={"calculatedfinishedsquarefeet": "total_sqft", "bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "taxvaluedollarcnt": "value_assessed", "taxamount": "tax_amount", "yearbuilt": "year_built", "fips": "county_code"})

    return df

############################## ZILLOW SPLIT ##############################

def zillow_split(df, target):
    '''
    This function take in get_zillow  from aquire.py and performs a train, validate, test split
    Returns train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test
    and prints out the shape of train, validate, test
    '''
    #create train_validate and test datasets
    train, test = train_test_split(df, train_size = 0.8, random_state = 123)
    #create train and validate datasets
    train, validate = train_test_split(train, train_size = 0.7, random_state = 123)

    #Split into X and y
    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    # Have function print datasets shape
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
   
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test

############################## MIN MAX SCALER ##############################


def min_max_scaler(X_train, X_validate, X_test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).
    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])
    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])
    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )
    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )
    # Overwriting columns in our input dataframes for simplicity
    for i in numeric_cols:
        X_train[i] = X_train_scaled[i]
        X_validate[i] = X_validate_scaled[i]
        X_test[i] = X_test_scaled[i]
    return X_train, X_validate, X_test

############################## RFE Function ##############################
def rfe(X, y, n):
    lm = LinearRegression()
    rfe = RFE(lm, n)
    rfe.fit(X, y)
    
    n_features = X.columns[rfe.support_]
    
    return n_features

############################## SelectKBest Function ##############################


def select_kbest(X,y,k): 
    f_selector = SelectKBest(f_regression, k)
    f_selector.fit(X, y)
    k_features = X.columns[f_selector.get_support()]

    return k_features