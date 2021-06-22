import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

########################### NULLS BY COLUMN ###########################

#get nulls by column
def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'pct_rows_missing': prcnt_miss})
    return cols_missing

########################### NULLS BY ROW ###########################

#get nulls by row
def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'pct_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'pct_cols_missing']).count()\
    .rename(index=str, columns={'index':'num_rows'}).reset_index()
    return rows_missing

########################### SUMMARIZE FUNCTION ###########################
#summarize data in the df
#head, info, describe, value counts, nulls

def summarize(df):
    '''
    this function will take in a single argument (a pandas df) 
    output various statistics on that df, including:
    #.head()
    #.info()
    #.describe()
    #.value_counts()
    #nulls 
    '''
    #print head
    print('=================================================')
    print('Dataframe head: ')
    print(df.head(3))
    
    #print info
    print('=================================================')
    print('Dataframe info: ')
    print(df.info())
    
    #print descriptive stats
    print('=================================================')
    print('DataFrame Description')
    print(df.describe())
    num_cols = df.select_dtypes(exclude='O').columns.to_list()
    cat_cols = df.select_dtypes(include='O').columns.to_list()
    
    #print value counts
    print('=================================================')
    print('Dataframe value counts: ')
    for col in df. columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort = False))
    
    #print nulls by column
    print('=================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    
    #print nulls by row
    print('=================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('=================================================')

########################## REMOVE COLUMNS FUNCTION ###########################

def remove_columns (df, cols_to_remove):
    df = df.drop(columns = cols_to_remove)
    return df

########################### FILLNA FUNCTION ###########################

def handle_missing_values(df, prop_required_columns = 0.5, prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh= threshold)
    return df

########################### DATA PREP FUNCTION ###########################

def data_prep(df, cols_to_remove=[], prop_required_columns =0.5, prop_required_row=0.75):
    df = remove_columns(df, cols_to_remove)
    df= handle_missing_values(df, prop_required_columns, prop_required_row)
    return df

########################### UPPER OUTLIER FUNCTION ###########################

#create outlier function
def get_upper_outliers(s, k= 1.5):
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x-upper_bound, 0]))


#apply the function
def add_upper_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df [col + '_outlier_upper'] = get_upper_outliers(df[col], k)
    return df
    
############################## DROP OUTLIERS FUNCTION ##############################

def outlier_bound_calculation(df, variable):
    '''
    calcualtes the lower and upper bound to locate outliers in variables
    '''
    quartile1, quartile3 = np.percentile(df[variable], [25,75])
    IQR_value = quartile3 - quartile1
    lower_bound = quartile1 - (1.5 * IQR_value)
    upper_bound = quartile3 + (1.5 * IQR_value)
    '''
    returns the lowerbound and upperbound values
    '''
    return print(f'For {variable} the lower bound is {lower_bound} and  upper bound is {upper_bound}')

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

############################## MinMax Scaler ##############################


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
    #
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


############################## CREATE CLUSTERS ##############################

def create_cluster(df, X, k):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    
    scaler = StandardScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    kmeans.predict(X_scaled)
    df['cluster'] = kmeans.predict(X_scaled)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return df, X_scaled, scaler, kmeans, centroids

############################## SCATTERPLOT ##############################

def create_scatter_plot(x,y,df,kmeans, X_scaled, scaler):
    
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = x, y = y, data = df, hue = 'cluster')
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')