import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans


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
    '''
    #print head
    print('==================================================================================================')
    print('Dataframe head: ')
    print(df.head(3))
    
    #print info
    print('==================================================================================================')
    print('Dataframe info: ')
    print(df.info())
    
    #print descriptive stats
    print('==================================================================================================')
    print('DataFrame Description')
    print(df.describe())
    num_cols = df.select_dtypes(exclude='O').columns.to_list()
    cat_cols = df.select_dtypes(include='O').columns.to_list()
    
    #print value counts
    print('==================================================================================================')
    print('Dataframe value counts: ')
    for col in df. columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort = False))


 ############################ PRINT NULLS ############################
def nulls_output(df):

    #print nulls by row
    print('=================================================')
    print('nulls in dataframe by row: ')
    print(df.isnull().sum())
    print('=================================================')
    
    #print nulls by column
    print('=================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=================================================')

    #print nulls by row
    print('=================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('=================================================')


############################## CREATE CLUSTERS ##############################

def create_cluster(df, X, k):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    
    scaler = StandardScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, random_state = 123)
    kmeans.fit(X_scaled)
    kmeans.predict(X_scaled)
    df['cluster'] = kmeans.predict(X_scaled)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return df, X_scaled, scaler, kmeans, centroids

############################## SCATTERPLOT ##############################

def create_scatter_plot(x,y,df,kmeans, X_scaled, scaler):
    
    ''' 
    Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot
    '''
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = x, y = y, data = df, hue = 'cluster')
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')

############################## PLOT VARIABLES ##############################

def plot_variable_pairs(df, cont_vars = 2):
    combos = itertools.combinations(df,cont_vars)
    for i in combos:
        plt.figure(figsize=(8,3))
        sns.regplot(data=df, x=i[0], y =i[1],line_kws={"color":"black"})
        plt.show()

############################## GROUP SCATTERPLOT  ##############################

def scatter_plots(X_scaled, col_name= 'column_one', col_name_two= 'column_two'):
    '''
    This function takes in two columns and 
    creates a range of scatter plots based on varying k values
    '''
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    for ax, k in zip(axs.ravel(), range(2, 6)):
        clusters = KMeans(k).fit(X_scaled).predict(X_scaled)
        ax.scatter(X_scaled[col_name], X_scaled[col_name_two], c=clusters)
        ax.set(title='k = {}'.format(k), xlabel=col_name, ylabel=col_name_two)

############################## CREATE SCATTERPLOT  ##############################

def create_scatter_plot(x, y, df, kmeans, X_scaled, scaler, hue_column = 'hue_column'):
    '''
    This function takes in x and y as strings, 
    and returns creates a plot
    '''
    plt.figure(figsize=(14, 9))
    sns.scatterplot(x = x, y = y, data = df)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')

############################## CREATE CLUSTERS  ##############################

def create_cluster(df, X, k, col_name = None):
    
    ''' 
    This function takes in df, X (dataframe with variables you want to cluster on) and k
    It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    the scaler and kmeans object and scaled centroids as a dataframe
    '''
    scaler = StandardScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, random_state = 123)
    kmeans.fit(X_scaled)
    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns = list(X))
    
    if col_name == None:
        #clusters on dataframe 
        df[f'clusters_{k}'] = kmeans.predict(X_scaled)
    else:
        df[col_name] = kmeans.predict(X_scaled)
    
    
    return df, X_scaled, scaler, kmeans, centroids_scaled
