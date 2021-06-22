#################### ACQUIRE ZILLOW CLUSTERING PROJECT####################


#import libraries
import pandas as pd
import numpy as np
import os
from pydataset import data

# acquire
from env import host, user, password

######################### URL Connection Function ###########################

def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    from env import host, user, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

############################ Acquire Zillow Function ##############################

def get_zillow():
    '''
    This function reads in the Zillow data from the Codeup db
    with all tables joined
    returns: a pandas DataFrame 
    '''
    
    zc_query = '''
    select prop.parcelid
        , pred.logerror
        , bathroomcnt
        , bedroomcnt
        , calculatedfinishedsquarefeet
        , fips
        , latitude
        , longitude
        , lotsizesquarefeet
        , regionidcity
        , regionidcounty
        , regionidzip
        , yearbuilt
        , structuretaxvaluedollarcnt
        , taxvaluedollarcnt
        , landtaxvaluedollarcnt
        , taxamount
        , transactiondate
    from properties_2017 prop
    inner join predictions_2017 pred on prop.parcelid = pred.parcelid
    where propertylandusetypeid = 261;
    '''

    return pd.read_sql(zc_query, get_connection('zillow'))

############################ Zillow CSV Function ##############################


def get_zillow_file():
    '''
    This function reads in the zillow csv if it is available
    if not, one is created and read in as a pandas dataframe
    '''
    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
    
    else:
        df = get_zillow()
        df.to_csv('zillow.csv')
    
    return df
