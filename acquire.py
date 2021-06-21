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
    SELECT calculatedfinishedsquarefeet, bedroomcnt, bathroomcnt, taxvaluedollarcnt, taxamount, yearbuilt, fips, properties_2017.parcelid
    FROM properties_2017
    JOIN predictions_2017 ON properties_2017.parcelid = predictions_2017.parcelid
    JOIN propertylandusetype ON properties_2017.propertylandusetypeid= propertylandusetype.propertylandusetypeid
    WHERE properties_2017.propertylandusetypeid IN (260, 261, 262, 263, 264, 265, 268, 273, 275, 276, 279)
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