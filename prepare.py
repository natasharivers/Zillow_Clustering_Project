#################### PREPARE- ZILLOW CLUSTERING PROJECT ####################
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

############################## NULL VALUES  ##############################


def handle_missing_values(df, prop_required_columns = 0.5, prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh= threshold)
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

    #drop null values
    df = handle_missing_values(df, prop_required_columns = 0.5, prop_required_row=0.75)
    #drop remaining null values
    df = df.dropna()

    # create dummy columns for species
    county_dummies = pd.get_dummies(df.fips)
    # add dummy columns to df
    df = pd.concat([df, county_dummies], axis=1)

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
    #df['6037.0'] = df['6037.0'].astype(int)
    #df['6059.0'] = df['6059.0'].astype(int)
    #df['6111.0'] = df['6111.0'].astype(int)

    #change column names to be more legible
    df = df.rename(columns={"calculatedfinishedsquarefeet": "total_sqft", "bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "taxvaluedollarcnt": "value_assessed", "taxamount": "tax_amount", "yearbuilt": "year_built", "fips": "county_code", "6037.0": "Los Angeles", "6059.0": "Orange ", "6037.0": "Ventura" })

    #remove outliers for bedrooms
    q1_bed = df['bedrooms'].quantile(0.25)
    q3_bed = df['bedrooms'].quantile(0.75)
    iqr_bed = q3_bed - q1_bed
    lowerbound_bed = q1_bed - (1.5 * iqr_bed)
    upperbound_bed = q3_bed + (1.5 * iqr_bed)
    df= df[df.bedrooms > lowerbound_bed]
    df= df[df.bedrooms < upperbound_bed]

    #remove outliers for bathrooms
    q1_bath = df['bathrooms'].quantile(0.25)
    q3_bath = df['bathrooms'].quantile(0.75)
    iqr_bath = q3_bath - q1_bath
    lowerbound_bath = q1_bath - (1.5 * iqr_bath)
    upperbound_bath = q3_bath + (1.5 * iqr_bath)
    df= df[df.bathrooms > lowerbound_bath]
    df= df[df.bathrooms < upperbound_bath]

    #remove outliers for value assessed
    q1_tax = df['value_assessed'].quantile(0.25)
    q3_tax = df['value_assessed'].quantile(0.75)
    iqr_tax = q3_tax- q1_tax
    lowerbound_tax = q1_tax - (1.5 * iqr_tax)
    upperbound_tax = q3_tax + (1.5 * iqr_tax)
    df= df[df.value_assessed > lowerbound_tax]
    df= df[df.value_assessed < upperbound_tax]

    #remove outliers for total sqft
    q1_sqft = df['total_sqft'].quantile(0.25)
    q3_sqft = df['total_sqft'].quantile(0.75)
    iqr_sqft = q3_sqft - q1_sqft
    lowerbound_sqft = q1_sqft - (1.5 * iqr_sqft)
    upperbound_sqft = q3_sqft + (1.5 * iqr_sqft)
    df= df[df.total_sqft > lowerbound_sqft]
    df= df[df.total_sqft < upperbound_sqft]

    #drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

