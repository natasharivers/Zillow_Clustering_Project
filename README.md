# Zillow Clustering Project

_____________________________________________________________________________
___________________________________________________________________________________

## Goals:

- Find what is driving the errors in the Zillow Zestimate
    - (logerror= predicted sale value - actual sale value)

___________________________________________________________________________________
___________________________________________________________________________________


## Planning:

my TRELLO board is available at link: https://trello.com/b/UaxUz5KU/zillow-clustering-project

___________________________________________________________________________________
___________________________________________________________________________________


## Data Dictionary:

| Target          |       Datatype          |    Definition                                       |
|-----------------|-------------------------|:---------------------------------------------------:|
| logerror        | 48646 non-null: float64 | difference between actual value and predicted value | 


| Feature                    |       Datatype         |    Definition                       |
|----------------------------|------------------------|:-----------------------------------:|
|parcelid                    |48646 non-null: int64   |unique property id                   |
|bedrooms                    |48646 non-null: int64   |number of bedrooms                   |
|bathrooms                   |48646 non-null: float64 |number of bathrooms                  |
|total_sqft                  |48646 non-null: int64   |total calculated square feet         |
|county_code                 |48646 non-null: int64   |county code                          |
|latidude                    |48646 non-null: int64   |latitude of home location            |
|longitude                   |48646 non-null: int64   |longitude of home location           |
|lotsizesquarefeet           |48646 non-null: int64   |total calculated square feet         |
|regionidzip                 |48646 non-null: int64   |zip code of property                 |
|year_built                  |48646 non-null: int64   |year the property was built          |
|structuretaxvaluedollarcnt  |48646 non-null: int64   |value of structure                   |
|value_assessed              |48646 non-null: int64   |value of entire property             |
|landtaxvaluedollarcnt       |48646 non-null: int64   |value of land on which property sits |
|tax_amount                  |48646 non-null: int64   |tax amount                           |
|transactiondate             |48646 non-null: object  |date property was purchased          |
|county                      |48646 non-null: object  |engineered column- county name       |
|property_age                |48646 non-null: object  |engineered column- age of property   |



| propertyusetypeid   |    Definition                            |
|---------------------|:----------------------------------------:|
|261                  |Single Family Residential                 |
|263                  |Mobile Home                               |
|264                  |Townhouse                                 |
|265                  |Cluster Home                              |
|273                  |Bungalow                                  |
|275                  |Manufactured, Modular, Prefabricated Home |       
|279                  |Inferred Single Family Residential        |


___________________________________________________________________________________
___________________________________________________________________________________

## Questions & Hypothesis:

### Key Questions:
- Is there a correlation between logerror and total square feet
- Is there a correlation between logerror and the age of the property(in years) 
- Is there a correlation between logerror and longitude of property
- Is there a relationship between logerror and bedroom count

### Hypothesis 1: Correlation Test (Sqft vs Logerror)
- Null Hypothesis: There is no correlation between logerror and total square feet of the property
- Alterate Hypothesis: There is a correlation between logerror and total square feet of the property

### Hypothesis 2: Correlation Test (Age of Property vs Logerror)
- Null Hypothesis: There is no correlation between logerror and property_age
- Alterate Hypothesis: There is a correlation between logerror and property_age

### Hypothesis 3: Correlation Test (Longitude vs Logerror)
- Null Hypothesis: There is no correlation between logerror and longitude
- Alterate Hypothesis: There is a correlation between logerror and longitude

### Hypothesis 4: T-Test (Bedrooms vs Logerror)
- Null Hypothesis: There is no relationship between logerror and bedroom count
- Alterate Hypothesis: There is a relationship between logerror and bedroom count
___________________________________________________________________________________
___________________________________________________________________________________


## Executive Summary- Conclusions: 

### My findings are:

- All null hypothesis have been rejected

<br>

- 2nd degree Polynomial regression model performed best

| Model                            | rmse_train  | rmse_validate |
|----------------------------------|:-----------:|---------------|
|mean_baseline                     | 0.16529     | 0.17102       |
|OLS                               | 0.16520     | 0.17103       |
|LassoLars (alpha 2)               | 0.16529     | 0.17102       |
|Polynomial Regression (degree=2)  | 0.16512     | 0.17101       |

<br>

- Clusters have been found between age of property and total sqft of the property

___________________________________________________________________________________
___________________________________________________________________________________

## Pipeline Stages Breakdown

### --> Plan

- Use Trello Board to organize thoughts
    - Link is available here: https://trello.com/b/UaxUz5KU/zillow-clustering-project

<br>

### --> Acquire

- Store functions that are needed to acquire data from the Zillow database on the Codeup data science database server; make sure the acquire.py module contains the necessary imports to run my code.

- The final function will return a pandas DataFrame.

- Import the acquire function from the acquire.py module and use it to acquire the data in the Final Report Notebook.

- cache file

<br>

### --> Prepare 

- Store functions needed to prepare the Zillow data; make sure the module contains the necessary imports to run the code. 

- The final function should do the following: 
    - change column names to make them more legible
    - remove any duplicates
    - removes outliers
    - split the data using "zillow_split" function
    - scale numeric data using "min_max_scaler" function
    
- Import the prepare function from the prepare.py module and use it to prepare the data in the Final Report Notebook.

<br>

### --> Explore

- unscaled data 

- Interaction between independent variables and the target variable is explored using visualization and statistical testing

- Clustering is used to explore the data. A conclusion, supported by statistical testing and visualization, is drawn on whether or not the clusters are helpful/useful. 

- At least 3 combinations of features for clustering


<br>

### --> Model 

- on scaled data

- At least 4 different models are created and their performance is compared. One model is the distinct combination of algorithm, hyperparameters, and features.

- Supervised model- Regression with different features, hyperparameters, different algorithms (use KMeans in clustering)

<br>

### --> Deliver

- Presentation of Final Notebook
    - Audience: Zillow Data Science Team (use technical language)

- Github repository holding:
    - Final Notebook walkthrough to the Zillow Data Science team
    - README that explains the project, how to reproduce the project and notes
    - Custom acquire, prepare, wrangle, explore, etc files


___________________________________________________________________________________
___________________________________________________________________________________

## Reproduce My Project

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook.

- Read this README.md
- Download the aquire.py, prepare.py, explore.py and final_report.ipynb files into your working directory
- Add your own env file to your directory. (user, password, host)
- Run the final_report.ipynb notebook




** Docstrings in functions!!!
*** Document judgement calls 
- use clusters for exploration
- encode clusters are driver (if you want)
- create a model based on clusters (if you want)
