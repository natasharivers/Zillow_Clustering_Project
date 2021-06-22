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

| Target          |       Datatype          |    Definition                      |
|-----------------|-------------------------|:----------------------------------:|
| logerror        | 28391 non-null: float64 | assessed value of homes in dataset | 


| Feature                 |       Datatype         |    Definition               |
|-------------------------|------------------------|:---------------------------:|
|total_sqft               |28321 non-null: float64 |total calculated square feet |
|bedrooms                 |28392 non-null: float64 |number of bedrooms           |
|bathrooms                |28392 non-null: float64 |number of bathrooms          |
|tax_amount               |28391 non-null: float64 |tax amount                   |
|year_built               |28298 non-null: float64 |year the property was built  |
|county_code              |28392 non-null: float64 |county code                  |
|parcelid                 |28392 non-null: int64   |unique property id           |
|tax_rate                 |28390 non-null: float64 |created column for taxrate   |

___________________________________________________________________________________
___________________________________________________________________________________

## Questions & Hypothesis:

### Key Questions:


___________________________________________________________________________________
___________________________________________________________________________________


## Executive Summary- Conclusions: 

### My findings are:



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

- 5 minute presentation of Final Notebook
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
