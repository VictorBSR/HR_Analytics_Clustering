# HR Analytics and Clustering<a class="anchor" id="top-bullet"></a>

<img src="images/Overview1.png" alt="Overview1" />

----

Victor Barros e S. dos Reis
Aug, 2024

### Discovery -> Insights -> Actions -> Outcomes

## Business Problem
- [Kaggle Dataset](https://www.kaggle.com/datasets/anshika2301/hr-analytics-dataset)
- A HR department from a big company has hired us to help them understand their employee profiles from a dataset of the employee board with data from all throughout 2023 and determine if there are issues negatively affecting the turnover rates. They would like to know if there are certain groups of employees that may have higher propensity to attrition, understand their common attributes, and finally come up with solutions that can be implemented in order to reduce turnover rates.


## Goals
- Our main goals are:
    - To determine if the Attrition/Turnover rates are indeed abnormal and if there are patterns for it
    - Identify possible causes that explain abnormal Attrition/Turnover rates by raising hypothesis
    - Accept/refute the hypothesis by analysing data correlation and the business rules
    - Classify the employees in different groups/clusters in order to suggest customized initiatives for each of them
- We are analysing the *'HR_Analytics.csv'* dataset
- And by using data analysis techniques such as correlation, regression and different plots
- We intend to effectively categorize our employees in groups with similar characteristics (**features**) and find out which of them are possibly more related with high Attrition/Turnover rates (our **target**)
- Data will be presented in a PowerBI report, together with the findings and recommendations


## Steps
- Load Data
- Describe the Data
- Pre-Processing: Data Cleaning, EDA and Feature Engineering
- Train-Test Split
- ML Model Building
- Model Performance
- Model Deployment
- Conclusion


## Table of Contents:
* [Business Problem](#first-bullet)
* [Data Loading](#second-bullet)
* [Data Cleaning](#third-bullet)
* [EDA](#fourth-bullet)
* [Feature Engineering](#fifth-bullet)
* [Choosing ML models](#sixth-bullet)
* [Results Analysis](#ninth-bullet)

---

# Business Problem<a class="anchor" id="first-bullet"></a>
- The HR department is almost sure that their attrition rates can be explained by a few reasons that some groups of employees may have in common, but they would like a specialized data consultant to perform this analysis and help them pinpoint a few initiatives to improve their overall scenario and employee well-being

## Understanding the business rules
- Employee Well-being is a multi-dimentional aspect that is affected by several factors/pilars, including: career satisfaction, healthy relationships, emotional stability, financial security, physical health, only to name a few.
- Some factors such as a high overtime working schedule, advanced age and receival of attractive job offers, for example, may explain a tendency to leave the company.

----

# Data Loading<a class="anchor" id="second-bullet"></a>
- All needed data will be loaded from the 'HR_Analytics.csv' dataset, from Kaggle
- A Mindmap for the features will be made in order to better understand what may be related to the Attrition/Turnover

### Imports


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
```


```python
# configurations and lib settings
pd.options.display.max_columns #20
pd.set_option('display.max_columns', None)
pio.renderers.default = 'notebook'
```

### Load Data


```python
filepath=r''
```


```python
df = pd.read_csv(filepath+'\HR_Analytics.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpID</th>
      <th>Age</th>
      <th>AgeGroup</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobRole</th>
      <th>JobSatisfaction</th>
      <th>MaritalStatus</th>
      <th>MonthlyIncome</th>
      <th>SalarySlab</th>
      <th>MonthlyRate</th>
      <th>NumCompaniesWorked</th>
      <th>Over18</th>
      <th>OverTime</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RM297</td>
      <td>18</td>
      <td>18-25</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>230</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>405</td>
      <td>3</td>
      <td>Male</td>
      <td>54</td>
      <td>3</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>3</td>
      <td>Single</td>
      <td>1420</td>
      <td>Upto 5k</td>
      <td>25233</td>
      <td>1</td>
      <td>Y</td>
      <td>No</td>
      <td>13</td>
      <td>3</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RM302</td>
      <td>18</td>
      <td>18-25</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>812</td>
      <td>Sales</td>
      <td>10</td>
      <td>3</td>
      <td>Medical</td>
      <td>1</td>
      <td>411</td>
      <td>4</td>
      <td>Female</td>
      <td>69</td>
      <td>2</td>
      <td>1</td>
      <td>Sales Representative</td>
      <td>3</td>
      <td>Single</td>
      <td>1200</td>
      <td>Upto 5k</td>
      <td>9724</td>
      <td>1</td>
      <td>Y</td>
      <td>No</td>
      <td>12</td>
      <td>3</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RM458</td>
      <td>18</td>
      <td>18-25</td>
      <td>Yes</td>
      <td>Travel_Frequently</td>
      <td>1306</td>
      <td>Sales</td>
      <td>5</td>
      <td>3</td>
      <td>Marketing</td>
      <td>1</td>
      <td>614</td>
      <td>2</td>
      <td>Male</td>
      <td>69</td>
      <td>3</td>
      <td>1</td>
      <td>Sales Representative</td>
      <td>2</td>
      <td>Single</td>
      <td>1878</td>
      <td>Upto 5k</td>
      <td>8059</td>
      <td>1</td>
      <td>Y</td>
      <td>Yes</td>
      <td>14</td>
      <td>3</td>
      <td>4</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RM728</td>
      <td>18</td>
      <td>18-25</td>
      <td>No</td>
      <td>Non-Travel</td>
      <td>287</td>
      <td>Research &amp; Development</td>
      <td>5</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1012</td>
      <td>2</td>
      <td>Male</td>
      <td>73</td>
      <td>3</td>
      <td>1</td>
      <td>Research Scientist</td>
      <td>4</td>
      <td>Single</td>
      <td>1051</td>
      <td>Upto 5k</td>
      <td>13493</td>
      <td>1</td>
      <td>Y</td>
      <td>No</td>
      <td>15</td>
      <td>3</td>
      <td>4</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RM829</td>
      <td>18</td>
      <td>18-25</td>
      <td>Yes</td>
      <td>Non-Travel</td>
      <td>247</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>1156</td>
      <td>3</td>
      <td>Male</td>
      <td>80</td>
      <td>3</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>3</td>
      <td>Single</td>
      <td>1904</td>
      <td>Upto 5k</td>
      <td>13556</td>
      <td>1</td>
      <td>Y</td>
      <td>No</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['EmpID', 'Age', 'AgeGroup', 'Attrition', 'BusinessTravel', 'DailyRate',
           'Department', 'DistanceFromHome', 'Education', 'EducationField',
           'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender',
           'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole',
           'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'SalarySlab',
           'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime',
           'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
           'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
           'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
           'YearsInCurrentRole', 'YearsSinceLastPromotion',
           'YearsWithCurrManager'],
          dtype='object')



### Data Description


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1480 entries, 0 to 1479
    Data columns (total 38 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   EmpID                     1480 non-null   object 
     1   Age                       1480 non-null   int64  
     2   AgeGroup                  1480 non-null   object 
     3   Attrition                 1480 non-null   object 
     4   BusinessTravel            1480 non-null   object 
     5   DailyRate                 1480 non-null   int64  
     6   Department                1480 non-null   object 
     7   DistanceFromHome          1480 non-null   int64  
     8   Education                 1480 non-null   int64  
     9   EducationField            1480 non-null   object 
     10  EmployeeCount             1480 non-null   int64  
     11  EmployeeNumber            1480 non-null   int64  
     12  EnvironmentSatisfaction   1480 non-null   int64  
     13  Gender                    1480 non-null   object 
     14  HourlyRate                1480 non-null   int64  
     15  JobInvolvement            1480 non-null   int64  
     16  JobLevel                  1480 non-null   int64  
     17  JobRole                   1480 non-null   object 
     18  JobSatisfaction           1480 non-null   int64  
     19  MaritalStatus             1480 non-null   object 
     20  MonthlyIncome             1480 non-null   int64  
     21  SalarySlab                1480 non-null   object 
     22  MonthlyRate               1480 non-null   int64  
     23  NumCompaniesWorked        1480 non-null   int64  
     24  Over18                    1480 non-null   object 
     25  OverTime                  1480 non-null   object 
     26  PercentSalaryHike         1480 non-null   int64  
     27  PerformanceRating         1480 non-null   int64  
     28  RelationshipSatisfaction  1480 non-null   int64  
     29  StandardHours             1480 non-null   int64  
     30  StockOptionLevel          1480 non-null   int64  
     31  TotalWorkingYears         1480 non-null   int64  
     32  TrainingTimesLastYear     1480 non-null   int64  
     33  WorkLifeBalance           1480 non-null   int64  
     34  YearsAtCompany            1480 non-null   int64  
     35  YearsInCurrentRole        1480 non-null   int64  
     36  YearsSinceLastPromotion   1480 non-null   int64  
     37  YearsWithCurrManager      1423 non-null   float64
    dtypes: float64(1), int64(25), object(12)
    memory usage: 439.5+ KB
    


```python
# Describe our data for basic statistics
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobSatisfaction</th>
      <th>MonthlyIncome</th>
      <th>MonthlyRate</th>
      <th>NumCompaniesWorked</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.0</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.0</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1480.000000</td>
      <td>1423.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36.917568</td>
      <td>801.384459</td>
      <td>9.220270</td>
      <td>2.910811</td>
      <td>1.0</td>
      <td>1031.860811</td>
      <td>2.724324</td>
      <td>65.845270</td>
      <td>2.729730</td>
      <td>2.064865</td>
      <td>2.725000</td>
      <td>6504.985811</td>
      <td>14298.460811</td>
      <td>2.687162</td>
      <td>15.210135</td>
      <td>3.153378</td>
      <td>2.708784</td>
      <td>80.0</td>
      <td>0.791892</td>
      <td>11.281757</td>
      <td>2.797973</td>
      <td>2.760811</td>
      <td>7.009459</td>
      <td>4.228378</td>
      <td>2.182432</td>
      <td>4.118060</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.128559</td>
      <td>403.126988</td>
      <td>8.131201</td>
      <td>1.023796</td>
      <td>0.0</td>
      <td>605.955046</td>
      <td>1.092579</td>
      <td>20.328266</td>
      <td>0.713007</td>
      <td>1.105574</td>
      <td>1.104137</td>
      <td>4700.261400</td>
      <td>7112.056802</td>
      <td>2.494098</td>
      <td>3.655338</td>
      <td>0.360474</td>
      <td>1.081995</td>
      <td>0.0</td>
      <td>0.850527</td>
      <td>7.770870</td>
      <td>1.288791</td>
      <td>0.707024</td>
      <td>6.117945</td>
      <td>3.616020</td>
      <td>3.219357</td>
      <td>3.555484</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>102.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1009.000000</td>
      <td>2094.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.000000</td>
      <td>465.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.0</td>
      <td>493.750000</td>
      <td>2.000000</td>
      <td>48.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2922.250000</td>
      <td>8051.000000</td>
      <td>1.000000</td>
      <td>12.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36.000000</td>
      <td>800.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>1.0</td>
      <td>1027.500000</td>
      <td>3.000000</td>
      <td>66.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>4933.000000</td>
      <td>14220.000000</td>
      <td>2.000000</td>
      <td>14.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.000000</td>
      <td>1157.000000</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>1568.250000</td>
      <td>4.000000</td>
      <td>83.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>8383.750000</td>
      <td>20460.500000</td>
      <td>4.000000</td>
      <td>18.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>15.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>60.000000</td>
      <td>1499.000000</td>
      <td>29.000000</td>
      <td>5.000000</td>
      <td>1.0</td>
      <td>2068.000000</td>
      <td>4.000000</td>
      <td>100.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>19999.000000</td>
      <td>26999.000000</td>
      <td>9.000000</td>
      <td>25.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>3.000000</td>
      <td>40.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>40.000000</td>
      <td>18.000000</td>
      <td>15.000000</td>
      <td>17.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Data Cleaning<a class="anchor" id="third-bullet"></a>

### Missing Data


```python
df.isna().sum()
```

We notice there are 57 rows in which the feature 'YearsWithCurrManager' is empty.


```python
# check rows in which there are NaN values
len(df[df['YearsWithCurrManager'].isna()])
```




    57




```python
len(df[df['YearsWithCurrManager'].isna()]) / len(df)
```




    0.038513513513513516



Here we could have two options: either go with the analysis without those rows where 'YearsWithCurrManager' are NaN or not consider this column at all in the analysis/model. I choose at first to go without those rows since they represent only 3.8% of our data and should not affect as much the analysis overall.


```python
# dropping rows with NaN
df.dropna(subset=['YearsWithCurrManager'], inplace=True)
len(df)
```




    1423



### Duplicated data


```python
df.duplicated().sum()
```




    7




```python
# there are duplicated rows, and that makes no sense for employee data, so we promptly delete them
df=df.drop_duplicates(keep='first')
```

### Delete columns that are not relevant


```python
# we shall remove data that are not relevant/helpful to our analysis, such as EmployeeNumber and StandardHours, 
# or columns that are redundant
df['StandardHours'].unique()

df.drop(['EmpID', 'EmployeeNumber', 'StandardHours', 'EmployeeCount', 'AgeGroup', 'Over18'], axis=1, inplace=True)
```

### Outliers


```python
# I'll check for outliers only in the columns they might apply (financial or time-related)
for col in ['DailyRate', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate']:
    fig = px.box(df, x=col)
    fig.update_layout(title=f'Boxplot for {col}', yaxis_title=col)
    fig.show()
```

<img src="images/box1.png" alt="box1" />


```python
# the ouliers are noticed only in MonthlyIncome. If we check the JobRoles for them we notice they're managers and research 
# directors, which justify the high income

df[df['MonthlyIncome']>=16555].JobRole.unique()
```




    array(['Manager', 'Research Director'], dtype=object)



Since we intend to group the employees in different clusters, it makes no sense to remove the outliers as of now, since they might cause an important influence on the grouping algorithm and they should be noticed.

### Mindmap
According to our data and some research on the subject, is it safe to assume that factors that may affect Attrition/Turnover rate can be categorized into **External** factors (family-related, personal life or things that happen outside the company) or **Internal** factors (everything job and function related, as well as salary and interpersonal relationship), resulting in the following mindmap:

<img src="images/Mindmap.png" alt="Alternative text" />

### Initial Hypothesis
Here we can formulate a few hypothesis based on the mindmap and on the business inputs, and have them tested in order to both have a better understanding of the data as well as find interesting relationships between features. Let's suppose the initial hypothesis that were proposed are the following:

1. The higher the 'Age', the more propension of Attrition due to retirement or because companies seek to refresh their workforce with younger individuals
2. The furthest employees with lower compensations are located from work, the more likely they are to leave the company.
3. Job satisfaction is related (same orientation) to job involvement and when both are low, the higher the Attrition
4. Employees that accumulate lots of years since last promotion and are from the lowest job levels have higher Attrition
5. Education field has little or no relation to Attrition at all
6. Poor Work-Life Balance and frequent OverTime increase employee Attrition

# EDA<a class="anchor" id="fourth-bullet"></a>
In the Exploratory Data Analysis step, we will first assess our initial hypothesis for a more structured approach.


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobSatisfaction</th>
      <th>MonthlyIncome</th>
      <th>MonthlyRate</th>
      <th>NumCompaniesWorked</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
      <td>1416.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36.924435</td>
      <td>803.329802</td>
      <td>9.238701</td>
      <td>2.907486</td>
      <td>2.725989</td>
      <td>65.989407</td>
      <td>2.726695</td>
      <td>2.069209</td>
      <td>2.728814</td>
      <td>6516.679379</td>
      <td>14319.355932</td>
      <td>2.711158</td>
      <td>15.199153</td>
      <td>3.151130</td>
      <td>2.704802</td>
      <td>0.799435</td>
      <td>11.298729</td>
      <td>2.802260</td>
      <td>2.762712</td>
      <td>7.037429</td>
      <td>4.254944</td>
      <td>2.213277</td>
      <td>4.117232</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.135606</td>
      <td>404.133280</td>
      <td>8.122617</td>
      <td>1.024395</td>
      <td>1.090169</td>
      <td>20.396197</td>
      <td>0.711953</td>
      <td>1.108023</td>
      <td>1.099266</td>
      <td>4723.565527</td>
      <td>7112.986512</td>
      <td>2.507778</td>
      <td>3.638219</td>
      <td>0.358302</td>
      <td>1.080704</td>
      <td>0.851952</td>
      <td>7.825239</td>
      <td>1.288885</td>
      <td>0.709487</td>
      <td>6.151044</td>
      <td>3.636385</td>
      <td>3.249310</td>
      <td>3.559344</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>102.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1009.000000</td>
      <td>2097.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.000000</td>
      <td>465.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>48.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2922.250000</td>
      <td>8057.500000</td>
      <td>1.000000</td>
      <td>12.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36.000000</td>
      <td>804.500000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>66.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>4938.500000</td>
      <td>14288.500000</td>
      <td>2.000000</td>
      <td>14.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.000000</td>
      <td>1159.000000</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>84.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>8380.250000</td>
      <td>20440.500000</td>
      <td>4.000000</td>
      <td>18.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>15.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>60.000000</td>
      <td>1499.000000</td>
      <td>29.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>100.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>19999.000000</td>
      <td>26999.000000</td>
      <td>9.000000</td>
      <td>25.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>40.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>40.000000</td>
      <td>18.000000</td>
      <td>15.000000</td>
      <td>17.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# if we count the employees with and without a positive Attrition, we can see the following distribution.
#'Yes' labels represent about 15% of the total rows
sns.countplot(data=df, x='Attrition')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18a1e612c50>




    
![png](README_files/README_39_1.png)
    


### Testing the Hypothesis
**1. The higher the 'Age', the more propension of Attrition due to retirement or because companies seek to refresh their workforce with younger individuals**



```python
# first we shall see the distribution of employee by age
plt.figure(figsize=(10,5))
ax = sns.countplot(data=df, x='Age', color='blue')
ax.tick_params(axis='x', rotation=90)
```


    
![png](README_files/README_41_0.png)
    


Then, since the age distribution is not uniform, and there are many distinct values, it makes sense to to group those ages into ranges (bins) by using a method called qcut and then calculate an *Attrition Rate* for each of those ranges by dividing the sum of Attritions = 'Yes' by the amount of employees in each range.


```python
n_bins = 5
df['AgeGroup'] = pd.qcut(df['Age'], q=n_bins, labels=['18-29', '30-34', '35-38', '39-45', '46-60'])
#pd.qcut(df['Age'], q=n_bins).unique()

print(df['AgeGroup'].value_counts())
```

    18-29    316
    30-34    310
    39-45    279
    46-60    265
    35-38    246
    Name: AgeGroup, dtype: int64
    


```python
# calculating the attrition rate
age_group_attrition = df[df['Attrition'] == 'Yes'].groupby('AgeGroup')['Attrition'].count() / df.groupby('AgeGroup')['Attrition'].count() * 100
age_group_attrition
```




    AgeGroup
    18-29    28.164557
    30-34    17.741935
    35-38     8.943089
    39-45    10.394265
    46-60    12.830189
    Name: Attrition, dtype: float64




```python
# plotting the results
plt.figure(figsize=(10,5))
fig = px.bar(age_group_attrition, x=age_group_attrition.index, y=age_group_attrition.values, text_auto='.4s', labels={'y': 'Attrition Rate (%)', 'x': 'Age Group'})
fig.show()
```

<img src="images/fig1.png" alt="fig1" />

Contrary to our first thought, the group that presents the higher attrition rate is the younger one (18-29 years).

Then, we will perform a Chi-Squared analysis to check for Independence. This way we can see if there really is an association between Age and Attrition Rate for each age group:


```python
# creating a temporary table
contingency_table = pd.crosstab(df['AgeGroup'], df['Attrition'])
contingency_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Attrition</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>AgeGroup</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18-29</th>
      <td>227</td>
      <td>89</td>
    </tr>
    <tr>
      <th>30-34</th>
      <td>255</td>
      <td>55</td>
    </tr>
    <tr>
      <th>35-38</th>
      <td>224</td>
      <td>22</td>
    </tr>
    <tr>
      <th>39-45</th>
      <td>250</td>
      <td>29</td>
    </tr>
    <tr>
      <th>46-60</th>
      <td>231</td>
      <td>34</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Chi-Square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Square Test Statistic: {chi2}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("There is a statistically significant difference in attrition rates between age groups.")
else:
    print("There is no statistically significant difference in attrition rates between age groups.")
```

    Chi-Square Test Statistic: 52.62270800373975
    P-value: 1.022038785148418e-10
    There is a statistically significant difference in attrition rates between age groups.
    


```python
# if we calculate the rate for the the younger group and compare it to the rate of all the other groups, we get a relative risk
# of 2.21, which means that this younger group is twice more likely to present an Attrition in comparison to others!
rate_18_29 = df[df['AgeGroup'] == '18-29']['Attrition'].value_counts(normalize=True)['Yes']
rate_other_groups = df[df['AgeGroup'] != '18-29']['Attrition'].value_counts(normalize=True)['Yes']

relative_risk = rate_18_29 / rate_other_groups
print(f"Relative Risk: {relative_risk}")
```

    Relative Risk: 2.212929475587704
    

So Hypothesis 1 turns out to be **FALSE**. Group with **ages 18-29 is the one that has the highest Attrition Rate** and this was proven by a chi-square test at the 95% confidence level. The oldest one has the third higher Attrition rate out of 5.

**2. The furthest employees with lower compensations are located from work, the more likely they are to leave the company.**

First we shall analyse the relationship between the compensation columns. We have 'DailyRate', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', but it's not necessary to use all of them. We choose to use 'MonthlyIncome' since it's the one that varies the most (and is compatible) with each 'JobLevel' step.


```python
job_level_avg = df.groupby('JobLevel')['MonthlyIncome'].mean().reset_index()
fig = px.bar(job_level_avg, 
             x='JobLevel', 
             y='MonthlyIncome', 
             title='Average MonthlyIncome by JobLevel',
             labels={'JobLevel': 'Job Level', 'MonthlyIncome': 'Average MonthlyIncome'},
             text_auto=True)
fig.show()
```

<img src="images/fig2.png" alt="fig2" />

Similarly to Age, DistanceFromHome contains many unique values, so we create ranges as well:


```python
n_bins = 5
pd.qcut(df['DistanceFromHome'], q=n_bins).unique()
```




    [(2.0, 5.0], (9.0, 17.0], (5.0, 9.0], (0.999, 2.0], (17.0, 29.0]]
    Categories (5, interval[float64, right]): [(0.999, 2.0] < (2.0, 5.0] < (5.0, 9.0] < (9.0, 17.0] < (17.0, 29.0]]




```python
df['DistanceGroup'] = pd.qcut(df['DistanceFromHome'], q=n_bins, labels=['Very Near', 'Near', 'Moderate', 'Far', 'Very Far'])
print(df['DistanceGroup'].value_counts())
```

    Very Near    401
    Moderate     295
    Very Far     270
    Far          245
    Near         205
    Name: DistanceGroup, dtype: int64
    

We do the same for the income:


```python
df['CompensationGroup'] = pd.qcut(df['MonthlyIncome'], q=4, labels=["Low", "Medium", "High", "Very High"])
```

We can use scatterplot to view the relation between both categories:


```python
sns.scatterplot(data=df, x='DistanceFromHome', y='MonthlyIncome', hue='Attrition')
plt.show()
```


    
![png](README_files/README_63_0.png)
    


Not much can be taken from this, but it seems that lower compensations tend to have more Attrition, which tipically would make sense. Let's develop it further:


```python
# again we shall use chi-square test in order to check for significance between Attrition and DistanceGroup
contingency_table_distance = pd.crosstab(df['DistanceGroup'], df['Attrition'])

# Chi-Square Test
chi2_distance, p_distance, dof_distance, expected_distance = stats.chi2_contingency(contingency_table_distance)

print(f"Chi-Square Statistic for DistanceGroup: {chi2_distance}")
print(f"P-value for DistanceGroup: {p_distance}")
```

    Chi-Square Statistic for DistanceGroup: 7.621635296523076
    P-value for DistanceGroup: 0.10646376469344543
    


```python
# significance between Attrition and CompensationGroup
contingency_table_comp = pd.crosstab(df['CompensationGroup'], df['Attrition'])

# Performing Chi-Square Test
chi2_comp, p_comp, dof_comp, expected_comp = stats.chi2_contingency(contingency_table_comp)

print(f"Chi-Square Statistic for CompensationGroup: {chi2_comp}")
print(f"P-value for CompensationGroup: {p_comp}")
```

    Chi-Square Statistic for CompensationGroup: 72.98711293746298
    P-value for CompensationGroup: 9.782402293640052e-16
    


```python
# checking rates of Attrition for each group
for _ in df.DistanceGroup.unique():
    print(_)
    display(df[df['DistanceGroup'] == _]['Attrition'].value_counts(normalize=True))
```

    Near
    


    No     0.84878
    Yes    0.15122
    Name: Attrition, dtype: float64


    Far
    


    No     0.812245
    Yes    0.187755
    Name: Attrition, dtype: float64


    Moderate
    


    No     0.854237
    Yes    0.145763
    Name: Attrition, dtype: float64


    Very Near
    


    No     0.865337
    Yes    0.134663
    Name: Attrition, dtype: float64


    Very Far
    


    No     0.796296
    Yes    0.203704
    Name: Attrition, dtype: float64



```python
# checking rates of Attrition for each group
for _ in df.CompensationGroup.unique():
    print(_)
    display(df[df['CompensationGroup'] == _]['Attrition'].value_counts(normalize=True))
```

    Low
    


    No     0.694915
    Yes    0.305085
    Name: Attrition, dtype: float64


    Medium
    


    No     0.867232
    Yes    0.132768
    Name: Attrition, dtype: float64


    High
    


    No     0.892655
    Yes    0.107345
    Name: Attrition, dtype: float64


    Very High
    


    No     0.898305
    Yes    0.101695
    Name: Attrition, dtype: float64


As noted by the chi-squared test's p-value and rate of Attrition='Yes' for each CompensationGroup, we see that there's a **very significant relationship between Compensation and Attrition**, where Low compensations show a rate of 30% of Attrition! As for distance, it doesn't seem to have a lot of impact in Attrition. So Hypothesis 2 is **PARTIALLY TRUE**.

**3. Job satisfaction is related (same orientation) to job involvement and when both are low, the higher the Attrition**

Now we should analyse the relationship between job satisfaction and job involvement. We do that by checking the correlation:


```python
pd.crosstab(df['JobInvolvement'], df['JobSatisfaction'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>JobSatisfaction</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>JobInvolvement</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>10</td>
      <td>28</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>73</td>
      <td>81</td>
      <td>104</td>
      <td>107</td>
    </tr>
    <tr>
      <th>3</th>
      <td>158</td>
      <td>159</td>
      <td>254</td>
      <td>262</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31</td>
      <td>24</td>
      <td>41</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr_matrix = df[['JobInvolvement', 'JobSatisfaction']].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation between Job Involvement and Job Satisfaction')
plt.show()
```


    
![png](README_files/README_73_0.png)
    


From what we can see by both the crosstab and the correlation heatmap, both features are not intimately related. So we need to analyse the impact of each one towards Attrition:

* JobInvolvement:


```python
fig = px.histogram(df, x='JobInvolvement', color='Attrition', histfunc='sum', 
                   barnorm='percent', nbins=4, text_auto=True,
                   category_orders={'JobInvolvement': ['1', '2', '3', '4']})
fig.update_traces(hovertemplate='Job Involvement: %{x}<br>Weighted Count: %{y}')
fig.update_layout(title='Job Involvement by Attrition', xaxis_title='Job Involvement', yaxis_title='Weighted Count')
fig.show()
```

<img src="images/fig3.png" alt="fig3" />


```python
contingency_table = pd.crosstab(df['JobInvolvement'], df['Attrition'])

# Chi-Square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Square Test Statistic: {chi2}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("There is a statistically significant difference in attrition rates between JobInvolvement")
else:
    print("There is no statistically significant difference in attrition rates between JobInvolvement")
```

    Chi-Square Test Statistic: 24.722546069649113
    P-value: 1.7646599376375843e-05
    There is a statistically significant difference in attrition rates between JobInvolvement
    

* JobSatisfaction:


```python
fig = px.histogram(df, x='JobSatisfaction', color='Attrition', histfunc='sum', 
                   barnorm='percent', nbins=4, text_auto=True,
                   category_orders={'JobSatisfaction': ['1', '2', '3', '4']})
fig.update_traces(hovertemplate='Job Involvement: %{x}<br>Weighted Count: %{y}')
fig.update_layout(title='Job Satisfaction by Attrition', xaxis_title='Job Satisfaction', yaxis_title='Weighted Count')
fig.show()
```

<img src="images/fig4.png" alt="fig4" />


```python
contingency_table = pd.crosstab(df['JobSatisfaction'], df['Attrition'])

# Chi-Square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Square Test Statistic: {chi2}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("There is a statistically significant difference in attrition rates between JobSatisfaction")
else:
    print("There is no statistically significant difference in attrition rates between JobSatisfaction")
```

    Chi-Square Test Statistic: 15.885986380435646
    P-value: 0.0011966782844478302
    There is a statistically significant difference in attrition rates between JobSatisfaction
    

We see that both present a significant relationship with Attrition, but we still want to confirm if there's a pattern between them and attrition. One option is to plot a crosstab with the rates for Attrition for each combination (pair) of JobInvolvement and JobSatisfaction values.


```python
crosstab = pd.crosstab([df['JobInvolvement'], df['JobSatisfaction']], df['Attrition'], normalize='index')
crosstab.plot(kind='bar', stacked=True, color=['green', 'red'])
plt.title('Attrition Rate by Job Involvement and Job Satisfaction')
plt.xlabel('Job Involvement and Job Satisfaction')
plt.ylabel('Rate / proportion')
plt.show()
```


    
![png](README_files/README_84_0.png)
    


It is clear that Attrition rates are higher for individuals whose 'Job Involvement' levels were 1 and when 'Job Satisfaction' is 1 together with 'Job Involvement' 2. Overall, we can confirm that **Job Involvement** is the factor that influences the most the Attrition.

For this Hypothesis, we conclude that it's **PARTIALLY TRUE** as well. Even that those two features are not as closely related as one could think, we notice that a low JobInvolvement is significantly related to Attrition.

**4. Employees that accumulate lots of years since last promotion and are from the lowest job levels have higher Attrition**


It makes sense to think that employees that are not gratified with promotions for several years and have lower job levels tend to be less motivated and more inclined to leave the company. First let's analyze the relationship between this feature and Attrition.


```python
crosstab = pd.crosstab(df['YearsSinceLastPromotion'], df['Attrition'])

chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)

print(f"Chi-Square Test Statistic: {chi2}")
print(f"P-value: {p_value}")
```

    Chi-Square Test Statistic: 20.82733094024123
    P-value: 0.14243746588971717
    


```python
sns.countplot(data=df, x='YearsSinceLastPromotion', hue='Attrition')
plt.title("Attrition by Years Since Last Promotion")
plt.show()
```


    
![png](README_files/README_90_0.png)
    


There doesn't seem to be any significant relationship at all between high 'YearsSinceLastPromotion' and Attrition. In fact rates are very similar for every value:


```python
for _ in df.YearsSinceLastPromotion.unique():
    print(_)
    display(df[df['YearsSinceLastPromotion'] == _]['Attrition'].value_counts(normalize=True))
```

    0
    


    No     0.809695
    Yes    0.190305
    Name: Attrition, dtype: float64


    1
    


    No     0.863372
    Yes    0.136628
    Name: Attrition, dtype: float64


    2
    


    No     0.821192
    Yes    0.178808
    Name: Attrition, dtype: float64


    3
    


    No     0.823529
    Yes    0.176471
    Name: Attrition, dtype: float64


    5
    


    No     0.954545
    Yes    0.045455
    Name: Attrition, dtype: float64


    4
    


    No     0.913793
    Yes    0.086207
    Name: Attrition, dtype: float64


    7
    


    No     0.786667
    Yes    0.213333
    Name: Attrition, dtype: float64


    6
    


    No     0.83871
    Yes    0.16129
    Name: Attrition, dtype: float64


    8
    


    No    1.0
    Name: Attrition, dtype: float64


    10
    


    No     0.833333
    Yes    0.166667
    Name: Attrition, dtype: float64


    9
    


    No     0.8125
    Yes    0.1875
    Name: Attrition, dtype: float64


    11
    


    No     0.913043
    Yes    0.086957
    Name: Attrition, dtype: float64


    12
    


    No    1.0
    Name: Attrition, dtype: float64


    15
    


    No     0.769231
    Yes    0.230769
    Name: Attrition, dtype: float64


    13
    


    No     0.8
    Yes    0.2
    Name: Attrition, dtype: float64


    14
    


    No     0.888889
    Yes    0.111111
    Name: Attrition, dtype: float64


Now let's check if there is any sort of relationship between 'YearsSinceLastPromotion' and Attrition when we consider 'JobLevel' as well. To facilitate let's also group up 'YearsSinceLastPromotion' in ranges:


```python
conditions = [
    (df['YearsSinceLastPromotion'] >= 0) & (df['YearsSinceLastPromotion'] <= 3),
    (df['YearsSinceLastPromotion'] >= 4) & (df['YearsSinceLastPromotion'] <= 7),
    (df['YearsSinceLastPromotion'] >= 8) & (df['YearsSinceLastPromotion'] <= 11),
    (df['YearsSinceLastPromotion'] >= 12) & (df['YearsSinceLastPromotion'] <= 15)
]

ranges = ['0-3', '4-7', '8-11', '12-15']
df['YSLPGroup'] = np.select(conditions, ranges, default='15+')
```


```python
crosstab_job_promo = pd.crosstab([df['JobLevel'], df['YSLPGroup']], df['Attrition'])
chi2, p_value, _, _ = stats.chi2_contingency(crosstab_job_promo)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p_value}")
```

    Chi-Square Statistic: 78.07145800283662
    P-value: 3.988565926785441e-09
    

Now the chi-square test clearly points out a strong relationship when we consider both features combined! For further analysis, we plot:


```python
crosstab_job_promo = pd.crosstab([df['JobLevel'], df['YSLPGroup']], df['Attrition'], normalize='index')
crosstab_job_promo.plot(kind='bar', stacked=True, figsize=(12, 5))
plt.title("Attrition Rate by Job Level and Years Since Last Promotion")
plt.ylabel("Proportion")
plt.show()
```


    
![png](README_files/README_97_0.png)
    


Now we can clearly observe that Attrition rates are higher when JobLevel is equal to 1 and in the bars where YearsSinceLastPromotion range from 8-11 and 12-15. This makes the Hypothesis **TRUE**.

**5. Education field has little or no relation to Attrition at all**

Similarly, we plot a crosstab and perform a chi-square test:


```python
crosstab_education_attrition = pd.crosstab(df['EducationField'], df['Attrition'])
print(crosstab_education_attrition)
```

    Attrition          No  Yes
    EducationField            
    Human Resources    19    7
    Life Sciences     497   88
    Marketing         119   34
    Medical           385   57
    Other              70   11
    Technical Degree   97   32
    


```python
crosstab_education_attrition_norm = pd.crosstab(df['EducationField'], df['Attrition'], normalize='index')
crosstab_education_attrition_norm.plot(kind='bar', stacked=True)
plt.title("Attrition Rate by EducationField")
plt.ylabel("Proportion")
plt.show()
```


    
![png](README_files/README_102_0.png)
    



```python
chi2_stat, p_val, dof, ex = stats.chi2_contingency(crosstab_education_attrition)

print(f"Chi-Square Statistic: {chi2_stat}")
print(f"P-value: {p_val}")
```

    Chi-Square Statistic: 17.89249419919997
    P-value: 0.0030842011625653496
    

Contrary to what the Hypothesis says, Education Field indeed has some significant relationship in relation to Attrition! **Human Resources, Marketing and Techincal Degree** are the backgrounds that show the highest Attrition rates. So the Hypothesis is **FALSE**. 

**6. Poor Work-Life Balance and frequent OverTime increase employee Attrition**

Plotting the countplot and checking the relationship (chi-square) between WorkLifeBalance and Attrition:


```python
sns.countplot(data=df, x='WorkLifeBalance', hue='Attrition')
plt.title('Attrition by Work-Life Balance')
plt.show()
```


    
![png](README_files/README_107_0.png)
    



```python
crosstab_wlb_attrition_norm = pd.crosstab(df['WorkLifeBalance'], df['Attrition'], normalize='index')
crosstab_wlb_attrition_norm.plot(kind='bar', stacked=True)
plt.title("Attrition Rate by WorkLifeBalance")
plt.ylabel("Proportion")
plt.show()
```


    
![png](README_files/README_108_0.png)
    



```python
crosstab_wlb_attrition = pd.crosstab(df['WorkLifeBalance'], df['Attrition'])
chi2_stat, p_val, dof, ex = stats.chi2_contingency(crosstab_wlb_attrition)

print(f"Chi-Square Statistic for WorkLifeBalance: {chi2_stat}")
print(f"P-value for WorkLifeBalance: {p_val}")
```

    Chi-Square Statistic for WorkLifeBalance: 16.85060840367404
    P-value for WorkLifeBalance: 0.0007585400406444005
    

Plotting and checking the relationship between OverTime and Attrition:


```python
sns.countplot(data=df, x='OverTime', hue='Attrition')
plt.title('Attrition by OverTime')
plt.show()
```


    
![png](README_files/README_111_0.png)
    



```python
crosstab_ot_attrition_norm = pd.crosstab(df['OverTime'], df['Attrition'], normalize='index')
crosstab_ot_attrition_norm.plot(kind='bar', stacked=True)
plt.title("Attrition Rate by OverTime")
plt.ylabel("Proportion")
plt.show()
```


    
![png](README_files/README_112_0.png)
    



```python
crosstab_ot_attrition = pd.crosstab(df['OverTime'], df['Attrition'])
chi2_stat, p_val, dof, ex = stats.chi2_contingency(crosstab_ot_attrition)

print(f"Chi-Square Statistic for OverTime: {chi2_stat}")
print(f"P-value for OverTime: {p_val}")
```

    Chi-Square Statistic for OverTime: 88.80401326788505
    P-value for OverTime: 4.359306478857773e-21
    

Looking at the obtained results it's possible that there might be an important influence of the 'Work-Life Balance' aspect on the Attrition rate, but when it comes to 'Over Time' this relationship is very strong. Let's analyse them both combined just to see the proportions:


```python
crosstab_wlb_ot = pd.crosstab([df['WorkLifeBalance'], df['OverTime']], df['Attrition'])
chi2, p_value, _, _ = stats.chi2_contingency(crosstab_wlb_ot)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p_value}")
```

    Chi-Square Statistic: 108.3626889213076
    P-value: 2.0068481142934058e-20
    


```python
crosstab_wlb_ot_normalized = pd.crosstab([df['WorkLifeBalance'], df['OverTime']], df['Attrition'], normalize='index')
crosstab_wlb_ot_normalized.plot(kind='bar', stacked=True)
plt.title("Attrition Rate by WorkLifeBalance and OverTime")
plt.ylabel("Proportion")
plt.show()
```


    
![png](README_files/README_116_0.png)
    


It's very clear that Over Time is a major factor when it comes to Attrition, and the rates are slightly higher when Work-Life Balance is low. It makes sense to think that Over Time may even directly influence a lower Work-Life Balance grade. This Hypothesis is **TRUE**.

### Insights Obtained

Overall from what we could notice, we can assemble some **actionable insights** in the form of **recommendations** to the HR department, regarding each of the raised Hypothesis:

**1.The higher the 'Age', the more propension of Attrition due to retirement or because companies seek to refresh their workforce with younger individuals (FALSE):**
- The group with highest attrition is the younger one, so this means there should be reasons as why these employees are leaving.
- It's suggested that the company focus on retaining younger employees, possibly by addressing their needs for career development, competitive compensation, and implement a deeper study on their work-life balance.
- Mentorship programs are also good alternatives to engage younger employees and help them feel more connected to the organization.

**2.The furthest employees with lower compensations are located from work, the more likely they are to leave the company (PARTIALLY TRUE):**
- Lower compensation is the major factor here that may be related to attrition, while distance have little to no impact.
- Consider improving compensation packages, especially for lower-paid employees.
- Conduct a survey on the employees' thoughts on commuting distance and their overall job involvement. Maybe offer transportation benefits or remote work options to mitigate any potential dissatisfaction due to commuting distance.

**3.Job satisfaction is related (same orientation) to job involvement and when both are low, the higher the Attrition (PARTIALLY TRUE):**
- While job satisfaction and job involvement are not significantly related, low job involvement is associated with higher attrition.
- Regularly measure job involvement and enhance it by understanding what makes their work more engaging.
- Implement a more rewarding and more positive culture regarding employees' contributions and their recognition.
- Look for more opportunities for growth and development.

**4.Employees that accumulate lots of years since last promotion and are from the lowest job levels have higher Attrition (TRUE):**
- Attrition rates are indeed higher for employees at Job Level 1 who haven't been promoted for 8-15 years.
- Consider offering more opportunities for promotions, skill development, and role enhancements to reduce stagnation.
- Develop clear career progression paths for employees, especially those at lower job levels, and collect their inputs and expectations on the matter.

**5.Education field has little or no relation to Attrition at all (FALSE):**
- Although not very strong, Human Resources, Marketing, and Technical Degree backgrounds show some relationship with Attrition.
- Further analyse which departments have a majority of employees with those backgrounds, and if there is a pattern.
- Tailor retention strategies and development programs for employees based on their educational backgrounds. For example provide up-to-date specialization courses and carrer path opportunities for those professionals.

**6.Poor Work-Life Balance and frequent OverTime increase employee Attrition (TRUE):**
- Overtime is a major factor contributing to higher attrition, and poor work-life balance also correlates with slightly higher attrition rates.
- Implement policies that promote a better work-life balance, such as flexible working hours, work-from-home options, and limiting overtime
- Regularly monitor and adjust workloads to prevent burnout and ensure employees have a healthy balance between work and personal life. Identify if there is a pattern regaring specific departments and/or managers with more employees working overtime in order to reduce it.
- Consider improving the office's infrastructure, environmental and well-being factors such as snacks, drinks, good internet connection, leisure spots...

# Feature Engineering<a class="anchor" id="fifth-bullet"></a>
Now our next goal is to classify the employees into clusters, in order to check for major differences and patterns between them, which will help the HR dept. have new ideas and insights to tackle specific employee groups, together with our previous suggestions.

Now aiming to select and train a clustering ML model, we must first prepare our data and choose which columns/features to include in the model training.

### Feature Selection

We can discard some variables that are colinear, or in other words, express the same information and would add complexity and redundancy to the model.


```python
# create a copy of the dataset if needed
df_eda = df.copy()
```


```python
# for compensation, we are going to choose MonthlyIncome as our main feature
# discard 'DailyRate', 'HourlyRate', 'MonthlyRate', 'AgeGroup', 'DistanceGroup', 'CompensationGroup', 'YSLPGroup'
df.head()
df.drop(['DailyRate', 'HourlyRate', 'MonthlyRate', 'AgeGroup', 'DistanceGroup', 'CompensationGroup', 'YSLPGroup'], axis=1, inplace=True)
```


```python
# drop our target, which is what we want to analyse after cluster formation
df.drop(['Attrition'], axis=1, inplace=True)
```

Applying a heatmap to the correlation table for all columns, we can analyse which features are possibly colinear (do not contribute to the model training and might affect performance and results):


```python
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap="YlGnBu", ax=ax);
```


    
![png](README_files/README_127_0.png)
    


According to convention, when two features have correlation next to 0.8 they can already be considered highly correlated, and thus one of them can be removed from the dataset. In our case, we will drop **'TotalWorkingYears'**, because it seems to be highly correlated with JobLevel and MonthlyIncome (it makes sense as a natural progression in a typical career). We also drop **'YearsAtCompany'** due to its high correlation to other 'Year' columns. Another relationship that is almost fully dependent is the one between JobLevel and MonthlyIncome. In this case I chose to drop **'JobLevel'** because MonthlyIncome is more granular and can capture nuances that JobLevel might miss.


```python
df.drop(['TotalWorkingYears', 'YearsAtCompany', 'JobLevel'], axis=1, inplace=True)
```

### Encoding

In order to include the categorical columns into our analysis, they must be turned into numeric columns. This is done by using an one-hot encoding technique. This time we'll use get_dummies() method due to it's simplicity.


```python
# getting non-numeric columns
non_numeric_columns = df.select_dtypes(exclude='number').columns
for _ in non_numeric_columns:
    print(f'{_}->\n{df[_].unique()}\n')
```

    BusinessTravel->
    ['Travel_Rarely' 'Travel_Frequently' 'Non-Travel' 'TravelRarely']
    
    Department->
    ['Research & Development' 'Sales' 'Human Resources']
    
    EducationField->
    ['Life Sciences' 'Medical' 'Marketing' 'Technical Degree' 'Other'
     'Human Resources']
    
    Gender->
    ['Male' 'Female']
    
    JobRole->
    ['Laboratory Technician' 'Sales Representative' 'Research Scientist'
     'Human Resources' 'Manufacturing Director' 'Sales Executive'
     'Healthcare Representative' 'Research Director' 'Manager']
    
    MaritalStatus->
    ['Single' 'Divorced' 'Married']
    
    SalarySlab->
    ['Upto 5k' '5k-10k' '10k-15k' '15k+']
    
    OverTime->
    ['No' 'Yes']
    
    


```python
df_encoded = pd.get_dummies(df, columns=non_numeric_columns, drop_first=True)
df_encoded.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EnvironmentSatisfaction</th>
      <th>JobInvolvement</th>
      <th>JobSatisfaction</th>
      <th>MonthlyIncome</th>
      <th>NumCompaniesWorked</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StockOptionLevel</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
      <th>BusinessTravel_TravelRarely</th>
      <th>BusinessTravel_Travel_Frequently</th>
      <th>BusinessTravel_Travel_Rarely</th>
      <th>Department_Research &amp; Development</th>
      <th>Department_Sales</th>
      <th>EducationField_Life Sciences</th>
      <th>EducationField_Marketing</th>
      <th>EducationField_Medical</th>
      <th>EducationField_Other</th>
      <th>EducationField_Technical Degree</th>
      <th>Gender_Male</th>
      <th>JobRole_Human Resources</th>
      <th>JobRole_Laboratory Technician</th>
      <th>JobRole_Manager</th>
      <th>JobRole_Manufacturing Director</th>
      <th>JobRole_Research Director</th>
      <th>JobRole_Research Scientist</th>
      <th>JobRole_Sales Executive</th>
      <th>JobRole_Sales Representative</th>
      <th>MaritalStatus_Married</th>
      <th>MaritalStatus_Single</th>
      <th>SalarySlab_15k+</th>
      <th>SalarySlab_5k-10k</th>
      <th>SalarySlab_Upto 5k</th>
      <th>OverTime_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1420</td>
      <td>1</td>
      <td>13</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>10</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>1200</td>
      <td>1</td>
      <td>12</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1878</td>
      <td>1</td>
      <td>14</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>1051</td>
      <td>1</td>
      <td>15</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1904</td>
      <td>1</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# I'd run a heatmap for correlation again here, but since a lot of new columns were added, it would become cluttered and hard to analyse.
# Because of such, I decided to include only a few categorical columns here on this second run of the heatmap

df_encoded_alt = pd.get_dummies(df, columns=['Department','Gender','MaritalStatus','SalarySlab','OverTime'], drop_first=True)

fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df_encoded_alt.corr(), vmin=-1, vmax=1, annot=True, cmap="YlGnBu", ax=ax);
```


    
![png](README_files/README_134_0.png)
    



```python
# After dropping a few more extra columns, we are ready to move on
df_encoded.drop(['SalarySlab_15k+', 'PerformanceRating'], axis=1, inplace=True)
```


```python
# we have at the moment 40 columns
len(df_encoded.columns)
```




    40



### Variance Thresholding

Another very common tecnique for feature selection involves checking the variance. Features with low variance might not contribute much to the model training. For this we use the *VarianceThreshold* method from sklearn.


```python
# save a copy
df_feature = df_encoded.copy()
```


```python
from sklearn.feature_selection import VarianceThreshold

# Define threshold. The best value for this is generally determined by trial and error.
threshold = 0.85 # this removes columns whose values are 15% or more similar.

selector = VarianceThreshold(threshold)
df_high_variance = selector.fit_transform(df_encoded)

# Get the indices of the kept features
kept_features = selector.get_support(indices=True)
print(df_encoded.iloc[:, kept_features].columns)

# Create a DataFrame with the remaining features
df_high_variance = df_encoded.iloc[:, kept_features]

# this shows the columns that have been removed
#df_encoded.columns[~selector.get_support()]
```

    Index(['Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
           'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
           'PercentSalaryHike', 'RelationshipSatisfaction',
           'TrainingTimesLastYear', 'YearsInCurrentRole',
           'YearsSinceLastPromotion', 'YearsWithCurrManager'],
          dtype='object')
    

### Feature Importance

At last, another useful method for selecting features is by determining the score of feature importance. This can be done by implementing a tree-based model exclusively for this purpose, for example a *Random Forest Classifier*. It specializes into determining the best features that separate the dataset.

This can be done by instantiating the method from sklearn.ensemble and then invoking the attribute 'feature_importances_'.


```python
# The RF model needs a target 'y', which is the Attrition column
y = df_eda['Attrition']
```


```python
from sklearn.ensemble import RandomForestClassifier

# Fit a random forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(df_encoded, y) # using default 100 trees

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for easy visualization
feature_importances = pd.DataFrame({'Feature': df_encoded.columns, 'Importance': importances})

# Sort by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
feature_importances
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>MonthlyIncome</td>
      <td>0.108535</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>0.081827</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DistanceFromHome</td>
      <td>0.061594</td>
    </tr>
    <tr>
      <th>39</th>
      <td>OverTime_Yes</td>
      <td>0.057394</td>
    </tr>
    <tr>
      <th>15</th>
      <td>YearsWithCurrManager</td>
      <td>0.050052</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PercentSalaryHike</td>
      <td>0.047038</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NumCompaniesWorked</td>
      <td>0.043056</td>
    </tr>
    <tr>
      <th>13</th>
      <td>YearsInCurrentRole</td>
      <td>0.041463</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EnvironmentSatisfaction</td>
      <td>0.037804</td>
    </tr>
    <tr>
      <th>14</th>
      <td>YearsSinceLastPromotion</td>
      <td>0.035669</td>
    </tr>
    <tr>
      <th>10</th>
      <td>StockOptionLevel</td>
      <td>0.034151</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TrainingTimesLastYear</td>
      <td>0.032612</td>
    </tr>
    <tr>
      <th>9</th>
      <td>RelationshipSatisfaction</td>
      <td>0.031547</td>
    </tr>
    <tr>
      <th>5</th>
      <td>JobSatisfaction</td>
      <td>0.031469</td>
    </tr>
    <tr>
      <th>12</th>
      <td>WorkLifeBalance</td>
      <td>0.030761</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JobInvolvement</td>
      <td>0.030121</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Education</td>
      <td>0.027476</td>
    </tr>
    <tr>
      <th>36</th>
      <td>MaritalStatus_Single</td>
      <td>0.023249</td>
    </tr>
    <tr>
      <th>17</th>
      <td>BusinessTravel_Travel_Frequently</td>
      <td>0.017690</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Department_Sales</td>
      <td>0.012651</td>
    </tr>
    <tr>
      <th>28</th>
      <td>JobRole_Laboratory Technician</td>
      <td>0.012307</td>
    </tr>
    <tr>
      <th>38</th>
      <td>SalarySlab_Upto 5k</td>
      <td>0.012194</td>
    </tr>
    <tr>
      <th>18</th>
      <td>BusinessTravel_Travel_Rarely</td>
      <td>0.012151</td>
    </tr>
    <tr>
      <th>23</th>
      <td>EducationField_Medical</td>
      <td>0.011856</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Gender_Male</td>
      <td>0.011697</td>
    </tr>
    <tr>
      <th>34</th>
      <td>JobRole_Sales Representative</td>
      <td>0.011634</td>
    </tr>
    <tr>
      <th>25</th>
      <td>EducationField_Technical Degree</td>
      <td>0.011349</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MaritalStatus_Married</td>
      <td>0.010666</td>
    </tr>
    <tr>
      <th>21</th>
      <td>EducationField_Life Sciences</td>
      <td>0.010401</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Department_Research &amp; Development</td>
      <td>0.009617</td>
    </tr>
    <tr>
      <th>22</th>
      <td>EducationField_Marketing</td>
      <td>0.009488</td>
    </tr>
    <tr>
      <th>33</th>
      <td>JobRole_Sales Executive</td>
      <td>0.008872</td>
    </tr>
    <tr>
      <th>32</th>
      <td>JobRole_Research Scientist</td>
      <td>0.008632</td>
    </tr>
    <tr>
      <th>37</th>
      <td>SalarySlab_5k-10k</td>
      <td>0.006055</td>
    </tr>
    <tr>
      <th>30</th>
      <td>JobRole_Manufacturing Director</td>
      <td>0.004464</td>
    </tr>
    <tr>
      <th>27</th>
      <td>JobRole_Human Resources</td>
      <td>0.004142</td>
    </tr>
    <tr>
      <th>24</th>
      <td>EducationField_Other</td>
      <td>0.003883</td>
    </tr>
    <tr>
      <th>29</th>
      <td>JobRole_Manager</td>
      <td>0.002676</td>
    </tr>
    <tr>
      <th>31</th>
      <td>JobRole_Research Director</td>
      <td>0.001473</td>
    </tr>
    <tr>
      <th>16</th>
      <td>BusinessTravel_TravelRarely</td>
      <td>0.000286</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Looking at the table above, we should try selecting features that have an importance level higher than the threshold.
# A value of 0.05 was chosen after first testing with 0.01 and 0.03, because it yielded better results in the clustering algorithm

# Set a threshold for feature importance
threshold = 0.05

# Keep only the features above the threshold
important_features = feature_importances[feature_importances['Importance'] > threshold]['Feature']

# Subset the original DataFrame to keep only important features
df_importance = df_encoded[important_features]
```

With the dataframes resulting from both variance and feature importance analysis at hand, we get the common columns:


```python
df_high_variance.columns
```




    Index(['Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
           'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
           'PercentSalaryHike', 'RelationshipSatisfaction',
           'TrainingTimesLastYear', 'YearsInCurrentRole',
           'YearsSinceLastPromotion', 'YearsWithCurrManager'],
          dtype='object')




```python
df_importance.columns
```




    Index(['MonthlyIncome', 'Age', 'DistanceFromHome', 'OverTime_Yes',
           'YearsWithCurrManager'],
          dtype='object')



Now we have two options: either proceed with the modeling for each dataframe, or create another one with all their columns combined.
They'll be run separately due to their difference in the number of columns, so we can observe what happens to each of them.


```python
#columns_high_variance = set(df_high_variance.columns)
#columns_importance = set(df_importance.columns)

#combined_columns = list(columns_high_variance.union(columns_importance))
```


```python
#df_combined = df_encoded[combined_columns]
```

### Scaling

Scaling is a very important step before our modeling because it makes all numeric data be at the same scale. This is especially true when talking about clustering algorithms because it allows the model to take all features into consideration the same way when calculating distances, instead of prioritizing a few features that may have greater range than others.


```python
# initialize Scaler
scaler = MinMaxScaler()
df_scaled_hv = pd.DataFrame(scaler.fit_transform(df_high_variance), columns=df_high_variance.columns)
df_scaled_imp = pd.DataFrame(scaler.fit_transform(df_importance), columns=df_importance.columns)

df_scaled_imp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MonthlyIncome</th>
      <th>Age</th>
      <th>DistanceFromHome</th>
      <th>OverTime_Yes</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.021643</td>
      <td>0.0</td>
      <td>0.071429</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.010058</td>
      <td>0.0</td>
      <td>0.321429</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.045761</td>
      <td>0.0</td>
      <td>0.142857</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.002212</td>
      <td>0.0</td>
      <td>0.142857</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.047130</td>
      <td>0.0</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# save a copy
df_ml_hv = df_scaled_hv.copy()
df_ml_imp = df_scaled_imp.copy()
```

# Choosing ML models<a class="anchor" id="sixth-bullet"></a>
In this step we have a few choices of clustering models to choose from. We shall test with two of the most commonly used models:
- K-Means
- DBScan

### K-Means
K-Means is an algorithm that works like the following: it creates K centroids in the N-dimensional plane, and calculates the distances between every point of the dataset and the centroids. The ones closest to the centroids are assigned the same cluster label. Then, the centroids are repositioned at the center of those points of the same label, and again the distances are calculated. This process of creating new clusters and labeling points repeats many times until the end of the process.

In order to implement it we must inform the number of clusters (K) we would like to split our data into. In order not to just guess a value for K, there's a test called Elbow Method, in which we run the algorithm with various values of K and check a metric called inertia, which indicates cluster formation stability. When the inertia starts to stabilize, we check the respective K value and that indicates a good cluster number parameter.


```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```


```python
# We will test from 2 to 15 clusters
inertia_hv = []
inertia_imp = []
range_clusters = range(2, 16)

# high variance dataset
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42) # random_state serves to freeze the memory variables aiming reproducibility
    kmeans.fit(df_scaled_hv)
    inertia_hv.append(kmeans.inertia_)
    
# high importance dataset
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42) # random_state serves to freeze the memory variables aiming reproducibility
    kmeans.fit(df_scaled_imp)
    inertia_imp.append(kmeans.inertia_)
```


```python
# Plotting the Elbow graph, we must check where the inertia starts to decreasce more slowly, indicating cluster formation stability

fig, axs = plt.subplots(1, 2, figsize=(12,5))

axs[0].plot(range_clusters, inertia_hv, marker='o')
axs[0].set_title('Elbow Method for Optimal K - High Variance')
axs[0].set_xlabel('Number of Clusters')
axs[0].set_ylabel('Inertia')
    
axs[1].plot(range_clusters, inertia_imp, marker='o')
axs[1].set_title('Elbow Method for Optimal K - Importance')
axs[1].set_xlabel('Number of Clusters')
axs[1].set_ylabel('Inertia')

plt.tight_layout()
plt.show()
```


    
![png](README_files/README_160_0.png)
    


For the high variance dataset it's noticed a slightly slower decreasce of inertia around K=5, although not very apparent. But for the importance dataset we can clearly observe an aparent bent at K=5. Let's perform a few more steps to determine the ideal number of clusters.

Another useful metric to check is the Silhouette Score. It measures how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates better clustering (well-defined boundaries).


```python
# testing silhouette scores for different values
silhouette_scores_hv = []
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled_hv)
    score = silhouette_score(df_scaled_hv, kmeans.labels_)
    silhouette_scores_hv.append(score)
    
silhouette_scores_imp = []
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled_imp)
    score = silhouette_score(df_scaled_imp, kmeans.labels_)
    silhouette_scores_imp.append(score)

fig, axs = plt.subplots(1, 2, figsize=(12,5))

axs[0].plot(range_clusters, silhouette_scores_hv, marker='o')
axs[0].set_title('Silhouette Score - High Variance')
axs[0].set_xlabel('Number of Clusters')
axs[0].set_ylabel('Silhouette Score')
    
axs[1].plot(range_clusters, silhouette_scores_imp, marker='o')
axs[1].set_title('Silhouette Score - Importance')
axs[1].set_xlabel('Number of Clusters')
axs[1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show()
```


    
![png](README_files/README_163_0.png)
    


If a silhouette score is close to 1, that means a better cluster formation. The scores obtained with the first dataset are not very good, but the dataset made with the feature selection considering the **importance was the one with better results**, so it shall be the only one to be tested from now on. We could go with K=2, which would yield the best silhouette score.

Before making any conclusions, a good practice is to visualize cluster formation for different K values. For this, we can apply a technique called PCA (Principal Component Analysis), which reduces the dimensionality of our dataset while preserving the variance. Examples for 2 dimensions:


```python
df_scaled = df_scaled_imp.copy()
```


```python
from sklearn.decomposition import PCA

range_clusters = range(2,5)
for k in range_clusters:
    print(f'Number of clusters: {k}')
    
    # applying K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(df_scaled)

    # Perform PCA to reduce the data to 2 dimensions
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    # Plot the PCA-reduced data with cluster assignments
    plt.figure(figsize=(10, 7))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
    plt.title('2D PCA Visualization of Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()
```

    Number of clusters: 2
    


    
![png](README_files/README_167_1.png)
    


    Number of clusters: 3
    


    
![png](README_files/README_167_3.png)
    


    Number of clusters: 4
    


    
![png](README_files/README_167_5.png)
    


Considering the cluster formation, when K=2 there are undoubtedly two well-defined clusters. K=3 also seems reasonable, although there would be two clusters very close to each other. Next we'll try another algorithm:

### DBScan
DBSCAN is another clustering algorithm that groups together points that are closely packed together and marks points that are in low-density regions as outliers. DBSCAN has two main parameters: **eps**, which is the maximum distance between two points to be considered in the same neighborhood, and **min_samples**, which is the minimum number of points required to form a cluster.


```python
from sklearn.cluster import DBSCAN

# Testing for eps range between 0.1 and 0.4, stepping every 0.05:
eps_range = np.arange(0.1, 0.5, 0.05)
# Testing for min_sample between 5 and 55, stepping every 10:
min_samples_range = np.arange(5, 56, 10)

# Lists to store results
results = []

for min_sample in min_samples_range:
    for eps in eps_range:
        # Initialize DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_sample)

        # Fit DBSCAN to your scaled data
        dbscan.fit(df_scaled)

        # The cluster labels assigned by DBSCAN
        labels = dbscan.labels_

        # Number of clusters in labels (ignoring noise if present)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Append results to the list
        results.append({
            'min_sample': min_sample,
            'eps': eps,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        })

# Create a DataFrame from the results
df_results = pd.DataFrame(results)
```


```python
df_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min_sample</th>
      <th>eps</th>
      <th>n_clusters</th>
      <th>n_noise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>0.10</td>
      <td>21</td>
      <td>759</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>0.15</td>
      <td>10</td>
      <td>292</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0.20</td>
      <td>5</td>
      <td>136</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.25</td>
      <td>3</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.30</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.35</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5</td>
      <td>0.40</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5</td>
      <td>0.45</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15</td>
      <td>0.10</td>
      <td>3</td>
      <td>1212</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15</td>
      <td>0.15</td>
      <td>5</td>
      <td>670</td>
    </tr>
    <tr>
      <th>10</th>
      <td>15</td>
      <td>0.20</td>
      <td>3</td>
      <td>301</td>
    </tr>
    <tr>
      <th>11</th>
      <td>15</td>
      <td>0.25</td>
      <td>2</td>
      <td>157</td>
    </tr>
    <tr>
      <th>12</th>
      <td>15</td>
      <td>0.30</td>
      <td>2</td>
      <td>82</td>
    </tr>
    <tr>
      <th>13</th>
      <td>15</td>
      <td>0.35</td>
      <td>2</td>
      <td>21</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.40</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>0.45</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>25</td>
      <td>0.10</td>
      <td>1</td>
      <td>1346</td>
    </tr>
    <tr>
      <th>17</th>
      <td>25</td>
      <td>0.15</td>
      <td>2</td>
      <td>877</td>
    </tr>
    <tr>
      <th>18</th>
      <td>25</td>
      <td>0.20</td>
      <td>2</td>
      <td>512</td>
    </tr>
    <tr>
      <th>19</th>
      <td>25</td>
      <td>0.25</td>
      <td>2</td>
      <td>229</td>
    </tr>
    <tr>
      <th>20</th>
      <td>25</td>
      <td>0.30</td>
      <td>2</td>
      <td>112</td>
    </tr>
    <tr>
      <th>21</th>
      <td>25</td>
      <td>0.35</td>
      <td>2</td>
      <td>61</td>
    </tr>
    <tr>
      <th>22</th>
      <td>25</td>
      <td>0.40</td>
      <td>2</td>
      <td>13</td>
    </tr>
    <tr>
      <th>23</th>
      <td>25</td>
      <td>0.45</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>35</td>
      <td>0.10</td>
      <td>0</td>
      <td>1416</td>
    </tr>
    <tr>
      <th>25</th>
      <td>35</td>
      <td>0.15</td>
      <td>2</td>
      <td>1040</td>
    </tr>
    <tr>
      <th>26</th>
      <td>35</td>
      <td>0.20</td>
      <td>3</td>
      <td>626</td>
    </tr>
    <tr>
      <th>27</th>
      <td>35</td>
      <td>0.25</td>
      <td>2</td>
      <td>345</td>
    </tr>
    <tr>
      <th>28</th>
      <td>35</td>
      <td>0.30</td>
      <td>2</td>
      <td>146</td>
    </tr>
    <tr>
      <th>29</th>
      <td>35</td>
      <td>0.35</td>
      <td>2</td>
      <td>79</td>
    </tr>
    <tr>
      <th>30</th>
      <td>35</td>
      <td>0.40</td>
      <td>2</td>
      <td>43</td>
    </tr>
    <tr>
      <th>31</th>
      <td>35</td>
      <td>0.45</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>32</th>
      <td>45</td>
      <td>0.10</td>
      <td>0</td>
      <td>1416</td>
    </tr>
    <tr>
      <th>33</th>
      <td>45</td>
      <td>0.15</td>
      <td>1</td>
      <td>1145</td>
    </tr>
    <tr>
      <th>34</th>
      <td>45</td>
      <td>0.20</td>
      <td>2</td>
      <td>705</td>
    </tr>
    <tr>
      <th>35</th>
      <td>45</td>
      <td>0.25</td>
      <td>2</td>
      <td>418</td>
    </tr>
    <tr>
      <th>36</th>
      <td>45</td>
      <td>0.30</td>
      <td>2</td>
      <td>242</td>
    </tr>
    <tr>
      <th>37</th>
      <td>45</td>
      <td>0.35</td>
      <td>2</td>
      <td>101</td>
    </tr>
    <tr>
      <th>38</th>
      <td>45</td>
      <td>0.40</td>
      <td>2</td>
      <td>49</td>
    </tr>
    <tr>
      <th>39</th>
      <td>45</td>
      <td>0.45</td>
      <td>2</td>
      <td>19</td>
    </tr>
    <tr>
      <th>40</th>
      <td>55</td>
      <td>0.10</td>
      <td>0</td>
      <td>1416</td>
    </tr>
    <tr>
      <th>41</th>
      <td>55</td>
      <td>0.15</td>
      <td>1</td>
      <td>1214</td>
    </tr>
    <tr>
      <th>42</th>
      <td>55</td>
      <td>0.20</td>
      <td>2</td>
      <td>804</td>
    </tr>
    <tr>
      <th>43</th>
      <td>55</td>
      <td>0.25</td>
      <td>2</td>
      <td>496</td>
    </tr>
    <tr>
      <th>44</th>
      <td>55</td>
      <td>0.30</td>
      <td>2</td>
      <td>275</td>
    </tr>
    <tr>
      <th>45</th>
      <td>55</td>
      <td>0.35</td>
      <td>2</td>
      <td>126</td>
    </tr>
    <tr>
      <th>46</th>
      <td>55</td>
      <td>0.40</td>
      <td>2</td>
      <td>56</td>
    </tr>
    <tr>
      <th>47</th>
      <td>55</td>
      <td>0.45</td>
      <td>2</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>



In our scenario the eps is very senstive, which means the data is very grouped/dense. We should look for a scenario with a good amount of cluters and a moderate noise. If we take min_sample=55 and an eps of 0.45 for example, that gives us 2 well-defined clusters. Let's visualize them:


```python
# Initialize DBSCAN
dbscan = DBSCAN(eps=0.45, min_samples=55)

# Fit DBSCAN to your scaled data
dbscan.fit(df_scaled)

# The cluster labels assigned by DBSCAN
labels = dbscan.labels_

# Perform PCA to reduce the data to 2 dimensions - already done but good practice to remember
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Plotting the clusters
plt.figure(figsize=(10, 7))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.title('DBSCAN Clustering Visualization')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()
```


    
![png](README_files/README_173_0.png)
    



```python
# Calculating the silhouette score
sil_score = silhouette_score(df_scaled, labels)
print(f'Silhouette Score: {sil_score}')
```

    Silhouette Score: 0.4478192401116123
    

Given the results, both methods indicate that two clusters is the ideal scenario. Given the business problem at hand, we could also choose K=3 if the HR department wishes to have more than two groups to be analyzed. We choose to go with 3 clusters for now.

### Comparing other Scaling methods
Scaling can significantly affect clustering algorithms. Aside from MinMaxScaler, there are other two commonly used scaling methods: StandardScaler and RobustScaler. In order to see if there are big changes to our first found results, we create a pipeline in order to test each dataframe with different scalings with varying parameters from both algorithms:


```python
# Initialize each scaler

scaler = MinMaxScaler()
df_mm = pd.DataFrame(scaler.fit_transform(df_importance), columns=df_importance.columns)

scaler = StandardScaler()
df_ss = pd.DataFrame(scaler.fit_transform(df_importance), columns=df_importance.columns)

scaler = RobustScaler()
df_rs = pd.DataFrame(scaler.fit_transform(df_importance), columns=df_importance.columns)
```


```python
# Dataframe list
df_dict = {'MinMaxScaler': df_mm, 'StandardScaler': df_ss, 'RobustScaler': df_rs}
```


```python
def apply_kmeans(df, range_clusters):
    silhouette_scores = []
    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        score = silhouette_score(df, kmeans.labels_)
        
        silhouette_scores.append({
            'k': k,
            'sil_score': score
        })
        
    return pd.DataFrame(silhouette_scores)
```


```python
def apply_dbscan(df, eps_range, min_samples_range):
    silhouette_scores = []
    for min_sample in min_samples_range:
        min_sample = round(min_sample, 2)
        for eps in eps_range:
            eps = round(eps, 2)
            dbscan = DBSCAN(eps=eps, min_samples=min_sample)
            dbscan.fit(df)
            
            labels = dbscan.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if len(set(labels)) > 1:  # Silhouette score is not defined for 1 cluster
                score = silhouette_score(df, labels)
            else:
                score = 0
                
            silhouette_scores.append({
                'min_sample': min_sample,
                'eps': eps,
                'sil_score': score
            })
                
    df_scores = pd.DataFrame(silhouette_scores)
    pivot_df = df_scores.pivot(index='eps', columns='min_sample', values='sil_score')
    
    # Plotting the pivot table
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".3f")
    plt.title('Silhouette Score Heatmap')
    plt.xlabel('min_sample')
    plt.ylabel('eps')
    plt.show()
    
    return df_scores
```


```python
# Pipeline for testing clustering algorithms

for scaler, df in df_dict.items():
    print('='*100)
    print(f'Testing {scaler}:')
    
    # Using the same ranges from the previous tests:
    kmeans_range = range(2, 5)
    df_kmeans = apply_kmeans(df, kmeans_range)
    display(df_kmeans)
    
    eps_range = np.arange(0.1, 0.5, 0.05)
    min_samples_range = np.arange(5, 56, 10)
    df_dbscan = apply_dbscan(df, eps_range, min_samples_range)
```

    ====================================================================================================
    Testing MinMaxScaler:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>sil_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.480134</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.366526</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0.378701</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_181_2.png)
    


    ====================================================================================================
    Testing StandardScaler:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>sil_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.263922</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.283693</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0.284055</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_181_5.png)
    


    ====================================================================================================
    Testing RobustScaler:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>sil_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.349683</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.260509</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0.249667</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_181_8.png)
    


From our tests it's clear that the MinMaxScaler was the transformation with better results. Now let's apply the final results into our dataset:


```python
# load previously saved dataset, before transformations
df = df_eda.copy()

# applying K-Means with the final parameters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(df_scaled)
df['GroupLabel'] = kmeans_labels
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobRole</th>
      <th>JobSatisfaction</th>
      <th>MaritalStatus</th>
      <th>MonthlyIncome</th>
      <th>SalarySlab</th>
      <th>MonthlyRate</th>
      <th>NumCompaniesWorked</th>
      <th>OverTime</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
      <th>AgeGroup</th>
      <th>DistanceGroup</th>
      <th>CompensationGroup</th>
      <th>YSLPGroup</th>
      <th>GroupLabel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>230</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>3</td>
      <td>Male</td>
      <td>54</td>
      <td>3</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>3</td>
      <td>Single</td>
      <td>1420</td>
      <td>Upto 5k</td>
      <td>25233</td>
      <td>1</td>
      <td>No</td>
      <td>13</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>18-29</td>
      <td>Near</td>
      <td>Low</td>
      <td>0-3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>812</td>
      <td>Sales</td>
      <td>10</td>
      <td>3</td>
      <td>Medical</td>
      <td>4</td>
      <td>Female</td>
      <td>69</td>
      <td>2</td>
      <td>1</td>
      <td>Sales Representative</td>
      <td>3</td>
      <td>Single</td>
      <td>1200</td>
      <td>Upto 5k</td>
      <td>9724</td>
      <td>1</td>
      <td>No</td>
      <td>12</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>18-29</td>
      <td>Far</td>
      <td>Low</td>
      <td>0-3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>Yes</td>
      <td>Travel_Frequently</td>
      <td>1306</td>
      <td>Sales</td>
      <td>5</td>
      <td>3</td>
      <td>Marketing</td>
      <td>2</td>
      <td>Male</td>
      <td>69</td>
      <td>3</td>
      <td>1</td>
      <td>Sales Representative</td>
      <td>2</td>
      <td>Single</td>
      <td>1878</td>
      <td>Upto 5k</td>
      <td>8059</td>
      <td>1</td>
      <td>Yes</td>
      <td>14</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>18-29</td>
      <td>Near</td>
      <td>Low</td>
      <td>0-3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>No</td>
      <td>Non-Travel</td>
      <td>287</td>
      <td>Research &amp; Development</td>
      <td>5</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>2</td>
      <td>Male</td>
      <td>73</td>
      <td>3</td>
      <td>1</td>
      <td>Research Scientist</td>
      <td>4</td>
      <td>Single</td>
      <td>1051</td>
      <td>Upto 5k</td>
      <td>13493</td>
      <td>1</td>
      <td>No</td>
      <td>15</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>18-29</td>
      <td>Near</td>
      <td>Low</td>
      <td>0-3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>Yes</td>
      <td>Non-Travel</td>
      <td>247</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Medical</td>
      <td>3</td>
      <td>Male</td>
      <td>80</td>
      <td>3</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>3</td>
      <td>Single</td>
      <td>1904</td>
      <td>Upto 5k</td>
      <td>13556</td>
      <td>1</td>
      <td>No</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>18-29</td>
      <td>Moderate</td>
      <td>Low</td>
      <td>0-3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Results Analysis<a class="anchor" id="ninth-bullet"></a>

Now we should analyze the newly obtained cluster labels, in order to determine **which characteristics they have in common**. This will help the HR Department to implement more precise initiatives and strategies to each group.

Instead of performing this exploration here, we will load the dataset with the clusters labels into a **PowerBI Report**, which will be the final deliverable product to the HR team, along with the insights and their recommendations.


```python
# saving the dataset to a .csv, to be loaded into PowerBI
df.to_csv(filepath+'\HR_Analytics_Final.csv', index = False, encoding='utf-8')
```

Link to the PowerBI Report:
[PowerBI Report](https://app.powerbi.com/groups/9589d4f1-2310-40a5-bce6-a9536ee6967c/reports/4ca4c50d-04b3-47af-b8aa-86c7e9bd37eb/ReportSection?ctid=85b28421-d45a-4b07-889d-24b528c7f250&experience=power-bi)

*(if the link doesn't work due to corporative sharing restrictions, I've included some prints of the most important parts below)*

## Understanding the PowerBI Report
For this report, I've creted an Overview tab, with general analysis and comparisons of all three groups, and general but very useful observations.

Then I created one tab for each group, containing the same views/charts as a way to understand and compare the particularities of each one in more detail:

<img src="images/Overview1.png" alt="Overview1" />

<img src="images/Overview2.png" alt="Overview2" />

Regarding the Overview tab, the most useful insights we can take from it is that **Overtime is an exclusive feature from Group1**, and they should be receive more attention on this matter.

Group 0 is the one with more members, and they all share the same average age, job level and income (which means those features were not the most representative ones to differentiate the clusters).

Another difference between them is the Average Distance From Home, although this by itself does not say anything about the Attrition.

I've added a descriptive box with some text when the icon next to the leftmost graphs are clicked, in which I included observations about the compensation within Group1 (the one with Overtime): those with smaller compensations show **significantly higher Attrition Rates!**

<img src="images/Group0.png" alt="Group0" />

<img src="images/Group1.png" alt="Group1" />

<img src="images/Group2.png" alt="Group2" />

Regarding the Group-speficic tabs, aside from understanding the demographics from each group, it's expected from the HR Team to try pointing out **which employees show lower levels of satisfaction, involvement and work-life balance**. For example, women between 35-38 years old from Group0 show lower levels of 'Relationship Satisfaction', in comparison to all the other averages. This may be worth investigating in more depth within the company routine.


The 'Department' filter on the top of the page might be a useful tool in order to refine this kind of analysis.

Another very noticeable insight is that the Attrition Rate is higher for Lower compensations, in all Groups.

## Conclusions

In this project, we were given the challenge to better understand the factors influencing attrition, based on employee data. The main objective was to initially explore the data, uncovering patters, implement a clustering strategy if applicable, and develop actionable insights in order to reduce the turnover.

The main challenged that were faced included: ensuring the data was clean, understanding and selecting the most relevant features, and intepreting the results and their relation to Attrition to drive meaningful conclusions. The Exploratory Data Analysis step presented itself as crucial in indentifying correlations between factors like age, income and work-life balance, and how they might affect turnover, which are real and challenging factors to be overcome in corporative worlds.

Paired with the initial hypothesis, several key insights were discovered through our analysis, which proved themselves statistically relevant and could potentially proedict or explain the attrition. Additionally, a clustering algorithm was succesfully implemented in order to group up the employees with similar attributes, allowing us to understand and target specific groups with tailored initiatives.

PowerBI is always an excellent tool for visualizing the data and presenting insights to stakeholders, making it easier to communicate the results clearly to the company’s leadership. It's versatility allows for building dashboards at various formats, with dynamic filtering, slicers and even distributing it by e-mail or in slideshow presentation format.

Finally, let's recap our recommendations to the HR Department, based on the findings:

- Provide mentorship programs and other engagement-focused activities to increase younger employees' connection to the company, as well as clear career progression paths and opportunities for promotions


- Closely monitor satisfaction levels related to the 'distance to work' metrics


- Consider implementing retention initiatives or improving compensation packages for employees with low compensations


- Regularly measure job involvement and improve factors such as recognition, growth and development in order to keep it ah high levels


- Provide custom specialization courses based on the employees' educational backgrounds, and give them opportunities to interact and familiarize with other areas and how each contribute to the big picture


- Alleviate overtime whenever possible, considering it's the main factor impacting the Attrition rates. Flexible working hours, hybrid work regime and providing more quality of life and comfort in the office are good measures in order to mitigate overtime impact


- And last but not least, collect feedback from the employees! Applying interactive surveys or games are also fun ways to especially target younger audiences and engage them in sharing their sentiment and ideas for the company.

Final observations:

Thinking ahead, possible improvements on this project might include predictive modeling for attrition, especially for new employees, improving feature selection (having a feedback from employees in the future could potentially be one of the main factors to measure insatisfaction), and even integrating real-time analytics to monitor employee sentiment. This would further refine the insights and allow for proactive HR strategies.

---

[Back to the top](#top-bullet)
