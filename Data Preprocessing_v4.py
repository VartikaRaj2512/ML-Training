############# Data Pre-processing ##############

################ Type casting #################
import pandas as pd

data = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\ethnic diversity.csv")
data.dtypes

'''
EmpID is Integer - Python automatically identify the data types by interpreting the values. 
As the data for EmpID is numeric Python detects the values as int64.

From measurement levels prespective the EmpID is a Nominal data as it is an identity for each employee.

If we have to alter the data type which is defined by Python then we can use astype() function

'''

help(data.astype)

# Convert 'int64' to 'str' (string) type. 
data.EmpID = data.EmpID.astype('str')
data.dtypes

data.Zip = data.Zip.astype('str')
data.dtypes

# For practice:
# Convert data types of columns from:
    
# 'float64' into 'int64' type. 
data.Salaries = data.Salaries.astype('int64')
data.dtypes

# int to float
data.age = data.age.astype('float32')
data.dtypes


##############################################
### Identify duplicate records in the data ###
import pandas as pd
data = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\mtcars_dup.csv")

# Duplicates in rows
help(data.duplicated)

duplicate = data.duplicated()  # Returns Boolean Series denoting duplicate rows.
duplicate

sum(duplicate)

# Parameters
duplicate = data.duplicated(keep = 'last')
duplicate

duplicate = data.duplicated(keep = False)
duplicate


# Removing Duplicates
data1 = data.drop_duplicates() # Returns DataFrame with duplicate rows removed.

# Parameters
data1 = data.drop_duplicates(keep = 'last')

data1 = data.drop_duplicates(keep = False)


# Duplicates in Columns
# We can use correlation coefficient values to identify columns which have duplicate information

import pandas as pd

cars = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\Cars.csv")

# Correlation coefficient
'''
Ranges from -1 to +1. 
Rule of thumb says |r| > 0.85 is a strong relation
'''
cars.corr()

'''We can observe that the correlation value for HP and SP is 0.973 and VOL and WT is 0.999 
& hence we can ignore one of the variables in these pairs.
'''

################################################
############## Outlier Treatment ###############
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\ethnic diversity.csv")
df.dtypes

# Let's find outliers in Salaries
sns.boxplot(df.Salaries)

sns.boxplot(df.age)
# No outliers in age column

# Detection of outliers (find limits for salary based on IQR)
IQR = df['Salaries'].quantile(0.75) - df['Salaries'].quantile(0.25)

lower_limit = df['Salaries'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Salaries'].quantile(0.75) + (IQR * 1.5)

############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# Let's flag the outliers in the dataset

outliers_df = np.where(df.Salaries > upper_limit, True, np.where(df.Salaries < lower_limit, True, False))
df_trimmed = df.loc[~(outliers_df), ]
df.shape, df_trimmed.shape

# Let's explore outliers in the trimmed dataset
sns.boxplot(df_trimmed.Salaries)

############### 2. Replace ###############
# Replace the outliers by the maximum and minimum limit
df['df_replaced'] = pd.DataFrame(np.where(df['Salaries'] > upper_limit, upper_limit, np.where(df['Salaries'] < lower_limit, lower_limit, df['Salaries'])))
sns.boxplot(df.df_replaced)


############### 3. Winsorization ###############
pip install feature_engine   install the package
from feature_engine.outliers import Winsorizer

# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['Salaries'])

df_s = winsor_iqr.fit_transform(df[['Salaries']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(df_s.Salaries)


# Define the model with Gaussian method
winsor_gaussian = Winsorizer(capping_method = 'gaussian', 
                             # choose IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 3,
                          variables = ['Salaries'])

df_t = winsor_gaussian.fit_transform(df[['Salaries']])
sns.boxplot(df_t.Salaries)


# Define the model with percentiles:
# Default values
# Right tail: 95th percentile
# Left tail: 5th percentile

winsor_percentile = Winsorizer(capping_method = 'quantiles',
                          tail = 'both', # cap left, right or both tails 
                          fold = 0.05, # limits will be the 5th and 95th percentiles
                          variables = ['Salaries'])

df_p = winsor_percentile.fit_transform(df[['Salaries']])
sns.boxplot(df_p.Salaries)


##############################################
#### zero variance and near zero variance ####

# If the variance is low or close to zero, then a feature is approximately constant and will not improve the performance of the model.
# In that case, it should be removed. 

df.var() # variance of numeric variables
df.var() == 0
df.var(axis = 0) == 0


#############
# Discretization

import pandas as pd
data = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\ethnic diversity.csv")
data.head()

data.info()

data.describe()

# Binarization
data['Salaries_new'] = pd.cut(data['Salaries'], 
                              bins = [min(data.Salaries), data.Salaries.mean(), max(data.Salaries)],
                              labels = ["Low", "High"])

# Look out for the break up of the categories.
data.Salaries_new.value_counts()


''' We can observe that the total number of values are 309. This is because one of the value has become NA.
This happens as the cut function by default does not consider the lowest (min) value while discretizing the values.
To over come this issue we can use the parameter 'include_lowest' set to True.
'''

data['Salaries_new1'] = pd.cut(data['Salaries'], 
                              bins = [min(data.Salaries), data.Salaries.mean(), max(data.Salaries)], 
                              include_lowest = True,
                              labels = ["Low", "High"])

data.Salaries_new1.value_counts()


# Discretization / Multiple bins
data['Salaries_multi'] = pd.cut(data['Salaries'], 
                              bins = [min(data.Salaries), 
                                      data.Salaries.quantile(0.25),
                                      data.Salaries.mean(),
                                      data.Salaries.quantile(0.75),
                                      max(data.Salaries)], 
                              include_lowest = True,
                              labels = ["P1", "P2", "P3", "P4"])

data.Salaries_multi.value_counts()


data.MaritalDesc.value_counts()

##################################################
################## Dummy Variables ###############
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

# Use the ethinc diversity dataset
df = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\ethnic diversity.csv")

df.columns # column names
df.shape # will give u shape of the dataframe

df.dtypes
df.info()

# Drop emp_name column
df.drop(['Employee_Name', 'EmpID', 'Zip'], axis = 1, inplace = True)


# Create dummy variables
df_new = pd.get_dummies(df)

df_new_1 = pd.get_dummies(df, drop_first = True)
# Created dummies for all categorical columns

##### One Hot Encoding works
df.columns
df = df[['Salaries', 'age', 'Position', 'State', 'Sex',
         'MaritalDesc', 'CitizenDesc', 'EmploymentStatus', 'Department', 'Race']]

from sklearn.preprocessing import OneHotEncoder
# Creating instance of One-Hot Encoder
enc = OneHotEncoder() # initializing method

enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:, 2:]).toarray())


#######################
# Label Encoder
from sklearn.preprocessing import LabelEncoder

# Creating instance of labelencoder
labelencoder = LabelEncoder()

# Data Split into Input and Output variables
X = df.iloc[:, :9]
y = df.iloc[:, 9]

X['Sex'] = labelencoder.fit_transform(X['Sex'])
X['MaritalDesc'] = labelencoder.fit_transform(X['MaritalDesc'])
X['CitizenDesc'] = labelencoder.fit_transform(X['CitizenDesc'])



#################### Missing Values - Imputation ###########################
import numpy as np
import pandas as pd

# Load modified ethnic dataset
df = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\ethnic diversity.csv") # for doing modifications

# Check for count of NA's in each column
df.isna().sum()

# Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data (Salaries)
# Mode is used for discrete data (ex: Position, Sex, MaritalDesc)

# For Mean, Median, Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

# Mean Imputer 
mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
df["Salaries"] = pd.DataFrame(mean_imputer.fit_transform(df[["Salaries"]]))
df["Salaries"].isna().sum()

# Median Imputer
median_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
df["age"] = pd.DataFrame(median_imputer.fit_transform(df[["age"]]))
df["age"].isna().sum()  # all records replaced by median 

df.isna().sum()

# Mode Imputer
mode_imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
df["Sex"] = pd.DataFrame(mode_imputer.fit_transform(df[["Sex"]]))
df["MaritalDesc"] = pd.DataFrame(mode_imputer.fit_transform(df[["MaritalDesc"]]))
df.isnull().sum()  # all Sex, MaritalDesc records replaced by mode

# Constant Value Imputer
constant_imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 'F')
# fill_value can be used for numeric or non-numeric values

df["Sex"] = pd.DataFrame(constant_imputer.fit_transform(df[["Sex"]]))

# Random Imputer
from feature_engine.imputation import RandomSampleImputer

random_imputer = RandomSampleImputer(['age'])
df["age"] = pd.DataFrame(random_imputer.fit_transform(df[["age"]]))
df["age"].isna().sum()  # all records replaced by median

#####################
# Normal Quantile-Quantile Plot

import pandas as pd

# Read data into Python
education = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\education.csv")

import scipy.stats as stats
import pylab

# Checking whether data is normally distributed
stats.probplot(education.gmat, dist = "norm", plot = pylab)

stats.probplot(education.workex, dist = "norm", plot = pylab)

import numpy as np

# Transformation to make workex variable normal
stats.probplot(np.log(education.workex), dist = "norm", plot = pylab)

# Import modules
import pandas as pd
from scipy import stats

# Plotting related modules
import seaborn as sns
import matplotlib.pyplot as plt
import pylab

# Read data into Python
education = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\education.csv")

# Original data
prob = stats.probplot(education.workex, dist = stats.norm, plot = pylab)

# Transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(education.workex)

# creating axes to draw plots
fig, ax = plt.subplots(1, 2)

# Plotting the original data (non-normal) and fitted data (normal)
sns.distplot(education.workex, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 2},
             label = "Non-Normal", color = "green", ax = ax[0])

sns.distplot(fitted_data, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 2},
             label = "Normal", color = "green", ax = ax[1])

# adding legends to the subplots
plt.legend(loc = "upper right")

# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)

print(f"Lambda value used for Transformation: {fitted_lambda}")

# Transformed data
prob = stats.probplot(fitted_data, dist = stats.norm, plot = pylab)

# Yeo-Johnson Transform

'''
We can apply it to our dataset without scaling the data.
It supports zero values and negative values. It does not require the values for 
each input variable to be strictly positive. 

In Box-Cox transform the input variable has to be positive.
'''

# import modules
import pandas as pd
from scipy import stats

# Plotting modules
import seaborn as sns
import matplotlib.pyplot as plt
import pylab

# Read data into Python
education = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\education.csv")

# Original data
prob = stats.probplot(education.workex, dist = stats.norm, plot = pylab)

from feature_engine import transformation

# Set up the variable transformer
tf = transformation.YeoJohnsonTransformer(variables = 'workex')

edu_tf = tf.fit_transform(education)

# Transformed data
prob = stats.probplot(edu_tf.workex, dist = stats.norm, plot = pylab)

####################################################
######## Standardization and Normalization #########
import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\mtcars.csv")

a = data.describe()

### Standardization
from sklearn.preprocessing import StandardScaler

# Initialise the Scaler
scaler = StandardScaler()

# To scale data
df = scaler.fit_transform(data)
# Convert the array back to a dataframe
dataset = pd.DataFrame(df)
res = dataset.describe()


### Normalization
## load dataset
ethnic1 = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\ethnic diversity.csv")
ethnic1.columns
ethnic1.drop(['Employee_Name', 'EmpID', 'Zip'], axis = 1, inplace = True)

a1 = ethnic1.describe()

# Get dummies
ethnic = pd.get_dummies(ethnic1, drop_first = True)

a2 = ethnic.describe()

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(ethnic)
b = df_norm.describe()

''' Alternatively we can use the below function'''
from sklearn.preprocessing import MinMaxScaler
minmaxscale = MinMaxScaler()

ethnic_minmax = minmaxscale.fit_transform(ethnic)
df_ethnic = pd.DataFrame(ethnic_minmax)
minmax_res = df_ethnic.describe()

'''Robust Scaling
Scale features using statistics that are robust to outliers'''

from sklearn.preprocessing import RobustScaler

robust_model = RobustScaler()

df_robust = robust_model.fit_transform(ethnic)

dataset_robust = pd.DataFrame(df_robust)
res_robust = dataset_robust.describe()
