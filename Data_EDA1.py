2 + 2 # Function F9
# Works as calculator

# Python Libraries (Packages)
# pip install <package name> - To install library (package), execute the code in Command prompt
# pip install pandas

import pandas as pd

dir(pd)

# Read data into Python
education = pd.read_csv(r"C:\Users\DELL\Desktop\Github\ML-Training\education.csv")


A = 10
a = 10.1

education.info()

# C:\Users\education.csv - this is windows default file path with a '\'
# C:\\Users\\education.csv - change it to '\\' to make it work in Python

# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
education.workex.mean() # '.' is used to refer to the variables within object
education.workex.median()
education.workex.mode()

# Measures of Dispersion / Second moment business decision
education.workex.var() # variance
education.workex.std() # standard deviation
range = max(education.workex) - min(education.workex) # range
range

# Third moment business decision
education.workex.skew()
education.gmat.skew()

# Fourth moment business decision
education.workex.kurt()

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

education.shape

# barplot
plt.bar(height = education.gmat, x = np.arange(1, 774, 1)) # initializing the parameter

# Histogram
plt.hist(education.gmat) # histogram
plt.hist(education.gmat, bins = [600, 680, 710, 740, 780], color = 'green', edgecolor="red") 
plt.hist(education.workex)
plt.hist(education.workex, color='red', edgecolor = "black", bins = 6)

help(plt.hist)

# Histogram using Seaborn
import seaborn as sns
sns.distplot(education.gmat) # Deprecated histogram function from seaborn

sns.displot(education.gmat) # Histogram from seaborn


# Boxplot
plt.figure()
plt.boxplot(education.gmat) # boxplot

help(plt.boxplot)


# Density Plot
sns.kdeplot(education.gmat) # Density plot
sns.kdeplot(education.gmat , bw = 0.5 , fill = True)


# Descriptive Statistics
# describe function will return descriptive statistics including the central tendency, dispersion and shape of a dataset's distribution

education.describe()


# Bivariate visualization
# Scatter plot
import pandas as pd
import matplotlib.pyplot as plt

cars = pd.read_csv("C:/Data/Cars.csv")

cars.info()

plt.scatter(x = cars['HP'], y = cars['MPG']) 

plt.scatter(x = cars['HP'], y = cars['SP'], color = 'green') 

