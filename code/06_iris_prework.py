
'''
SOLUTIONS: "Human Learning" with iris data
'''

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the famous iris data
iris = load_iris()

# what do you think these attributes represent?
iris.data
iris.data.shape
iris.feature_names
iris.target
iris.target_names

# intro to numpy
type(iris.data)


## PART 1: Read data into pandas and explore

features = [name[:-5].replace(' ', '_') for name in iris.feature_names]
iris_df = pd.DataFrame(iris.data, columns=features)


# the feature_names are a bit messy, let's clean them up. remove the (cm)
# at the end and replace any spaces with an underscore
# create a list "features" that holds the cleaned column names


# read the iris data into pandas, with our refined column names

# create a list of species (should be 150 elements) 
# using iris.target and iris.target_names
# resulting list should only have the words "setosa", "versicolor", and "virginica"

species = [iris.target_names[num] for num in iris.target]


# add the species list as a new DataFrame column

iris_df['species'] = species

# explore data numerically, looking for differences between species
# try grouping by species and check out the different predictors
iris_df.describe

# explore data by sorting, looking for differences between species

iris_df.speci.value_counts().plot(kind='hist', title='Number of Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()      

iris_df.petal_length.hist(bins=20)
plt.xlabel('Petal Length')
plt.ylabel('Frequency')  

iris_df.groupby('species').petal_width.mean()
iris_df.plot(kind='scatter', x='petal_width', y='petal_length', alpha=0.3)

                  
# I used values in order to see all of the data at once
# without .values, a dataframe is returned


# explore data visually, looking for differences between species
# try using a histogram or boxplot





## PART 2: Write a function to predict the species for each observation

# create a dictionary so we can reference columns by name



# define function that takes in a row of data and returns a predicted species


# make predictions and store as numpy array

# calculate the accuracy of the predictions
