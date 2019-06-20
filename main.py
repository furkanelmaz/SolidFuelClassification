# Import necessary packages
import numpy as np
import pandas as pd
from sklearn.externals import joblib

# Import training and testing functions in train.py and test.py scripts
from training import train_func
from testing import test_func


####### Training Procedure ####

#df = pd.read_excel('Dataset.xlsx') #Import dataset
#df.drop(['Unnamed: 0','Original Material','Reference'], axis = 1, inplace = True) #Drop all columns except features and output

## Train and save models as pkl files
#train_func(df)
#
########


###### Testing Procedure #######

# Enter proximate results of the material
fixed_carbon = 84.59
volatile_matter = 7.09
ash = 8.32

# Proximate results are converted to pandas series to convenience with test function
proximate_results = pd.Series([fixed_carbon,volatile_matter,ash])

# Import trained classifiers
clf_g1_g2 = joblib.load('clf_g1_g2.pkl')
clf_w_ar = joblib.load('clf_w_ar.pkl')
clf_c_mb = joblib.load('clf_c_mb.pkl')

# Send proximate analysis results to hiearchical classifier and print the predicted class
print(test_func(proximate_results,clf_g1_g2,clf_c_mb,clf_w_ar)[0])