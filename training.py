# Import Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib

def train_func(df):
    
    # Create a new dataframe where Woods and AR are labeled as Group 1 (G1), Coals and MB are labeled as Group 2 (G2)
    df_g1_g2 = df.replace(['Woods','Agricultural Residue','Manufactured Biomass'   
               ,'Coals'], ['G1','G1','G2','G2']);
    
    
    # Create a new dataframe where only includes Coals and MB
    df_c_mb = df.loc[  (df.iloc[:,-1] == 'Coals')
                     | (df.iloc[:,-1] == 'Manufactured Biomass')
                     ]
    
    # Create a new dataframe where only includes Woods and AR
    df_w_ar = df.loc[  (df.iloc[:,-1] == 'Woods')
                     | (df.iloc[:,-1] == 'Agricultural Residue')
                     ]
    
    # Define a SVM Classifier with C = 0.01, Kernel = 2nd degree polynomial
    clf_g1_g2 = SVC(C = 0.01,kernel = 'poly', degree = 2, random_state = 0, gamma = 'auto')
    
    # Train SVM Classifier to clasify G1 and G2 using previously defined dataframe (df_g1_g2)
    clf_g1_g2.fit(df_g1_g2.iloc[:,:-1],df_g1_g2.iloc[:,-1])
    
    
    # Define a KNN Classifier with 3 neighbors and 2 minkowski order
    clf_c_mb = KNeighborsClassifier(n_neighbors= 3, p = 2)
    
    # Train KNN Classifier to clasify Coals and MB using previously defined dataframe (df_c_mb)
    clf_c_mb.fit(df_c_mb.iloc[:,:-1],df_c_mb.iloc[:,-1])
    
    
     # Define a RF Classifier with 100 number of trees and 8 maximum tree depth for each
    clf_w_ar = RandomForestClassifier(n_estimators = 100,
                                                max_depth=8)
    
    # Train RF Classifier to clasify Woods and AR using previously defined dataframe (df_w_ar)
    clf_w_ar.fit(df_w_ar.iloc[:,:-1],df_w_ar.iloc[:,-1])
    
    # Deploy classifiers
    joblib.dump(clf_g1_g2, 'clf_g1_g2.pkl')
    joblib.dump(clf_w_ar, 'clf_w_ar.pkl')
    joblib.dump(clf_c_mb, 'clf_c_mb.pkl')


