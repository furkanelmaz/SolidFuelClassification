# Hierarchical Classification Framework for Solid Fuel Classification

This repository serves as a supplementary material for the following publication:

**Elmaz, Furkan, et al. "Classification of solid fuels with machine learning." Fuel 266 (2020): 117066.  
DOI: [10.1016/j.fuel.2020.117066](https://doi.org/10.1016/j.fuel.2020.117066)**  
  
Training and testing routines of the proposed hierarchical classifier for solid fuel classification task are illustrated below. Utilize main.py to use the model.  
  

## Training Phase

![Training Procedure of the Model](https://github.com/furkanelmaz/SolidFuelClassification/blob/master/images/Train.png?raw=true)




## Test Phase

![alt text](https://github.com/furkanelmaz/SolidFuelClassification/blob/master/images/Test.png?raw=true)


## Notes:  
  
1) In main.py, default is to use pre-trained models (.pkl files). Thus, training procedure is commented out. If models are wanted to be trained all over again. One can uncomment these lines and comment 'import trained classifiers' part at the end of the file
