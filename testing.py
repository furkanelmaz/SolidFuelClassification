def test_func(testdata,clf_g1_g2,clf_c_mb,clf_w_ar):
    
    # Initialize prediction list
    prediction_list = list()
         
    # Firstly, SVM Classifier (clf_g1_g2) classifies the data as G1 or G2
    pred_g1_g2 = clf_g1_g2.predict(testdata.values.reshape(1,-1))
    
    
    # If the prediction is G1, RF Classifier (clf_w_ar) classifies the data as rather Woods or AR
    # and this final prediction is added to the prediction list
    if pred_g1_g2 == 'G1':
                 
        prediction_list.append(clf_w_ar.predict(testdata.values.reshape(1,-1))[0])
    
    # If the prediction is G2, KNN Classifier (clf_c_mb) classifies the data as rather Coals or MB
    # and this final prediction is added to the prediction list
    elif pred_g1_g2 == 'G2':
        
        prediction_list.append(clf_c_mb.predict(testdata.values.reshape(1,-1))[0])
    
    # Return list of predictions
    return prediction_list