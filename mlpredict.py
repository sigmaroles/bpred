import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

if __name__=='__main__':
    fname = sys.argv[1]
    data = []
    labels = []
    with open(fname, 'r', 16777216) as fh:
        for row in fh:
            toks = row.split(' ')
            tnt = 0 if toks[1].strip()=='N' else 1
            data.append(list(map(lambda x: int(x), bin(int(toks[0], 16))[2:])))
            labels.append(tnt)
    data = np.array(data)
    labels = np.array(labels)
    
    print ("total data shape: ", data.shape, "total targets shape: ", labels.shape)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2)
    print ("train data shape:", X_train.shape, "train labels shape : ",y_train.shape)
    print ("test data shape:", X_test.shape, "test labels shape : ",y_test.shape)
    
    #create and train classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confMatrix = confusion_matrix(y_test, y_pred)
    print ("\nConfusion matrix : ")
    print (confMatrix)
    print ("\nAccuracy : ", accuracy_score(y_test, y_pred))
    print ("\nClassification report : \n")
    print (classification_report(y_test, y_pred))