import pytest
import numpy as np
from ml.model import train_model, compute_model_metrics
from sklearn.linear_model import LogisticRegression

# TODO: add necessary import

X_train =np.array([[1,2],[2,3],[3,4]]) #Train Features Example
y_train = np.array([0,1,0]) #Example labels for training
X_test = np.array ([[4,5],[5,6]]) # Test features examples.
y_test = np.array([1,0])

# TODO: implement the first test. Change the function name and input as needed
def test_one_trainmodel():
    """
    This test is to check if Logistic Regression model is used.
    """
    # Your code here
    model=train_model(X_train, y_train)

    assert isinstance(model, LogisticRegression), f"{type(model)} is invalid. Please enter LogisticRegression Model."



# TODO: implement the second test. Change the function name and input as needed
def test_two_compute_model_metrics_float():
    """
    This test is to check if the compute_model_metrics is returned as a valid float type for precision, recall, and F1 scores.
    """
    # Your code here
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    assert isinstance(precision,float), "Invalid float type please make sure precision is a float"
    assert isinstance(recall,float), "Invalid float type please make sure recall is a float"
    assert isinstance(fbeta,float), "Invalid float type please make sure F1 scores is a float"




# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
