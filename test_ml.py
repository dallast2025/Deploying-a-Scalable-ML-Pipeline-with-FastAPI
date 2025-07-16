import pytest
import numpy as np
from ml.model import train_model
from sklearn.linear_model import LogisticRegression

# TODO: add necessary import

X_train =np.array([[1,2],[2,3],[3,4]])
y_train = np.array([0,1,0])

# TODO: implement the first test. Change the function name and input as needed
def test_one_trainmodel():
    """
    This test is to check if Logistic Regression model is used.
    """
    # Your code here
    model=train_model(X_train, y_train)

    assert isinstance(model, LogisticRegression), f"{type(model)} is invalid. Please enter LogisticRegression Model."



# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
