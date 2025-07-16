import pytest
from ml.model import train_model
from sklearn.linear_model import LogisticRegression

# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_one():
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
