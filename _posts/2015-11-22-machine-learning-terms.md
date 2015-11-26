---
layout:     post
title:      "Machine Learning Terms"
subtitle:   "The Foundation"
---

We denote the input features as x<sub>0</sub>,
x<sub>1</sub>,.., x<sub>n</sub>  
The weights of the input features are denoted as w<sub>0</sub>,
w<sub>1</sub>,.., w<sub>n</sub>  
We denote the output as y

The weight of a feature is the role it plays in determining the
value of the output y.

*Error Rate*: It is the difference between the actual value and the
predicted value by our model.

*Overfitting*: It is when the model is too customized for our
training data i.e the model has a low error rate while predicting the
outputs for our training data but has a high error rate while making
predictions for new data or test_data.  
We use 'training_data' set for training our model and a
different data set called 'test_data' to check if our model is
overfitting.

*Underfitting*: It occurs when our model is not able to fit the data
well enough i.e the model has a high error rate even while
predicting the output for our training data.
