---
layout:     post
title:      "Linear Regression"
subtitle:   "The Fundamentals"
---

There are two types of Supervised Learning:  
1. Regression:  
We try to predict a continuous value, like 1, 2.3, 5.45, etc.  
2. Classification:  
We try to predict if an email is spam or not, does a patient have cancer
or not.  


Consider a dataset for predicting house prices:  
Since we have to predict price of the house which is a continuous
value, we user regression.

| Size in feet | Price in $ |
| --- | --- |
| 2104 | 460 |
| 1416 | 232 |
| 1534 | 315 |
| 852 | 178 |
{: .table.table-bordered }

Notation:  
'm': number of features ( 1 )  
'x': input variables or features ( Size of house )  
'y': output variable or target ( Price in $ )  
'n': number of training examples ( 4 )  

The basic structure of a Machine Learning Algorithm is:
![Machine Learning Algorithm
Flow](../../img/machine_learning_algorithm.png)

1. The algorithm takes a set of training data and outputs a hypothesis.
2. We can then use the hypothesis to predict the output for new data.

## Things to remember while using Linear Regression

#### Linear Regression Using One Variable (univariate linear regression):

You should use linear regression when there is a correlation between the
features/independent variables and the target/dependent variables.

You can find the correlation by using the numpy library in python:

{% highlight python %}
correlation = np.correlate(feature, target)
if abs(correlation) > 2/(math.sqrt(len(feature)):
    print("correlated")
else:
    print("not correlated")
{% endhighlight %}  

#### Linear Regression using multiple variables(Multivalent Linear Regression):

Points to remember:

1. In a multivalent linear regression the features should be correlated
to the target but the features should not be correlated with each other.
If the features are correlated with each other it will be tough to
determine which feature is effecting the variation in the target
variable. When two or more features are correlated it is known as
multicollinearity.

2. We should not blindly add any number of features as they may add unnecessary noise and cause overfitting of the data.

Steps to follow:

1. First check the correlation between the target variable and the features. Ignore the features that are not correlated to the target
variable.

2. Check  the correlation between the features. If two features are correlated ignore the feature that has lower correlation with the
target variable.
