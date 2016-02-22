---
layout:     post
title:      "Linear Regression in 1 Variable"
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
