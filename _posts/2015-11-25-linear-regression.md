---
layout:     post
title:      "Linear Regression"
subtitle:   "The first step to evolution"
---

Trying to predict weight from age. Linear Regression is used to predict
continuous values.

{% highlight python %}
# Import the linear regression class
from sklearn.linear_model import LinearRegression

# Initialize the linear regression class.
regressor = LinearRegression()# Import the linear regression class
from sklearn.linear_model import LinearRegression

# Initialize the linear regression class.
regressor = LinearRegression()

# We pass in a list when we select predictor columns from "training_data"
# to force pandas not to generate a series.
predictors = training_data[['age']]
to_predict = training_data['weight']

# Train the linear regression model on our dataset.
regressor.fit(predictors, to_predict)

# Make predictions for the test data
predictions = regressor.predict(test_data[['age']])

# Measuring error
squared_error = [math.pow(actual_value - predictions[index],2) for
index,actual_value in enumerate(test_data['weight'])]
mean_square_error = sum(squared_error)/len(test_data)
{% endhighlight %}
