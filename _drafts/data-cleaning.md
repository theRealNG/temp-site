---
layout:     post
title:      "Data Cleaning"
subtitle:   "Preparation for the evolution"
---

Data Cleaning is an essential process so that we can remove unwanted
data, invalid rows, convert data from one type to another etc.

{% highlight python %}
#removing rows with NaN values
training_data = training_data.dropna(subset=["age","weight"])

#converting columns from one type to another
training_data = training_data["age"].astype(int)

#selecting data
training_data = training_data[training_data['age'] > 18]

#deleting a column
del training_data['id']

{% endhighlight %}
