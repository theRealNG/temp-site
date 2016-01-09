---
layout:     post
title:      "Data Analysis"
subtitle:   "Understanding data"
---

#### Categorical Variables

Categorical variables are values that are names or labels  
ex: color of a ball: red, blue or yellow  
While modeling we would replace the color with two dummy variables:  
red and blue  
if the ball is red: red = 1, blue = 0;  
if the ball is blue: red = 0, blue = 1;  
if the ball is yellow: red = 0, blue = 0;

{% highlight python %}

ball_data = pandas.read('ball_data.csv')
ball_types = ball_data['color'].unique()

dummy = pandas.DataFrame()

for ball_type in ball_types[:-1]:
    dummy[str(ball_type) + '-color'] = (ball_data['color'] == ball_type).astype(int)

ball_data = pandas.concat([ball_data, dummy], axis=1)

del ball_data['color']

{% endhighlight %}
