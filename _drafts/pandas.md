---
layout:     post
title:      "Pandas Library"
subtitle:   "test"
date:       2015-11-24 01:00:00
author:     "NG"
header-img: "img/post-bg-04.jpg"
---

<h2 class="article-heading">Reading Data</h2>

{% highlight python %}
import pandas as pd
# reading a csv file
training_data = pd.read_csv("training_data.csv")

# reading a txt file with whitespaces delimiter and no column
names
col_names = ['age','weight']
training_data = pd.read_table('training.txt',
delim_whitespace=True, names=col_names
{% endhighlight %}

