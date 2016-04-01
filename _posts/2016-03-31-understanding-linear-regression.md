---
layout:     post
title:      "Understanding Linear Regression"
subtitle:   "Bisecting linear regression"
---

{% highlight python %}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
%matplotlib  inline
{% endhighlight %}


{% highlight python %}
train = pd.read_csv('Philadelphia_Crime_Rate_noNA.csv')
{% endhighlight %}


{% highlight python %}
print train.info()
features = train.columns.values

#removing target from list of features
features = np.delete(features, 0)

#removing 'Name' and 'County' as they are strings, from the list of features
features = np.delete(features, [4,5])

target = 'HousePrice'
print train.head(10)
{% endhighlight %}

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 99 entries, 0 to 98
    Data columns (total 7 columns):
    HousePrice         99 non-null int64
    HsPrc ($10,000)    99 non-null float64
    CrimeRate          99 non-null float64
    MilesPhila         99 non-null float64
    PopChg             98 non-null float64
    Name               99 non-null object
    County             99 non-null object
    dtypes: float64(4), int64(1), object(2)
    memory usage: 6.2+ KB
    None
       HousePrice  HsPrc ($10,000)  CrimeRate  MilesPhila  PopChg        Name  \
    0      140463          14.0463       29.7          10    -1.0    Abington   
    1      113033          11.3033       24.1          18     4.0      Ambler   
    2      124186          12.4186       19.5          25     8.0       Aston   
    3      110490          11.0490       49.4          25     2.7    Bensalem   
    4       79124           7.9124       54.1          19     3.9  Bristol B.   
    5       92634           9.2634       48.6          20     0.6  Bristol T.   
    6       89246           8.9246       30.8          15    -2.6  Brookhaven   
    7      195145          19.5145       10.8          20    -3.5  Bryn Athyn   
    8      297342          29.7342       20.2          14     0.6   Bryn Mawr   
    9      264298          26.4298       20.4          26     6.0  Buckingham   
    
         County  
    0  Montgome  
    1  Montgome  
    2  Delaware  
    3     Bucks  
    4     Bucks  
    5     Bucks  
    6  Delaware  
    7  Montgome  
    8  Montgome  
    9     Bucks  


Cleaning NaN Values in 'PopChg'


{% highlight python %}
train['PopChg'] =  train['PopChg'].fillna(0)
train.info()
{% endhighlight %}

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 99 entries, 0 to 98
    Data columns (total 7 columns):
    HousePrice         99 non-null int64
    HsPrc ($10,000)    99 non-null float64
    CrimeRate          99 non-null float64
    MilesPhila         99 non-null float64
    PopChg             99 non-null float64
    Name               99 non-null object
    County             99 non-null object
    dtypes: float64(4), int64(1), object(2)
    memory usage: 6.2+ KB


Writing a method to encode string values to a integer.


{% highlight python %}
def encode_column(col):
    encoded_values = pd.value_counts(train[col]).keys().tolist()
    for each_value in encoded_values:
        train.loc[ train[col] == each_value, 'Encoded '+ col] = encoded_values.index(each_value)
{% endhighlight %}

Encoding County column


{% highlight python %}
encode_column('County')
np.append(features, 'Encoded County')
print "Features are: "
print features
{% endhighlight %}

    Features are: 
    ['HsPrc ($10,000)' 'CrimeRate' 'MilesPhila' 'PopChg']


'HsPrc ($10,000)' field is nothing but the house price expressed in '10,000$'. 
So let's ignore HsPrc


{% highlight python %}
features = np.delete(features, 0)
{% endhighlight %}

Writing a method to find correlation between two features


{% highlight python %}
def is_correlated(field_1, field_2):
    correlation = np.corrcoef(train[field_1],train[field_2])[1,0]
    return (abs(correlation) > 2/(math.sqrt(len(train)))), correlation
{% endhighlight %}

Checking which all features are correlated with the target.


{% highlight python %}
for each_feature in features:
    is_correlated_bool , correlation_factor = is_correlated(each_feature, target)
    print each_feature + "  is correlated with target " + str(is_correlated_bool) + " factor: " + str(correlation_factor)
{% endhighlight %}

    CrimeRate  is correlated with target True factor: -0.249960930497
    MilesPhila  is correlated with target False factor: 0.149764402712
    PopChg  is correlated with target False factor: 0.109259549326


Checking if any of the feautres are correlated with each other


{% highlight python %}
for each_feature in features:
    for compare_to_feature in features:
        if each_feature != compare_to_feature:
            is_correlated_bool, correlation_factor = is_correlated(each_feature, compare_to_feature)
            print(each_feature + ' and ' + compare_to_feature + ' are correlated: ' +
                  str(is_correlated_bool) + ' correlation factor: ' + str(correlation_factor))
{% endhighlight %}

    CrimeRate and MilesPhila are correlated: True correlation factor: -0.249485210625
    CrimeRate and PopChg are correlated: False correlation factor: -0.027997342653
    MilesPhila and CrimeRate are correlated: True correlation factor: -0.249485210625
    MilesPhila and PopChg are correlated: True correlation factor: 0.571668269279
    PopChg and CrimeRate are correlated: False correlation factor: -0.027997342653
    PopChg and MilesPhila are correlated: True correlation factor: 0.571668269279


Since PopChg and MilesPhila is not correlated with the target therefore we drop both from the list of features.


{% highlight python %}
features = np.delete(features, [1,2])
{% endhighlight %}


{% highlight python %}
features = ['CrimeRate']
{% endhighlight %}

Plotting graph between CrimeRate and HousePrice


{% highlight python %}
plt.scatter(train['CrimeRate'], train['HousePrice'])
plt.xlabel('CrimeRate')
plt.ylabel('HousePrice')
plt.show()
{% endhighlight %}

    /usr/local/lib/python2.7/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if self._edgecolors == str('face'):



![Output]({{site.baseurl}}/img/understanding_linear_regression/output_21_1.png)


We can notice that there is a outlier whose CrimeRate is greater then 350. 
Let's ignore the outlier as it is adding unnecessary noise.


{% highlight python %}
train = train[train['CrimeRate'] < 350]
{% endhighlight %}

Plotting the graph again


{% highlight python %}
plt.scatter(train['CrimeRate'], train['HousePrice'])
plt.xlabel('CrimeRate')
plt.ylabel('HousePrice')
plt.show()
{% endhighlight %}


![Output]({{site.baseurl}}/img/understanding_linear_regression/output_25_0.png)


## Predicitng the trend using scikit's linear regression library.


{% highlight python %}
import sklearn.linear_model as linear_model
{% endhighlight %}

Splitting the data into training_set and test_set


{% highlight python %}
training_set = train.head(len(train)/2)
test_set = train.tail(len(train)/2)
test_set.info()
{% endhighlight %}

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 49 entries, 49 to 98
    Data columns (total 8 columns):
    HousePrice         49 non-null int64
    HsPrc ($10,000)    49 non-null float64
    CrimeRate          49 non-null float64
    MilesPhila         49 non-null float64
    PopChg             49 non-null float64
    Name               49 non-null object
    County             49 non-null object
    Encoded County     49 non-null float64
    dtypes: float64(5), int64(1), object(2)
    memory usage: 3.4+ KB



{% highlight python %}
model = linear_model.LinearRegression()

feature_values = training_set['CrimeRate'].values
target_values = training_set['HousePrice'].values
feature_values = feature_values.reshape(len(training_set),1)
target_values = target_values.reshape(len(training_set),1)

model.fit(feature_values, target_values)
{% endhighlight %}




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



Plotting the trend predicted by the model.


{% highlight python %}
test_values = test_set['CrimeRate'].values
test_values = test_values.reshape(len(test_set),1)

plt.scatter(test_set['CrimeRate'], test_set['HousePrice'], label='actual')
plt.plot(test_set['CrimeRate'], model.predict(test_values), label='predicted')
plt.xlabel('CrimeRate')
plt.ylabel('HousePrice')
plt.show()
{% endhighlight %}


![Output]({{site.baseurl}}/img/understanding_linear_regression/output_32_0.png)


Let's see what noise would have occurred if he had bilindly considered all the features.


{% highlight python %}
feature_values = training_set[['CrimeRate' , 'MilesPhila', 'PopChg']].values
feature_values = feature_values.reshape(len(training_set),3)

target_values = training_set['HousePrice'].values
target_values = target_values.reshape(len(training_set),1)

test_values = test_set[['CrimeRate', 'MilesPhila', 'PopChg']].values
test_values = test_values.reshape(len(test_set),3)

new_model = linear_model.LinearRegression()
new_model.fit(feature_values, target_values)

plt.scatter(test_set['CrimeRate'], test_set['HousePrice'], c='blue')
plt.scatter(test_set['CrimeRate'], new_model.predict(test_values), c='red')
plt.xlabel('CrimeRate')
plt.ylabel('HousePrice')
plt.show()
{% endhighlight %}


![Output]({{site.baseurl}}/img/understanding_linear_regression/output_34_0.png)


## Tuning the Linear Regression algorithm

Linear Regression Algorithm accetps the following parametes:  
fit_intercept=True, normalize=False, copy_X=True, n_jobs=1

Writing a method which accepts a model and title:
    the method fits training data and plots the graph with test data


{% highlight python %}
def fit_data_with_model(model, title):
    feature_values = training_set['CrimeRate'].values
    target_values = training_set['HousePrice'].values
    feature_values = feature_values.reshape(len(training_set),1)
    target_values = target_values.reshape(len(training_set),1)
    
    model.fit(feature_values, target_values)
    
    test_values = test_set['CrimeRate'].values
    test_values = test_values.reshape(len(test_set),1)
    
    plt.scatter(test_set['CrimeRate'], test_set['HousePrice'])
    plt.plot(test_set['CrimeRate'], model.predict(test_values))
    plt.xlabel('CrimeRate')
    plt.ylabel('HousePrice')
    plt.title(title)
    plt.show()
{% endhighlight %}

#### fit_intercept

fit_intercept when true calculates the intercept of the model, if false no intercept will be calculated ( expecting that the data is already centered)


{% highlight python %}
model = linear_model.LinearRegression(fit_intercept=False)
model_with_fit_intercept = linear_model.LinearRegression(fit_intercept=True)

fit_data_with_model(model, "With set_intercept as False")
fit_data_with_model(model_with_fit_intercept, "With set_intercept as True")
{% endhighlight %}


![Output]({{site.baseurl}}/img/understanding_linear_regression/output_41_0.png)



![Output]({{site.baseurl}}/img/understanding_linear_regression/output_41_1.png)


So we can notice that when the set_intercept is set as False, the algorithm considers the intercept of the line to be zero therefore the linear regression equation will become: y = mx

Whereas when the set_intercept is set to True, the algorithm computes the intercept value of the line therefore the linear regression equation will become: y = mx + intercept

#### normalize and copy_X

normalize
Default False, if True algorithm will normalizes the features before regression.

copy_X
Default True, if False will modify the features values rather then copying the features data.


{% highlight python %}
model = linear_model.LinearRegression(normalize=False)
model_normalize = linear_model.LinearRegression(normalize=True, copy_X=False)

fit_data_with_model(model, "With normalize as False")

feature_values = training_set['CrimeRate'].values
target_values = training_set['HousePrice'].values
feature_values = feature_values.reshape(len(training_set),1)
target_values = target_values.reshape(len(training_set),1)

model_normalize.fit(feature_values, target_values)
# the regression algorithm normalized the feature_values before fitting it 
# and since copy_X is false it modifies the feature_values variable rather
# then copying the data and then modifying the copied version
print feature_values

test_values = test_set['CrimeRate'].values
test_values = test_values.reshape(len(test_set),1)

plt.scatter(test_set['CrimeRate'], test_set['HousePrice'])
plt.plot(test_set['CrimeRate'], model_normalize.predict(test_values))
plt.xlabel('CrimeRate')
plt.ylabel('HousePrice')
plt.title('with normalize as True')
plt.show()
{% endhighlight %}


![Output]({{site.baseurl}}/img/understanding_linear_regression/output_45_0.png)


    [[ 0.01070856]
     [-0.05248348]
     [-0.10439123]
     [ 0.23300916]
     [ 0.28604534]
     [ 0.22398172]
     [ 0.02312129]
     [-0.20256459]
     [-0.09649223]
     [-0.09423537]
     [-0.12921668]
     [ 0.24316502]
     [ 0.06148789]
     [ 0.05584574]
     [ 0.19125727]
     [-0.07279521]
     [ 0.21833957]
     [-0.10551966]
     [ 0.48690577]
     [ 0.18448669]
     [ 0.02876344]
     [-0.04345605]
     [-0.01750217]
     [-0.12470296]
     [ 0.00280956]
     [-0.08295108]
     [-0.00170416]
     [ 0.00619485]
     [-0.12695982]
     [-0.13824412]
     [-0.12357453]
     [-0.12921668]
     [ 0.02763501]
     [-0.1608127 ]
     [-0.14727155]
     [-0.00170416]
     [-0.08972165]
     [ 0.00619485]
     [-0.10664809]
     [ 0.12129465]
     [-0.17773914]
     [-0.05248348]
     [ 0.1043682 ]
     [ 0.12806522]
     [ 0.08292805]
     [-0.02427275]
     [-0.04119919]
     [-0.15968427]
     [-0.11906082]]



![Output]({{site.baseurl}}/img/understanding_linear_regression/output_45_2.png)

#### n_jobs

Denotes the number of CPU's to be used.
If -1 all the CPU's are used. The more number of CPU's the algortihm computation will be faster
