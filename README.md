# My AIIP submition
This file is to express my thought process through making the MLP
and hopefully explains the choices I have made. It will be broken down
into the following parts:
1. Pre-processing
2. Model choice
3. Evaluation metric
4. Challenges

### Pre-processing
For this step I first got rid of all Null cases by interpolating them. The reason I interpolated 
and not simply removed them is because in my EDA I noticed a lot of rows had data missing here and there.
If I just snapped all the Null rows away, my dataset would be left very small.

Next I turn all the duplicated rows as part of cleaning the data. Lastly, I standardized all the numerical
data for the model to use.

The rationale of all these is to preserve the dataset's richness and to eliminate noise that could interfere
with the training

### Model Choice
I had to do extensive research online before coming to the conclusion that XGBoost works best.

I decided on XGBoost as it is designed to handle tabular data well, which in the context of this
challenge is exactly what I am working with. The model is also proven to be quite robust to overfitting
and can handle non-linear relationships (which I assume temperature would have with some of the data)

Lastly, it seemed to me that its very flexible in implementation which allows me to perform
more modification to my model to fit the data or anything else I noticed.

### Evaluation Metrics

I picked RMSE as the metric because during the EDA, I noticed some data in certain group were high variance.
I learnt from my school that RMSE specifically penalizes high variance and favours consistent performance 
over occasional large mistakes. 

This also adds up logically in the context of temperatures of a region as you would rarelye expect temperatures to vary too far from normal

### Challenges

This is my first time working with a Machine Learning Pipeline so many thing required Google and 
a lot of asking around. While I did study how basic Machine Learning works in my school mods,
it only covered simple PyTorch Neural Networks (which I don't think is suitable here).

This is also my first time hearing abnout XGBoost so it required some time to learn what it is
and whats so different from other models that exist.

### Result analysis

The overall performance of my model is not very ideal. There could be many contributing factors to this
but I am unable to solve it due to the time constraint of the project.

I suspect it has to be something with my pre-processing as the model seem to be set up properly. I 
maybe had to check through the data more thouroughly for more outliers or weird characteristics.

I would like to go back and attempt to clean this up in the future after having more time to learn and
improve.

