# deep-learning -re gression model
### Project overview 
We will be covering a regression task where we will be predicting the amount of premiums to be charged based on several  columns. We will  be perfominng this regression task using deep learning techniques where we will train the model an dwe will be using early stopping as our tuning parameter(works by when the model's perfomance stops improving then the model stops training.
 
 The columns include:
 
 1.Age - Ba sed on the age of the customer we can be able to determine the value of premiums one will pay in normal scenario the older one is the higher the premium rate is likely to be.

2.Sex - Based on sex we can be able to determine the amount one will indeed pay as premiums .Will one pay more as a male while compared to a female? we will answer this question when we plot a correlation matrix and feature importance plot.

3.BMI - BMI (Body Mass Index) is a commonly used measure to assess a person's weight status and is calculated using the following formula:

BMI = weight / (height^2)

where weight is in kilograms and height is in meters. BMI is a numerical value that provides an indication of whether a person is underweight, normal weight, overweight, or obese.

The BMI categories are generally defined as follows:

Underweight: BMI < 18.5 Normal weight: 18.5 <= BMI < 25 Overweight: 25 <= BMI < 30 Obese: BMI >= 30 .

4 .Number of children - based on the number of children we can be able do determine whether one will pay higher premiumns or not which ideally is the case.

5 Region - We will look where each customer comes from and based on region one comes from can we really predict the amount one will pay as premiums? we wil se shortly whether the region one comes from really affects the insurance premium
### Data ingestion and cleaning
We will then import several libraries for the running of our project:
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
from __future__ import absolute_import, division  , print_function

import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import   layers
```
We will look for missiing values in our dataset we have none (when dealing with null values if they are many values then we fill using the median its advisable since it will not introduce skewenes  to our data.When the values are less then we drop the null values since they dont have a significant effect to our data
### EDA
we will start looking at the distribution of our indipendent columns and look at the distribution of data while gaining more understanding of our data

We will loook at the distribution of our  **age column** using a distplot;
```
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(df['age'])
plt.title('Age distribution')
plt.show()
```

![a](https://github.com/stilinsk/deep-learning-model/assets/113185012/98671be4-ed11-43ba-9045-d83e4856f256)


Next we look for the **sex distribution**
```
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 6))

# Set the color palette for the countplot
colors = {'female': 'pink', 'male': 'blue'}
sns.set_palette(colors.values())

ax = sns.countplot(x='sex', data=df)
plt.title('Sex distribution')

# Add the count values as text on top of each bar
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height, height, ha="center")

plt.show()
```

![a](https://github.com/stilinsk/deep-learning-model/assets/113185012/645ddc26-4c7f-4316-8249-81e6a4e7a05e)

We will also look for the **bmi distribution**
```
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(df['bmi'])
plt.title('bmi distribution')
plt.show()
```
![a](https://github.com/stilinsk/deep-learning-model/assets/113185012/c4731c0e-b424-41b3-a6f1-9c7114a3e461)


We will  do a count plot for the **number of children per family**
```
#in the code below we will be looking for the value of each count of each family and number of children using a countplot
plt.figure(figsize =(6,6))
sns.countplot(x='children',data=df)
plt.title('choldren per family')
ax = sns.countplot(x='children', data=df)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height, height, ha="center")

plt.show()
```
![a](https://github.com/stilinsk/deep-learning-model/assets/113185012/b4e63e5a-1b24-4584-a730-e3916afc25e8)


We will look at the **smoker** distribution
```
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 6))

# Set the color palette for the countplot
colors = {'yes': 'green', 'no': 'red'}
sns.set_palette(colors.values())

ax = sns.countplot(x='smoker', data=df)
plt.title('Smokers Distribution')

# Add the count values as text on top of each bar
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height, height, ha="center")

plt.show()
```
![a](https://github.com/stilinsk/deep-learning-model/assets/113185012/b45c4411-b62c-4b0a-96f8-d75719d6a0dd)

We will also look at the **region distribution**
```
# Below we willl be looking for a count  of the values from each region using a countplot

plt.figure(figsize =(6,6))
sns.countplot(x='region',data=df)
plt.title('region distribution')
ax = sns.countplot(x='region', data=df)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height, height, ha="center")

plt.show()
```
![a](https://github.com/stilinsk/deep-learning-model/assets/113185012/f67b496f-9163-4bee-ace6-fac9b6a4245b)

We will  finally look at the distribution of our target column **charges** and look at its skeweness
```

sns.set()
plt.figure(figsize=(6,6))
sns.distplot(df['charges'])
plt.title('charges distribution')
plt.show()
```
![a](https://github.com/stilinsk/deep-learning-model/assets/113185012/ece5ced1-89f7-4d00-8a05-609ec469bdbe)

### conclusions and observations

- well our customers age is not skewed to any direction ad is distributed almost fairly more young people are insuring thats for sure and mid 40's and also 60's

- In the above plot we can really see that more male are taking insurance covers than their feamle counterparts
  
 -Our customers bmi is mainly from 25 to oround 38 from the information i gave above while discussing the problem statement is that from 25 -30 they are overweight and past 30 they are obese thus most of our patients are taking insurance cover beacuse they are not in good sahape and some may have more fears and we can outrightly predict that a higher obese would actually attract a higher premium rate charge

- Most patients are childless and this could be brought about by some could still be single thus as we looked before we saw that most people are in thie twenties .

- In the above plot we have the distribution of smokers though not by sex but we apparently have less smokers. Thus by default a smoker has a higher risk of diseases and thus we expect the insurance premiums to be higher

- The charges distributiion are left skewed thus the insurance cost is generally lower and this may also be a factor ( its affordable)

  ### Data preprocessing and model building
   We will need to first encode our data this is because we are dealing with non -numeric data
```
df.replace({'sex':{'male':0,'female':1}},inplace=True)
df.replace({'smoker':{'yes':0,'no':1}},inplace=True)
df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)
```
##### splitting the data
```
train=df.sample(frac=0.8,random_state =0)
test=df.drop(train.index)
```
#### model building
```
train_labels=train.pop("charges")
test_labels =test.pop("charges")
def norm(x):
    return(x-train_stats["mean"])/train_stats["std"]
norm_train_data = norm(train)

norm_test_data = norm(test)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

model = build_model()
model.summary()
```

We will look at our model perfomance and test whether our model is overfitting
```

import matplotlib.pyplot as plt

# Your existing code

history = model.fit(
  norm_train_data, train_labels,
  epochs=EPOCHS, validation_split=0.2, verbose=0,
  callbacks=[PrintDot()]
)

hist = pd.DataFrame(history.history)

 Plotting MAE and Validation MAE
plt.figure()
plt.plot(hist['mae'], label='MAE')
plt.plot(hist['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE vs. Validation MAE')
plt.legend()

Plotting MSE and Validation MSE
plt.figure()
plt.plot(hist['mse'], label='MSE')
plt.plot(hist['val_mse'], label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE vs. Validation MSE')
plt.legend()

plt.show()
```
![a](https://github.com/stilinsk/deep-learning-model/assets/113185012/971992d6-5060-4b6d-b205-02f1a3e74f97)


As we can see our model is not perfoming and this may be due to instancces of overfitting and thats why our validation mae and validation mse are tending to become constant thus we will tune our mode and use early stopping as our parameter where if the model notices that its perfomance is not improving then the model stops training we will see the implementation in the below code


#### Implementing the early stopping to our model
```
import matplotlib.pyplot as plt


model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    norm_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[early_stop, PrintDot()]
)

def plot_history(history):
    plt.figure()
    plt.plot(history.history['mae'], label='MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE vs. Validation MAE')
    plt.legend()

    plt.figure()
    plt.plot(history.history['mse'], label='MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE vs. Validation MSE')
    plt.legend()

    plt.show()
```
![a](https://github.com/stilinsk/deep-learning-model/assets/113185012/e5334477-93b8-4767-aaa4-d69f8858166c)


As wexcan see from our model the point it has sstopped improving then the training has stopped thus we can see the val mse and val mae is hence reducing thus our model perfomance is good
Lets look at the model's mean absolute error

```
loss, mae, mse = model.evaluate(norm_test_data, test_labels, verbose=0)
print("Testing set absolute error: {:5.2f}$".format(mae))
```
### The mean_absolute_error form our data is 3133 a a fair perfomance

