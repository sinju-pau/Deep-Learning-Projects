
# coding: utf-8

# # Deep Learning using Keras in Python : Customer Churn Predictions

# In the targeted approach the company tries to identify in advance customers who are likely to churn. The company then targets those customers with special programs or incentives. This approach can bring in huge loss for a company, if churn predictions are inaccurate, because then firms are wasting incentive money on customers who would have stayed anyway. There are numerous predictive modeling techniques for predicting customer churn.
# 
# The data files state that the data are "artificial based on claims similar to real world". These data are also contained in the C50 R package.
# 
# Data and associated files are also available at: http://www.sgi.com/tech/mlc/db/churn.data
# 
# The analysis is done in Keras library in Python. The task is to predict whether the customer will churn or not, using the given features.

# Start off by importing the libraries in Python

# In[1]:

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Now read the dataset from the url and view it.

# In[2]:

#Importing the dataset
custchurn = pd.read_csv("http://www.sgi.com/tech/mlc/db/churn.data", header = None)


# In[3]:

custchurn.head(10)


# Data is in a very usable format. Check for NA's

# ## Part I : Data Preprocessing

# In[4]:

custchurn.info()


# Now assign the column names from the description and give a random shuffle to the data

# In[5]:

custchurn.columns = ["state","account_length","area_code","phone_number","international_plan","voice_mail_plan","number_vmail_messages","total_day_minutes","total_day_calls","total_day_charge","total_eve_minutes","total_eve_calls","total_eve_charge","total_night_minutes","total_night_calls","total_night_charge","total_intl_minutes","total_intl_calls","total_intl_charge","number_customer_service_calls","churned"]
custchurn = custchurn.sample(frac=1).reset_index(drop=True)
custchurn.head(10)


# ### Create Feature vector and target vector 
# 
# Now create the feature vectors, avoiding 'state', 'area_code','phone_number'. The target vector is the 'churned' variable

# In[6]:

X = custchurn.iloc[:,[1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values
y = custchurn.iloc[:,20].values


# ### Encoding Categorical Features
# 
# We have categorical features,'international_plan', 'voice_mail_plan' and 'churned'.Encode them into numerics

# In[7]:

# Encoding categorical features
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
# Encoding Target variable 'churned'
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# ### Splitting the dataset into the Training set and Test set

# In[8]:

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ### Feature Scaling

# It is absolutely necessary to do feature scaling for neural networks because its computationally intensive !

# In[9]:

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Part II : Building the Artificial Neural Network
# 
# We build the ANN with the Keras library in Python. Also we import the class 'Sequential' for initializing the network as a sequence of layers and the class 'Dense' for building actual layers.

# In[10]:

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # initialize ANN
from keras.layers import Dense # build layers


# In[11]:

# Initialising the ANN
# Defining ANN as a sequence of layers
classifier = Sequential()


# Here we are building one input layer with 17 neurons corresponding to the 17 input features and two hidden layers with 9 neurons in it. The problem being a binary classification, ouput layer has only one neuron. No. of neurons in hidden layer may be calculated as : (17+1)/2 =9
# 
# Also, lets use the activation function 'recifier' for input and hidden layers and 'sigmoid' function for the ouput layer. 

# In[12]:

# Adding the input layer and the first hidden layer
# rectifier act. fun for hidden layers
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))
# output_dim = no of nodes in hidden layer = (17+1)/2 =9


# In[13]:

# Adding the second hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
# sigmoid act. function for output layers
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Now compile the ANN using the optimizer 'adam', loss function, 'binary_crossentropy with the accuracy as the evaluation parameter.

# In[14]:

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Having compliled the ANN, lets now fit the ANN based learning to our training set, in batches of 10, for 100 epochs/iterations

# In[15]:

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# ## Part III : Making the predictions and evaluating the model
# 
# We now make predictions using the predict method and print the confusion matrix

# In[16]:

#Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #important step here

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[17]:

cm


# In[18]:

(cm[0,0]+cm[1,1])/len(y_test)


# ## Concluding Remarks
# 
# 1. The Deep Learning classifier has an accuracy of 94.3 % on the test set. It is a good range for the dataset, but we can still improve the accuracy by adding more hidden layers and training for more number of epochs
# 
# 2. Statistical significance tests may be performed to find the relevant variables for the case
# 
