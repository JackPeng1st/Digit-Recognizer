import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding  
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train.csv',engine='python')
test_data= pd.read_csv('test.csv',engine='python')

train_data_ans=train_data['label']
train_data.drop('label',1,inplace=True)

train_data=train_data.values.reshape(42000, 28*28).astype('float32')
test_data=test_data.values.reshape(28000,28*28).astype('float32') 

train_data_ans = np_utils.to_categorical(train_data_ans) 

train_data=train_data/255
test_data=test_data/255

#x_train, x_test, y_train, y_test = train_test_split(train_data, y, test_size=0.2, random_state=0)

model=Sequential();

model.add(Dense(input_dim=784,units=500))
model.add(Activation('relu'))
#model.add(Dropout(0.7))

model.add(Dense(units=500))
model.add(Activation('relu'))

model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',#Step2: goodness of function
              #Step 3:pick the best function
              optimizer='adam',# 3.1 Configuration(like gradient decent way), ways: SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
              metrics=['accuracy'])

model.fit(train_data,train_data_ans,batch_size=100,epochs=20)

#model.evaluate(train_data,train_data_ans)

result=model.predict(test_data)

result_formal=result.argmax(1)

ID=pd.Series(range(1,28001))
final_result_NN=pd.DataFrame({'ImageId':ID,'Label':result_formal})

final_result_NN.to_csv("Prediction_NN.csv",index=False)
