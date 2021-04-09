# -*- coding: utf-8 -*-
"""bert

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JOXGFA3BIMmPXpWK-ecs6Eh6rvxPFRDP
"""

get_ipython().system('pip install ktrain')
get_ipython().system('pip install sentencepiece')

import pandas as pd

reviews=pd.read_csv("amzn_mobile_reviews.csv")
print(reviews.columns)
new=reviews[["Text","Score"]]
print(new.shape)
new.head()

import numpy as np
X_tr = new["Text"]
y_tr = new['Score']

# complex sentences
print(X_tr[1702])
print(y_tr[1702])

import numpy as np
import tensorflow as tf
import pandas as pd
import ktrain
from ktrain import text

#train test split and random splitting
from sklearn.model_selection import train_test_split
# create training and testing data (80% train and 20% test)
data_train, data_test = train_test_split(new,test_size=0.2,random_state=5)#test size here relates to second var i.e test

print("="*20+" train data (80%)"+"="*20)
print (data_train.shape)
print("="*20+" test data (20%)"+"="*20)
print (data_test.shape)

#activate tpu
import tensorflow as tf

import os
import tensorflow_datasets as tfds

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.experimental.TPUStrategy(resolver)

#with strategy.scope():
(X_train,y_train),(X_test,y_test), preprocess=text.texts_from_df(train_df=data_train,
                   text_column="Text",
                   label_columns="Score",
                   val_df=data_test,
                   maxlen=400,
                   preprocess_mode="bert")

#if reviews are less than 400 the reviews are padded otherwise they are truncated

X_train[0].shape

#initialize the model
with strategy.scope():
  model=text.text_classifier(name="bert",
                           train_data=(X_train,y_train),
                           preproc=preprocess)

#define the learning rate
#but the bert can find optimal learning rate
with strategy.scope():
  learner=ktrain.get_learner(model=model,
                           train_data=(X_train,y_train),
                           val_data=(X_test,y_test),
                           batch_size=12)

#with gpu's it takes 16hrs for 2 epochs with tpu's it takes 2hrs for 2 epochs

#already fitted the first epoch
with strategy.scope():
  learner.fit_onecycle(lr=2e-5,epochs=1)

predictor=ktrain.get_predictor(learner.model,preprocess)


#Un-comment to save the bert-trained model to disk
#predictor.save("bert_pretrained")



import numpy as np
import tensorflow as tf
import pandas as pd
import ktrain
from ktrain import text

# BERT pre-trained model Google Drive Link - https://drive.google.com/drive/folders/1V-LDcldNDYozEpsQ4vLpFNX5o99xZUXX?usp=sharing
# Downlaod the BERT and un-comment the line below for testing - 
#predictor = ktrain.load_predictor('bert_pretrained')

text="GREAT PRODUCT THAT IS AS GREAT FOR NEXTEL AS IT IS FOR BOOST INSERT YOUR SIM & GO!"
# data=[text]
print(predictor.predict_proba(text))
print(predictor.predict(text))  # Score is positve review and not_score is negative