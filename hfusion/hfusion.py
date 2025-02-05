# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 10:37:22 2017

@author: soujanyaporia
"""
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras import backend as K
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Input,Dense,GRU,LSTM,Concatenate,Dropout,Activation,Add, Masking
from keras.layers.pooling import AveragePooling1D,MaxPooling1D
from keras.layers.core import Flatten
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Reshape
from keras.backend import shape
from keras.utils import plot_model
from keras.layers.merge import Multiply,Concatenate
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from keras.optimizers import RMSprop,Adadelta,Adam

from keras.callbacks import Callback

import sys

def createOneHot(train_label,  test_label):


  maxlen = int(max(train_label.max(), test_label.max()))
  
  train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen+1))
  test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen+1))
  
  for i in range(train_label.shape[0]):
    for j in range(train_label.shape[1]):
      train[i,j,train_label[i,j]]=1

  for i in range(test_label.shape[0]):
    for j in range(test_label.shape[1]):
      test[i,j,test_label[i,j]]=1

  return train,  test

def calc_test_result(result, test_label, test_mask):

  true_label=[]
  predicted_label=[]

  for i in range(result.shape[0]):
    for j in range(result.shape[1]):
      if test_mask[i,j]==1:
        true_label.append(np.argmax(test_label[i,j] ))
        predicted_label.append(np.argmax(result[i,j] ))
    
  print("Confusion Matrix :")
  print(confusion_matrix(true_label, predicted_label))
  print("Classification Report :")
  print(classification_report(true_label, predicted_label,digits=4))
  print("Accuracy ", accuracy_score(true_label, predicted_label))
  print("Macro Classification Report :")
  print(precision_recall_fscore_support(true_label, predicted_label,average='macro'))
  print("Weighted Classification Report :")
  print(precision_recall_fscore_support(true_label, predicted_label,average='weighted'))
  #print "Normal Classification Report :"
  #print precision_recall_fscore_support(true_label, predicted_label)







# 切分出文本特征？
def crop(a):
    return a[:, 0:text_dim]
# 切分出视觉特征？
def crop1(a):
    return a[:, text_dim:(text_dim+visual_dim)]
# 切分出音频特征？
def crop2(a):
    return a[:, (text_dim+visual_dim):dim]
def output_of_lambda(input_shape):
    return (input_shape[0], text_dim)
def output_of_lambda1(input_shape):
    return (input_shape[0], visual_dim)
def output_of_lambda2(input_shape):
    return (input_shape[0], audio_dim)







# ################################################################################
# #Level 2

# #concatenate level 1 output to be sent to hfusion
# fused_tensor=Concatenate(axis=2)([context_1_2,context_1_3,context_2_3])


def load_bimodal_activations():     
  with open('./bimodal.pickle', 'rb') as handle:
      activations = pickle.load(handle, encoding = 'latin1')  
  merged_train_data = activations['train_bimodal']
  merged_test_data = activations['test_bimodal']
  train_mask=activations['train_mask']
  test_mask=activations['test_mask']
  train_label=activations['train_label']
  test_label=activations['test_label']


  #Setting dummy utterances to be 0
  for i in range(activations['train_bimodal'].shape[0]):
    for j in range(activations['train_bimodal'].shape[1]):
      if train_mask[i][j] == 0.0 :

        merged_train_data[i,j,:]=0.0

  for i in range(activations['test_bimodal'].shape[0]):
    for j in range(activations['test_bimodal'].shape[1]):
      if test_mask[i][j] == 0.0 :

        merged_test_data[i,j,:]=0.0

  audio_dim=500
  visual_dim=500
  text_dim=500
  dim = audio_dim + visual_dim + text_dim
  max_len = merged_train_data.shape[1] #max number of utterances per video
  dim_proj=450

  return merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask, max_len


def load_unimodal_activations():
  print("1. load_unimodal_activations ")

  with open('./unimodal.pickle', 'rb') as handle:
      unimodal_activations = pickle.load(handle, encoding = 'latin1')

  merged_train_data = np.concatenate((unimodal_activations['text_train'], unimodal_activations['audio_train'], unimodal_activations['video_train']), axis=2)
  merged_test_data = np.concatenate((unimodal_activations['text_test'], unimodal_activations['audio_test'], unimodal_activations['video_test']), axis=2)
  
  train_mask=unimodal_activations['train_mask']
  test_mask=unimodal_activations['test_mask']
  train_label=unimodal_activations['train_label']
  test_label=unimodal_activations['test_label']

  print("\n merged_train_data.shape; train_label.shape; train_mask.shape")
  print("\n merged_test_data.shape; test_label.shape; test_mask.shape")
  print(merged_train_data.shape)
  print(train_label.shape)
  print(train_mask.shape)
  print(merged_test_data.shape)
  print(test_label.shape)
  print(test_mask.shape)


  #Setting dummy utterances to be 0  填充0值
  for i in range(unimodal_activations['audio_train'].shape[0]):
    for j in range(unimodal_activations['audio_train'].shape[1]):
      if train_mask[i][j] == 0.0 :

        merged_train_data[i,j,:]=0.0

  for i in range(unimodal_activations['audio_test'].shape[0]):
    for j in range(unimodal_activations['audio_test'].shape[1]):
      if test_mask[i][j] == 0.0 :

        merged_test_data[i,j,:]=0.0

  
  global audio_dim, visual_dim, text_dim, dim, max_len, dim_proj


  audio_dim=unimodal_activations['audio_train'].shape[2]
  visual_dim=unimodal_activations['video_train'].shape[2]
  text_dim=unimodal_activations['text_train'].shape[2]
  dim = audio_dim + visual_dim + text_dim
  max_len = merged_train_data.shape[1]  # max number of utterances per video
  dim_proj=450

  print("\n audio_dim; visual_dim; text_dim; dim; max_len; dim_proj")
  print(audio_dim)
  print(visual_dim)
  print(text_dim)
  print(dim)
  print(max_len)
  print(dim_proj)

  return merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask


def Bimodal():

  class hadamard2(Layer):
      def __init__(self, prefix, **kwargs):
          print("\n调用了hadamard2类的__init__方法，并打印一下相关参数")
          print(prefix, **kwargs)
          self.supports_masking = True
          self.prefix = prefix
          super(hadamard2, self).__init__(**kwargs)

      def build(self, input_shape):
          print("\n调用了hadamard2类的build方法")
          print(len(input_shape))
          # Create a trainable weight variable for this layer.  为这一层创建一个可训练的权重变量
          self.output_dim = dim_proj

          self.kernel1 = self.add_weight(name=self.prefix+'kernel1', 
                                        shape=(self.output_dim,),
                                        initializer='TruncatedNormal',
                                        trainable=True)
          self.kernel2 = self.add_weight(name=self.prefix+'kernel2', 
                                        shape=(self.output_dim,),
                                        initializer='TruncatedNormal',
                                        trainable=True)
          self.bias = self.add_weight(name=self.prefix+'bias', 
                                        shape=(self.output_dim,),
                                        initializer='zeros',
                                        trainable=True)
          super(hadamard2, self).build(input_shape)  # Be sure to call this somewhere!

      def call(self, x):
          print("\n调用了hadamard2类的call方法")
          assert (K.int_shape(x[0])[1] <= dim_proj)
          return K.tanh(x[0]*self.kernel1 + x[1]*self.kernel2 + self.bias)

      def compute_output_shape(self, input_shape):
          return (input_shape[0][0],self.output_dim)


  # 获取单模态特征并获取模态相关维度
  merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask = load_unimodal_activations()
  
  #################################################################################
  #Level 1 sub-model : Hfusion

  # 输入
  x=Input(shape=(audio_dim+visual_dim+text_dim,), name='bi_input')
  print(x.shape)    # (?, 300)
  # 对mask_value =0.0的单元进行忽略
  masked = Masking(mask_value =0.0 , name='bi_mask')(x)
  x1 = Lambda(crop,output_shape=output_of_lambda, name='bi_crop1')(masked)   # 将函数封装成Layer层  text
  x2 =Lambda(crop1,output_shape=output_of_lambda1, name='bi_crop2')(masked)  # video
  x3 =Lambda(crop2,output_shape=output_of_lambda2, name='bi_crop3')(masked)  # audio

  print("\n x1; x2; x3")
  print(x1.shape)   # (?, 100)
  print(x2.shape)   # (?, 100)
  print(x3.shape)   # (?, 100)

  # 将各自的单模态分别经过一个Dense层进行统一维度
  dense1=Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bi_dense1')(x1)
  dense2=Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bi_dense2')(x2)
  dense3=Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bi_dense3')(x3)

  # 将两两模态进行结合
  fused_1_2=hadamard2('1')([dense1,dense2])  # 调用相关类
  fused_1_3=hadamard2('2')([dense1,dense3])
  fused_2_3=hadamard2('3')([dense2,dense3])

  print("\n fused_1_2; fused_1_3; fused_2_3")
  print(fused_1_2.shape)   # (?, 450)
  print(fused_1_3.shape)   # (?, 450)
  print(fused_2_3.shape)   # (?, 450)

  # 建立三个模型
  bimodal_1_2=Model(x,outputs=fused_1_2)
  bimodal_1_3=Model(x,outputs=fused_1_3)
  bimodal_2_3=Model(x,outputs=fused_2_3)

  bimodal_1_2.summary()
  bimodal_1_3.summary()
  bimodal_2_3.summary()


  ################################################################################
  #Level 1 sub-model : biLSTM 

  input_data = Input(shape=(max_len, K.int_shape(fused_1_2)[1],), name='bi_lstm_input1')
  print("\n input_data.shape")
  print(input_data.shape)     # (?, 110, 450)
  lstm = GRU(600, activation='tanh', return_sequences = True, dropout=0.4, name='bi_lstm1')(input_data)
  inter = Dropout(0.9, name='bi_dropout11')(lstm)
  inter = TimeDistributed(Dense(500,activation='tanh', name='bi_tdis11'))(inter)
  output1 = TimeDistributed(Dense(4,activation='softmax', name='bi_tdis12'))(inter)
  lstm1 = Model(input_data, output1)  # (110, 4)
  aux1 = Model(input_data, inter)

  print("\n lstm1; aux1")
  lstm1.summary()
  aux1.summary()


  input_data = Input(shape=(max_len, K.int_shape(fused_1_2)[1],), name='bi_lstm_input2')
  lstm = GRU(600, activation='tanh', return_sequences = True, dropout=0.4, name='bi_lstm2')(input_data)
  inter = Dropout(0.9, name='bi_dropout21')(lstm)
  inter = TimeDistributed(Dense(500,activation='tanh', name='bi_tdis21'))(inter)
  output2 = TimeDistributed(Dense(4,activation='softmax', name='bi_tdis22'))(inter)
  lstm2 = Model(input_data, output2)
  aux2 = Model(input_data, inter)

  print("\n lstm2; aux2")
  lstm2.summary()
  aux2.summary()


  input_data = Input(shape=(max_len, K.int_shape(fused_1_2)[1],), name='bi_lstm_input3')
  lstm = GRU(600, activation='tanh', return_sequences = True, dropout=0.4, name='bi_lstm3')(input_data)
  inter = Dropout(0.9, name='bi_dropout31')(lstm)
  inter = TimeDistributed(Dense(500,activation='tanh', name='bi_tdis31'))(inter)
  output3 = TimeDistributed(Dense(4,activation='softmax', name='bi_tdis32'))(inter)
  lstm3 = Model(input_data, output3)
  aux3 = Model(input_data, inter)

  print("\n lstm3; aux3")
  lstm3.summary()
  aux3.summary()

  ################################################################################
  #Complete Level 1 : Lstm applied on hfusion 

  main_input=Input(shape=(max_len,audio_dim+visual_dim+text_dim,), name='bi_main_input')

  print("\n main_input.shape")
  print(main_input)  # (?, 110, 300)   Tensor("bi_main_input:0", shape=(?, 110, 300), dtype=float32)

  video_uttr_1_2=TimeDistributed(bimodal_1_2, name='bi_tdis_b12')(main_input)
  video_uttr_1_3=TimeDistributed(bimodal_1_3, name='bi_tdis_b13')(main_input)
  video_uttr_2_3=TimeDistributed(bimodal_2_3, name='bi_tdis_b23')(main_input)

  print("\n video_uttr_1_2; video_uttr_1_3; video_uttr_2_3")
  print(video_uttr_1_2.shape)   # (?, 110, 450)
  print(video_uttr_1_3.shape)   # (?, 110, 450)
  print(video_uttr_2_3.shape)   # (?, 110, 450)
  
  aux_1_2=aux1(video_uttr_1_2)
  aux_1_3=aux2(video_uttr_1_3)
  aux_2_3=aux3(video_uttr_2_3)

  print("\n aux_1_2; aux_1_3; aux_2_3")
  print(aux_1_2.shape)   # (?, 110, 500)
  print(aux_1_3.shape)   # (?, 110, 500)
  print(aux_2_3.shape)   # (?, 110, 500)





  context_1_2=lstm1(video_uttr_1_2)   # video_uttr_1_2 = (?, 110, 450)
  BM1 = Model(main_input, context_1_2)
  Aux1 = Model(main_input, aux_1_2)

  print("\n BM1; Aux1")
  BM1.summary()
  Aux1.summary()

  context_1_3=lstm2(video_uttr_1_3)
  BM2 = Model(main_input, context_1_3)
  Aux2 = Model(main_input, aux_1_3)
  context_2_3=lstm3(video_uttr_2_3)
  BM3 = Model(main_input, context_2_3)
  Aux3 = Model(main_input, aux_2_3)


  #Training and fitting of Bimodals

  optimizer=Adam(lr=0.001)
  # BM1才是要编译的模型
  BM1.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal')
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)
  BM1.fit(merged_train_data, train_label,
                  epochs=200,
                  batch_size=20,
                  sample_weight=train_mask,
                  shuffle=True, callbacks=[early_stopping],
                  validation_split=0.1)

  # Aux1是预测模型？？
  train_result1 = Aux1.predict(merged_train_data)
  print("\n train_result1")
  print(train_result1.shape)
  test_result1 = Aux1.predict(merged_test_data)
  print("\n test_result1")
  print(test_result1.shape)

  BM2.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal')
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)
  BM2.fit(merged_train_data, train_label,
                  epochs=200,
                  batch_size=20,
                  sample_weight=train_mask,
                  shuffle=True, callbacks=[early_stopping],
                  validation_split=0.1)
  train_result2 = Aux2.predict(merged_train_data)
  test_result2 = Aux2.predict(merged_test_data)

  BM3.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal')
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)
  BM3.fit(merged_train_data, train_label,
                  epochs=200,
                  batch_size=20,
                  sample_weight=train_mask,
                  shuffle=True, callbacks=[early_stopping],
                  validation_split=0.1)
  train_result3 = Aux3.predict(merged_train_data)
  test_result3 = Aux3.predict(merged_test_data)

  train_bimodal = np.concatenate((train_result1, train_result2, train_result3), axis=2)
  test_bimodal = np.concatenate((test_result1, test_result2, test_result3), axis=2)
  print("\n train_bimodal; test_bimodal")
  print(test_bimodal.shape)  # (30, 110, 1500)
  print(test_bimodal.shape)  # (30, 110, 1500)

  bimodal_activations={}
  bimodal_activations['train_bimodal'] = train_bimodal
  bimodal_activations['test_bimodal'] = test_bimodal
  bimodal_activations['train_mask']=train_mask
  bimodal_activations['test_mask']= test_mask
  bimodal_activations['train_label']=train_label
  bimodal_activations['test_label']=test_label

  #merge all output
  print("Saving bimodal activations")
  with open('bimodal.pickle', 'wb') as handle:
       pickle.dump(bimodal_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)


def Trimodal():
  print("---------------Trimodal----------------------")

  # 加载Bimodal中的特征值
  merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask, \
              maxlen = load_bimodal_activations()

  print("\n merged_train_data, merged_test_data, test_label, train_mask, test_mask, maxlen")
  print(merged_train_data.shape)  # (120, 110, 1500)
  print(merged_test_data.shape)   # (30, 110, 1500)
  print(train_label.shape)        # (120, 110, 4)
  print(test_label.shape)         # (30, 110, 4)
  print(train_mask.shape)         # (120, 110)
  print(test_mask.shape)          # (30, 110)
  print(maxlen)                   # 110

  #################################################################################
  class TestCallback(Callback):
      def __init__(self, test_data):
          self.test_data = test_data

      def on_epoch_end(self, epoch, logs={}):
          x, y, z = self.test_data
          result = model.predict(x)
          calc_test_result(result, y, z)

  class hadamard3(Layer):
      def __init__(self, **kwargs):
          self.supports_masking = True
          super(hadamard3, self).__init__(**kwargs)

      def build(self, input_shape):
          # Create a trainable weight variable for this layer.
          self.output_dim = 500

          self.kernel1 = self.add_weight(name='kernel1', 
                                        shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        trainable=True)
          self.kernel2 = self.add_weight(name='kernel2', 
                                        shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        trainable=True)
          self.kernel3 = self.add_weight(name='kernel3', 
                                        shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        trainable=True)
          self.bias = self.add_weight(name='bias', 
                                        shape=(self.output_dim,),
                                        initializer='zeros',
                                        trainable=True)
          super(hadamard3, self).build(input_shape)  # Be sure to call this somewhere!

      def call(self, x, mask=None):

          def crop(a):
              return a[:, 0:500]
          def crop1(a):
              return a[:, 500:1000]
          def crop2(a):
              return a[:, 1000:1500]
          def output_of_lambda(input_shape):
              return (input_shape[0], 500)
          def output_of_lambda1(input_shape):
              return (input_shape[0], 500)
          def output_of_lambda2(input_shape):
              return (input_shape[0], 500)
          x1=Lambda(crop,output_shape=output_of_lambda)(x)
          x2=Lambda(crop1,output_shape=output_of_lambda1)(x)
          x3=Lambda(crop2,output_shape=output_of_lambda2)(x)

          ai = K.tanh(x1*self.kernel1 + x2*self.kernel2 + x3*self.kernel3 + self.bias)
          return ai

      def compute_output_shape(self, input_shape):
          return (input_shape[0],self.output_dim)
      def compute_mask(self, input, input_mask=None):
          if isinstance(input_mask, list):
              return [None] * len(input_mask)
          else:
              return None



  #################################################################################
  #Level 2: Hfusion3 trimodal

  x_tri=Input(shape=(1500,), name='tri_input')
  fused_1_2_3=hadamard3()(x_tri)
  hfusion_model=Model(x_tri,outputs=fused_1_2_3)

  print("\n hfusion_model")
  hfusion_model.summary()

  #################################################################################
  #Complete Level 2 : Lstm applied on hfusion3 
  main_input= Input(shape=(maxlen,1500,), name='tri_main_input')
  masked = Masking(mask_value =0.0, name='tri_masking')(main_input)
  video_uttr=TimeDistributed(hfusion_model, name='tri_td_1')(masked)
  lstm = GRU(500, activation='tanh', return_sequences = True, dropout=0.35, name='tri_lstm')(video_uttr)
  lstm = Masking(mask_value=0.)(lstm)
  inter = Dropout(0.86, name="tri_dropout1")(lstm)
  concatenate_=Concatenate(name="tri_concat")([video_uttr,inter])
  inter1_tri = TimeDistributed(Dense(450,activation='relu', name='tri_td_2'))(concatenate_)
  inter1_tri = Masking(mask_value=0.)(inter1_tri)
  context_fused_1_2_3 = Dropout(0.86, name="tri_dropout2")(inter1_tri)
  context_fused_1_2_3 = Masking(mask_value=0.)(context_fused_1_2_3)
  predictions = Dense(4, activation='softmax', name='tri_td_3')(context_fused_1_2_3)


  #################################################################################


  model=Model(main_input,outputs=predictions)  # 4 分类？

  print("\n model")
  model.summary()

  optimizer=Adam(lr=0.0001)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal')
                
  early_stopping = EarlyStopping(monitor='val_loss', patience=5)
  #print train_label.shape
  print(train_label[0])
  model.fit(merged_train_data, train_label,
                  epochs=100,
                  batch_size=20,
                  sample_weight=train_mask,
                  shuffle=True, callbacks=[TestCallback((merged_test_data,test_label,test_mask))],
                  validation_split=0.1)

  result = model.predict(merged_test_data)
  calc_test_result(result, test_label, test_mask)  


if __name__ == '__main__':


  Bimodal()

  Trimodal()







