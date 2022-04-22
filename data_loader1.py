'''
Data loader for Chronic Diseases Prevalence Dataset
'''

# Necessary packages
import os
import numpy as np
import pandas as pd
from utils import binary_sampler
# from tensorflow import keras 
from tensorflow.keras.datasets import mnist
import sys

def data_loader1(miss_rate, yy, add_index):
  diease_list = ['Obesity Prevalence', 'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence']
  year = yy
  diease_select_list = 1 # target disease
  diease_select_list2 = 0
  diease_select_list3 = 2

  # year = 2009
  N1 = 483
  N3 = 2017 - year + 1
  N33 = 2017 - year + 1
  data_x = np.ones((N1, N3), dtype='float64')
  data_m = np.ones((N1, N3), dtype='float64')

  data_tensor = np.ones((483, 2017 - year + 1, 3), dtype='float64')

  for y in range(year,2017+1):
    df = pd.read_csv("./data/Chronic_Diseases_Prevalence_Dataset.csv")
    ward_code_list=list(df['Ward Code'])
    # print(list(df))
    df1 = df[diease_list[diease_select_list]+"_"+str(y)]
    data_x[:,y - year] = df1.values

  for y in range(year, 2017 + 1):
    df = pd.read_csv("./data/Chronic_Diseases_Prevalence_Dataset.csv")
    df1 = df[diease_list[diease_select_list] + "_" + str(y)]
    data_tensor[:,y - year,0] = df1.values

    df2 = df[diease_list[diease_select_list2]+"_"+str(y)]
    data_tensor[:,y - year,1] = df2.values

    df3 = df[diease_list[diease_select_list3]+"_"+str(y)]
    data_tensor[:,y - year,2] = df3.values

  miss_data_x = data_x.copy()

  ward_number = int(N1 * (100 - miss_rate*100) / 100)  # select wards number

  for y in range(N33 - 1, N33):
    data_year = data_x[:,y]
    ward_list = []  # training set
    ward_nor_list = []  # testing set
    num = 0
    df_ward = pd.read_csv("./data/Variance_2008_2017_"+diease_list[diease_select_list]+"_NORMALIZE.csv")
    df_diease = pd.read_csv("./data/Ward_code_list.csv")
    ward_code_old = list(df_diease['Ward Code'])
    ward_var = list(df_ward["Ward_id_" + str(year) + "_" + str(2017)])
    iii = 0
    while num < ward_number:
      id = ward_var[iii]
      iii += 1
      ward_code = ward_code_old[id]
      if ward_code in ward_code_list:
        index1 = ward_code_list.index(ward_code)

        diease_rate = data_year[index1]
        
        if diease_rate!=0:
          num += 1
          ward_list.append(index1)

    # print(ward_list,'33333333333333333333333333')
    for min_item in add_index:
      # print(min_item, iii, index1, '444444444444444444444444444')
      id = ward_var[min_item]
      ward_code = ward_code_old[id]
      if ward_code in ward_code_list:
        index1 = ward_code_list.index(ward_code)
        diease_rate = data_year[index1]
        if diease_rate != 0 and index1 not in ward_list:
          ward_list.append(index1)
        
   
    print(len(ward_list),"ward list length is :           ")
    for i in range(N1):
      if i in ward_list:
        continue
      ward_nor_list.append(i)
      data_m[i,N33-1] = 0
      data_tensor[i,-1,0] = 0

  for y in range(N33 - 1, N33):
    data_year = data_tensor[:, y, 1]

    ward_list2 = []
    ward_nor_list2 = []
    num = 0
    df_ward = pd.read_csv("./data/Variance_2008_2017_"+diease_list[diease_select_list2]+"_NORMALIZE.csv")

    df_diease = pd.read_csv("./data/Ward_code_list.csv")
    ward_code_old = list(df_diease['Ward Code'])

    ward_var = list(df_ward["Ward_id_" + str(year) + "_" + str(2017)])
    iii = 0
    while num < ward_number:
      id = ward_var[iii]
      iii += 1
      ward_code = ward_code_old[id]
      if ward_code in ward_code_list:
        index1 = ward_code_list.index(ward_code)
        diease_rate = data_year[index1]
        if diease_rate!=0:
          num += 1
          ward_list2.append(index1)
    #ard_list2.append(add_index)
    # for min_item in add_index:
    #   if min_item not in ward_list2: 
    #     ward_list2.append(min_item)
   
    for i in range(N1):
      if i in ward_list2:
        continue
      ward_nor_list2.append(i)
      data_tensor[i,-1,1] = 0
  for y in range(N33 - 1, N33):
    data_year = data_tensor[:, y, 2]
    ward_list3 = []
    ward_nor_list3 = []
    num = 0
    df_ward = pd.read_csv("./data/Variance_2008_2017_"+diease_list[diease_select_list3]+"_NORMALIZE.csv")
    df_diease = pd.read_csv("./data/Ward_code_list.csv")
    ward_code_old = list(df_diease['Ward Code'])

    ward_var = list(df_ward["Ward_id_" + str(year) + "_" + str(2017)])
    iii = 0
    while num < ward_number:
      id = ward_var[iii]
      iii += 1
      ward_code = ward_code_old[id]
      if ward_code in ward_code_list:
        index1 = ward_code_list.index(ward_code)
        diease_rate = data_year[index1]
        #print(diease_rate)
        if diease_rate!=0:
          num += 1
          ward_list3.append(index1)
   #ward_list3.append(add_index)
    # for min_item in add_index:
    #   if min_item not in ward_list3: 
    #     ward_list3.append(min_item)
    # print(len(add_index),len(ward_list),len(ward_nor_list),' ww22222222222222222222')
    for i in range(N1):
      if i in ward_list3:
        continue
      ward_nor_list3.append(i)
      data_tensor[i,-1,2] = 0

  miss_data_x[data_m == 0] = np.nan
  data_image = data_tensor.reshape(1, 483, 2017 - year + 1, 3)
  #print("Data shape：",data_image.shape)

  return data_x, miss_data_x, data_m, ward_nor_list, data_image

