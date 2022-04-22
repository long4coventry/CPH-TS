'''
Main function for CPH.
'''
# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import pandas as pd
from data_loader import data_loader
from CPH import cph
from CPH1 import cph1
from utils import rmse_loss
from math import *
import random
from data_loader1 import data_loader1
import tensorflow as tf

def test_loss(ori_data_x,imputed_data_x,ward_nor_list):
    df = pd.read_csv("./data/Chronic_Diseases_Prevalence_Dataset.csv")
    ward_code_list=list(df['Ward Code'])
    n=0
    y=0
    y_mae=0
    no, dim = ori_data_x.shape
    # print("nuber is:    ", no, dim)
    dim2 = int(dim)
    #print(dim,dim2)       
    A = np.matrix(ori_data_x)
    rank = np.linalg.matrix_rank(A)
    # print(no, "   ",dim,"  ", rank, '  rank:   ****')
    R_original=ori_data_x[:,dim2-1]
    R_result = imputed_data_x[:,dim2-1]

    yy_mae = []
    count = 0
    all = 0

    # bootstrapping the unselected TS-A to the size of the entire regions
    bootstrapping_list = np.random.choice(ward_nor_list, size = no)
    for id in bootstrapping_list:
        result=R_result[id]
        origial=R_original[id]
        #print(id,origial,result)
        if str(origial)!="nan" and origial!=0:
            y=y+pow((origial-result),2)
            n+=1
            y_mae = y_mae + abs(origial - result)
            yy_mae.append(abs(origial - result))
            all = all+1
            if abs(origial - result) < 0.05:
              count = count +1
    RMSE=sqrt(y/n)
    print('qualified percentage is:', count/all, count, all, len(ward_nor_list))
    MAE=y_mae/n
    #print("RMSE:",RMSE)
    #print("MAE:",MAE)
    #print()
    return RMSE, MAE

def main (args,yy):
  '''
  Args:
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse, mae
  '''

  miss_rate = args.miss_rate
  
  cph_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}

  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m, ward_nor_list, data_image = data_loader(miss_rate, yy)

  # Impute missing data
  imputed_data_x, d_pro = cph(miss_data_x, cph_parameters,data_image)

  imputed_data_x, d_pro1 =  cph1(miss_data_x, cph_parameters,data_image)

  a = d_pro[:,0] - d_pro1[:,0]


  

  RMSE, MAE = test_loss(ori_data_x,imputed_data_x,ward_nor_list)

  return RMSE, MAE, a


#  add new instance one by one in an incremental fashion

def main1 (args,yy, index):
  '''
  Args:
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse, mae
  '''

  miss_rate = args.miss_rate
  
  cph_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}

  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m, ward_nor_list, data_image = data_loader1(miss_rate, yy, index)

  # Impute missing data
  imputed_data_x, d_pro = cph(miss_data_x, cph_parameters,data_image)
  # imputed_data_x, d_pro_1 =  cph(miss_data_x, cph_parameters,data_image)
  # a = d_pro-d_pro_1

  # print(a[0,0], '*******@@@*****')
  

  RMSE, MAE = test_loss(ori_data_x,imputed_data_x,ward_nor_list)

  return RMSE, MAE, imputed_data_x[:,0], data_image[0]



if __name__ == '__main__':
    missing_rate = 90 # missing rate
    f = open('result.txt','w')
    for yy in range(2008,2009):
        mmin = 100
        mmin1 = 100
        print("obesity, year:" + str(yy) + "-2017")
        for i in range(10):

            # Inputs for the main function
            parser = argparse.ArgumentParser()
            parser.add_argument(
              '--miss_rate',
              help='missing data probability',
              default=missing_rate/100,

              type=float)
            parser.add_argument(
              '--batch_size',
              help='the number of samples in mini-batch',
              default=483,
              type=int)
            parser.add_argument(
              '--hint_rate',
              help='hint probability',
              default=0.9,
              type=float)
            parser.add_argument(
              '--alpha',
              help='hyperparameter',
              default=100,
              type=float)
            parser.add_argument(
              '--iterations',
              help='number of training interations',
              default=10000,
              type=int)

            args = parser.parse_args()
            print(args)
            #sys.exit(1)
            # Calls main function
            RMSE, MAE, d_pro = main(args,yy)

            if RMSE+MAE < mmin:
                RMSE2= RMSE
                MAE2 = MAE
                mmin = RMSE+MAE

        print("target disease: obesity")
        print("RMSE:", RMSE2)
        print("MAE:", MAE2)
    f.write(str(RMSE2))



    # now add the new labeled instance to the ward_list one by one, totally 20 together
    # check if the new instance is already sampled in the last round


    # print(d_set, "********!!!!!!!!!!!!!!")
#select the top num of instance for 20 times
    acquisited_points = []
    min_set = [] 
    for l in range(30):
      for yy in range(2008,2009):
        mmin = 100
        print("obesity, year:" + str(yy) + "-2017")
        min_value = 0
        min_index = 0
        add_set =[]
        k = 0
        batch = 20
        dist = 0
        dist_list =[]
           
        RMSE, MAE, a, v = main1(args,yy, min_set)
        d_set = a
        dist_list = []
#get the set with the largest disagreement

        for j in range(batch):
            for i in d_set:
              if i == 0:
                continue
              else:
                if abs(i) > min_value and k not in min_set:
                  min_value = abs(i)
                  min_index = k
                  min_set.append(min_index)
                  add_set.append(min_index)
              k = k+1
            
            
            k = 0
        # print(min_set)

# select top num of instance to maximize the S_x
        num = 15 
        for d in add_set:
          # print(v,'*********')
          for i in add_set:
            dist = dist+np.linalg.norm(v[d]-v[i])   
          dist_list.append(dist)
          dist = 0
        final_index = sorted(range(len(dist_list)), key=lambda i: dist_list[i])[-num:] 
        final_index = [add_set[x] for x in final_index]
        print(add_set)
        print(final_index)
      
        acquisited_points = acquisited_points+final_index 

        for i in range(10):

            # Inputs for the main function
            parser = argparse.ArgumentParser()
            parser.add_argument(
              '--miss_rate',
              help='missing data probability',
              default=missing_rate/100,
              type=float)
            parser.add_argument(
              '--batch_size',
              help='the number of samples in mini-batch',
              default=483,
              type=int)
            parser.add_argument(
              '--hint_rate',
              help='hint probability',
              default=0.9,
              type=float)
            parser.add_argument(
              '--alpha',
              help='hyperparameter',
              default=100,
              type=float)
            parser.add_argument(
              '--iterations',
              help='number of training interations',
              default=10000,
              type=int)

            args = parser.parse_args()

            # Calls main function
            
            RMSE, MAE, d_pro, v = main1(args,yy, acquisited_points)
            

            if RMSE+MAE < mmin:
                RMSE2= RMSE
                MAE2 = MAE
                mmin = RMSE+MAE
        print("target disease: obesity")
        print("RMSE:", RMSE2)
        print("MAE:", MAE2)
        f.write(str(RMSE2)+" "+str(MAE2))


