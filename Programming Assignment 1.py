#Name: James Goedmakers
#Class: CAP 5625 - Dr. Michael DeGiorgio
#Assignment: 4
#Due Date: 10/20/2021
#Description: Programming Assignment 1 - Ridge Regression

import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import random

#this is only used in deliverable 6
from sklearn import linear_model

###Class Definitions###

###Function Definitions###

#Reads data from csv into X and y numpy arrays, qualitative values converted to binary
def get_data(fileName):

        temp_x = []
        temp_y = []

        csv_file = open(fileName, "r")
        reader = csv.reader(csv_file)

        #read csv values into temp arrays, qualitative values changed to binary
        next(reader)
        for row in reader:
            if(row[6] == "Male"):
                row[6] = 1
            else:
                row[6] = 0
            
            if(row[7] == "Yes"):
                row[7] = 1
            else:
                row[7] = 0

            if(row[8] == "Yes"):
                row[8] = 1
            else: row[8] = 0

            for i in range(len(row)):
                row[i] = float(row[i])

            temp_x.append(row[0:9])
            temp_y.append(row[9])

        csv_file.close()

        #create numpy arrays using temp
        a = np.array(temp_x, dtype=float)
        b = np.array(temp_y, dtype=float)

        return a, b

#vectorized algorithm, returns beta_hat parameter vector
def vectorize(lamb, x_array, y_array):
    
    #step 1 - choose learning rate alpha and tuning parameter lambda
    alpha = 0.00001
    lam = lamb

    #step 2 - create array X shape (N x p) and array y shape N
    X = x_array
    y = y_array

    #step 3 - randomly initialize parameter vector beta
    beta = []
    for i in range(0,9):
        beta.append(random.uniform(-1,1))
    beta = np.array(beta)

    #step 4-5 - update parameter vector using the specified equation for 10^5 iterations
    for i in range(0,100000):
        beta = beta - (2)*(alpha)*((lam)*(beta)-np.matmul(X.transpose(),(y-np.matmul(X,beta))))

    #step 6 - set the last updated parameter vector as beta_hat
    beta_hat = beta

    return beta_hat

#splits data into training and validation sets size 320 and 80, respectively, i specifies which slice of 80 for valid split
def train_valid_split(x_array, y_array, i):

    x_splits = np.split(x_array, 5)
    y_splits = np.split(y_array, 5)

    x_valid = x_splits[i]
    y_valid = y_splits[i]

    x_train_splits = []
    y_train_splits = []

    for j in range(0,5):
        if(j != i):
            x_train_splits.append(x_splits[j])
            y_train_splits.append(y_splits[j]) 
    
    x_train_splits = np.array(x_train_splits)
    y_train_splits = np.array(y_train_splits)

    x_train = np.concatenate((x_train_splits[0], x_train_splits[1]), axis=0)
    y_train = np.concatenate((y_train_splits[0], y_train_splits[1]), axis=0)

    for i in range(2,4):
        x_train = np.concatenate((x_train, x_train_splits[i]), axis=0)
        y_train = np.concatenate((y_train, y_train_splits[i]), axis=0)


    return x_train, y_train, x_valid, y_valid

#performs 5 fold cross validation and returns the CV error
def cross_valid(X, y, i):

    cv_error = 0
    for j in range(0,5):
        #get training and validation data splits (320 and 80 elements, respectively)
        X_train, y_train, X_valid, y_valid = train_valid_split(X,y,j)

        #standardize X_train and X_valid sets
        X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        X_valid = (X_valid - np.mean(X_valid, axis=0)) / np.std(X_valid, axis=0)

        #center y_train and y_valid sets
        y_train = (y_train - np.mean(y_train, axis=0))
        y_valid = (y_valid - np.mean(y_valid, axis=0))

        betas = vectorize(math.pow(10,i),X_train, y_train)

        squared_error_sum = 0
        for k in range(len(X_valid)):
            squared_error_sum += math.pow((y_valid[k] - np.sum(betas*X_valid[k])),2)

        mean_squared_error = squared_error_sum / 80
        cv_error += mean_squared_error

    cv_error = cv_error / 5
    
    return cv_error

#Deliverable 1 - displays a plot of the different beta_hat values
def deliverable_1():

    #get X array and y array from the csv file
    X, y = get_data("Credit_N400_p9.csv")

    #standardize X
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    #center y
    y = (y - np.mean(y, axis=0))

    #initialize array to hold the beta_hat vectors
    lambdas = []

    #vectorize 7 times for each of the stated lambda values, store the beta_hat vectors in the lambdas array
    for i in range(-2,5):
        lambdas.append(vectorize(math.pow(10,i), X, y))

    #turn into numpy array for easier manipulation
    lambdas = np.array(lambdas)
    
    #plot the beta_hat parameters
    plt.figure()
    plt.axis([-2,4, -500, 500])
    plt.xlabel("Log10(lambda)")
    plt.ylabel("Beta_hat sub j")

    plt.plot([-2,-1,0,1,2,3,4],lambdas[:,0], label="Income")
    plt.plot([-2,-1,0,1,2,3,4],lambdas[:,1], label="Limit")
    plt.plot([-2,-1,0,1,2,3,4],lambdas[:,2], label="Rating")
    plt.plot([-2,-1,0,1,2,3,4],lambdas[:,3], label="Cards")
    plt.plot([-2,-1,0,1,2,3,4],lambdas[:,4], label="Age")
    plt.plot([-2,-1,0,1,2,3,4],lambdas[:,5], label="Education")
    plt.plot([-2,-1,0,1,2,3,4],lambdas[:,6], label="Gender")
    plt.plot([-2,-1,0,1,2,3,4],lambdas[:,7], label="Student")
    plt.plot([-2,-1,0,1,2,3,4],lambdas[:,8], label="Married")

    plt.legend(loc='lower right')

    plt.show()

#Deliverable 2 - plot the 5-fold cross validation error for the seven lambda values
def deliverable_2():
    
    X, y = get_data("Credit_N400_p9.csv")

    #compute cv error for the seven lambda values
    errors = []
    for i in range(-2,5):
        errors.append(cross_valid(X, y, i))

    #plot the cv errors 
    plt.figure()
    plt.xlabel("Log10(lambda)")
    plt.ylabel("CV sub 5 error")
    plt.plot([-2,-1,0,1,2,3,4],errors)
    plt.show()

    return

#Deliverable 3 - indicate the lambda value that generated the lowest cross validation error
def deliverable_3():
    
    X, y = get_data("Credit_N400_p9.csv")

    #compute cv error for the seven lambda values
    errors = []
    for i in range(-2,5):
        errors.append(cross_valid(X, y, i))

    errors = np.array(errors)

    #find minimum cv error
    min = errors[0]
    min_index = 0
    for i in range(len(errors)):
        if min > errors[i]:
            min = errors[i]
            min_index = i - 2

    print("The lambda value of 10^" + str(min_index) + " achieved the lowest cross validation error of: " + str(min))

    return

#Deliverable 4 - retrain using the optimal lambda on the N = 400 dataset and provide the beta_hat parameters
def deliverable_4():

    X, y = get_data("Credit_N400_p9.csv")

    #compute cv error for the seven lambda values
    errors = []
    for i in range(-2,5):
        errors.append(cross_valid(X, y, i))

    errors = np.array(errors)

    #find minimum cv error
    min = errors[0]
    min_index = 0
    for i in range(len(errors)):
        if min > errors[i]:
            min = errors[i]
            min_index = i - 2

    #get X new array and y array from the csv file
    X, y = get_data("Credit_N400_p9.csv")

    #standardize X
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    #center y
    y = (y - np.mean(y, axis=0))

    best_fit = vectorize(math.pow(10,min_index), X, y)
    print("The best fit parameters are: ")
    print(best_fit)

    return

#Deliverable 6 - use sklearn to complete deliverables 1-4 and compare the results
def deliverable_6():
    from sklearn.linear_model import Ridge

    X, y = get_data("Credit_N400_p9.csv")

    #the RidgeCV class includes cross validation, cv=5 folds
    ridge = linear_model.RidgeCV(alphas=(0.01,0.1,1,10,100,1000,10000), cv=5)
    ridge.fit(X,y)

    #displays the optimal beta_hat parameters
    print(ridge.coef_)

    return

###MAIN###
#set random seed for consistency
random.seed(1)

###DELIVERABLES###
#Deliverable 1 - plot the beta_hat vectors
#deliverable_1()

#Deliverable 2 - plot the 5-fold cross validation error
deliverable_2()

#Deliverable 3 - indicate the lambda value that generated the lowest cross validation error
#deliverable_3()

#Deliverable 4 - use the optimal lambda above to retrain on the N = 400 dataset, provide the beta_hat parameters
#deliverable_4()

#Deliverable 5 - provide assignment 1 source code and usage instructions
#All of the source code is in this python file, each of the listed deliverable calls can be commented/uncommented
#to run them individually

#Deliverable 6 - use sklearn to complete deliverables 1-4 and compare the results
#deliverable_6()
#I am unsure how to fulfill the deliverable 1-3 requirements of plots and such. The RidgeCV class from the SKLearn
#library fulfills all of the steps in only a few lines of code, so the individual steps are obscured, and I do not know 
#how to access these points based on the class documentation. However, the end result using the machine learning library
#is still the same as Programming Assignment 1, with an optimal beta_hat vector produced for the given data set. These values
#are significantly different than my own results, so I can only assume that I coded something wrong. Probably something to do
#with the vectorize algorithm, as I had a lot of trouble representing the equation in Python code.
