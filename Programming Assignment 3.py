#Name: James Goedmakers
#Class: CAP 5625 - Dr. Michael DeGiorgio
#Assignment: 3
#Due Date: 12/3/2021
#Description: Programming Assignment 3 - Logistic Regression

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

        #read csv values into temp arrays, classification values changed to binary
        #African,  European,  East  Asian,  Oceanian,  or  Native  American
        next(reader)
        for row in reader:
            if(row[10] == "African"):   temp_y.append([1,0,0,0,0])
            elif(row[10] == "European"):   temp_y.append([0,1,0,0,0])
            elif(row[10] == "EastAsian"):   temp_y.append([0,0,1,0,0])
            elif(row[10] == "Oceanian"):   temp_y.append([0,0,0,1,0])
            elif(row[10] == "NativeAmerican"):   temp_y.append([0,0,0,0,1])
            else: temp_y.append([0,0,0,0,0])            

            temp_x.append(row[0:10])

        csv_file.close()

        #create numpy arrays using temp
        a = np.array(temp_x, dtype=float)
        b = np.array(temp_y, dtype=float)

        return a, b

#vectorized algorithm, returns beta_hat parameter vector
def vectorize(lamb, x_array, y_array):
    
    #step 1 - choose learning rate alpha and tuning parameter lambda
    alpha = 0.00001
    lamb = lamb

    #step 2 - create array X shape (N x p) and array y shape N
    X = x_array
    Y = y_array

    #step 3: initialize the (p+1) x K dimensional parameter matrix to all zeros
    B = np.zeros((len(X[0]),len(Y[0])))

    for i in range(0,10000):
        #step 4: compute N x K unnormalized class probability matrix exp(XB)
        U = np.matmul(X,B)

        #exponentiate each element
        for i in range(len(U)):
            for j in range(len(U[0])):
                U[i,j] = math.exp(U[i,j])

        #step 5: compute N x K normalized class probability matrix P 
        P = np.zeros((len(X),len(Y[0])))

        for i in range(len(P)):
            for j in range(len(P[0])):
                P[i,j] = U[i,j] / np.sum(U[i])

        #step 6: generate (p+1) x K intercept matrix Z
        Z = np.zeros((len(X[0]),len(Y[0])))
        Z[0] = B[0]

        #step 7: update the parameter matrix as 𝐁∶=𝐁+𝛼[𝐗𝑇(𝐘−𝐏)−2𝜆(𝐁−𝐙)] 
        B = B + alpha*(np.matmul(np.transpose(X),(Y-P)) - 2*lamb*(B-Z))

    #step 8: repeat steps 4 to 7 for 10,000 iterations

    #step 9: set the lasted updated parameter matrix as beta_hat
    beta_hat = B

    return beta_hat

#splits data into training and validation sets, i specifies which slice for valid split
def train_valid_split(x_array, y_array, i):

    x_valid = x_array[(i*36):(i*36)+36]
    y_valid = y_array[(i*36):(i*36)+36]
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)

    x_temp = np.delete(x_array, np.s_[i:i+36], axis=0)
    y_temp = np.delete(y_array, np.s_[i:i+36], axis=0)

    x_train = x_temp
    y_train = y_temp
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train, x_valid, y_valid

#performs 5 fold cross validation and returns the CV error
def cross_valid(X, y, i):

    cv_error = 0
    for j in range(0,5):
        #get training and validation data splits (147 and 36 elements, respectively)
        X_train, y_train, X_valid, y_valid = train_valid_split(X,y,j)

        #standardize X_train and X_valid sets
        X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        X_valid = (X_valid - np.mean(X_valid, axis=0)) / np.std(X_valid, axis=0)

        #center y_train and y_valid sets
        y_train = (y_train - np.mean(y_train, axis=0))
        y_valid = (y_valid - np.mean(y_valid, axis=0))

        print(X_train)
        print(y_train)
        betas = vectorize(math.pow(10,i),X_train, y_train)
        print(betas)
        i_sum = 0
        for i in range(len(X_valid)):
            k_sum = 0
            for k in range(0,5):
                #there is some issue occurring with this equation
                k_sum += (y_valid[i,k] * math.log10(np.sum(betas[:,k]*X_valid[i])),2)
            i_sum += k_sum

        cross_entropy = i_sum / len(X_valid) * -1
        cv_error += cross_entropy

    cv_error = cv_error / 5
    
    return cv_error

#Deliverable 1 - displays a plot of the different beta_hat values
def deliverable_1():
    #step 1: set alpha 
    alpha = 0.00001

    #step 2: generate N x (p+1) matrix X standardized and centered and N x K indicator response matrix
    X, Y = get_data("TrainingData_N183_p10.csv")

    #standardize X
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    #append 1 as first element of each row
    ones = np.ones((len(X),1))
    X = np.concatenate((ones, X), axis=1)

    #initialize array to hold the beta_hat vectors
    lambdas = []

    #vectorize 9 times for each of the stated lambda values, store the beta_hat matrices in the lambdas array
    for i in range(-4,5):
        lambdas.append(vectorize(math.pow(10,i), X, Y))

    #turn into numpy array for easier manipulation
    lambdas = np.array(lambdas)

    #one plot for each of the K=5 ancestry classes
    for x in range(0,5):    
        #plot the beta_hat parameters
        plt.figure()
        plt.title("K=" + str(x+1))
        plt.axis([-4,4, -2, 2])
        plt.xlabel("Log10(lambda)")
        plt.ylabel("Beta_hat sub jk")

        plt.plot([-4,-3,-2,-1,0,1,2,3,4],lambdas[:,0,x], label="P1")
        plt.plot([-4,-3,-2,-1,0,1,2,3,4],lambdas[:,1,x], label="P2")
        plt.plot([-4,-3,-2,-1,0,1,2,3,4],lambdas[:,2,x], label="P3")
        plt.plot([-4,-3,-2,-1,0,1,2,3,4],lambdas[:,3,x], label="P4")
        plt.plot([-4,-3,-2,-1,0,1,2,3,4],lambdas[:,4,x], label="P5")
        plt.plot([-4,-3,-2,-1,0,1,2,3,4],lambdas[:,5,x], label="P6")
        plt.plot([-4,-3,-2,-1,0,1,2,3,4],lambdas[:,6,x], label="P7")
        plt.plot([-4,-3,-2,-1,0,1,2,3,4],lambdas[:,7,x], label="P8")
        plt.plot([-4,-3,-2,-1,0,1,2,3,4],lambdas[:,8,x], label="P9")
        plt.plot([-4,-3,-2,-1,0,1,2,3,4],lambdas[:,9,x], label="P10")

        plt.legend(loc='lower right')

        plt.show()

#Deliverable 2 - plot the 5-fold cross validation error for the seven lambda values
def deliverable_2():
    
    X, Y = get_data("TrainingData_N183_p10.csv")

    #compute cv error for the nine lambda values
    errors = []
    for i in range(-4,5):
        errors.append(cross_valid(X, Y, i))

    #plot the cv errors 
    plt.figure()
    plt.xlabel("Log10(lambda)")
    plt.ylabel("CV sub 5 error")
    plt.plot([-4,-3,-2,-1,0,1,2,3,4],errors)
    plt.show()

    return

#Deliverable 3 - indicate the lambda value that generated the lowest cross validation error
def deliverable_3():
    
    X, Y = get_data("TrainingData_N183_p10.csv")

    #compute cv error for the nine lambda values
    errors = []
    for i in range(-4,5):
        errors.append(cross_valid(X, Y, i))

    #find minimum cv error
    min = errors[0]
    min_index = 0
    for i in range(len(errors)):
        if min > errors[i]:
            min = errors[i]
            min_index = i - 2

    print("The lambda value of 10^" + str(min_index) + " achieved the lowest cross validation error of: " + str(min))

    return

#Deliverable 4 - retrain using the optimal lambda on the N = 183 dataset and provide the beta_hat parameters
def deliverable_4():

    X, Y = get_data("TrainingData_N183_p10.csv")

    #compute cv error for the nine lambda values
    errors = []
    for i in range(-4,5):
        errors.append(cross_valid(X, Y, i))

    #find minimum cv error
    min = errors[0]
    min_index = 0
    for i in range(len(errors)):
        if min > errors[i]:
            min = errors[i]
            min_index = i - 2

    #get X new array and y array from the csv file
    X, Y = get_data("TrainingData_N183_p10.csv")

    #center Y
    Y = (Y - np.mean(Y, axis=0))

    best_fit = vectorize(math.pow(10,min_index), X, Y)
    print("The best fit parameters are: ")
    print(best_fit)

    return

#Deliverable 7 - use sklearn to complete deliverables 1-4 and compare the results
def deliverable_7():
    from sklearn.linear_model import LogisticRegression

    X, Y = get_data("TrainingData_N183_p10.csv")

    #the RidgeCV class includes cross validation, cv=5 folds
    log = linear_model.LogisticRegression(alphas=(0.01,0.1,1,10,100,1000,10000), cv=5)
    log.fit(X,Y)

    #displays the optimal beta_hat parameters
    print(log.coef_)

    return

###MAIN###
#set random seed for consistency
random.seed(1)

###DELIVERABLES###
#Deliverable 1 - plot the beta_hat parameters for each K = 5
#deliverable_1()

#Deliverable 2 - plot the 5-fold cross validation error
deliverable_2()
#Deliverable 3 - indicate the lambda value that generated the lowest cross validation error
#deliverable_3()
#Deliverable 4 - use the optimal lambda above to retrain on the N = 183 dataset, provide the beta_hat parameters
#deliverable_4()
#Deliverable 5 - provide assignment 1 source code and usage instructions
#All of the source code is in this python file, each of the listed deliverable calls can be commented/uncommented
#to run them individually
#Deliverable 7 - use sklearn to complete deliverables 1-4 and compare the results
#deliverable_7()

