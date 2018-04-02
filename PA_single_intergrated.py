import numpy as np
from pylab import *
import csv


def PA(C, loss, xn):
    return loss/xn

def PA1(C, loss, xn):
    return min([C, loss/xn])

def PA2(C, loss, xn):
    return loss/(xn+1/(2*C))

def PA_single(X, label, C, num_loop):
    
    D = len(X[0])
    w = np.zeros(D+1)
    res = []
    for loop in range(num_loop):
        for idx, x in enumerate(X):
            x = map(lambda y: float(y),x)
            x = concatenate((x,[1]))
            yh = label[idx]*np.dot(w, x)
            if yh<1:
                loss = 1-yh
                xn = np.sum(x**2)
                tau = PA1(C, loss, xn)
                w = w+label[idx]*tau*x
        if 0<loop:
            res.append(sum((w-wd)**2))
        wd = np.copy(w)
    return w, res

def PA_Single_test(X, label,w):

    D = len(X[0])
   
    res = []
    accu=0;
    inaccu=0;


    for idx, x in enumerate(X):
        x = map(lambda y: float(y),x)
        x = concatenate((x,[1]))
        yh = label[idx]*np.dot(w, x)

        if(yh>=0):
            accu+=1
        else:
            inaccu+=1
               
    print "accurate : ",accu," Inaccurate : ",inaccu;
    print "accuracy : ",(float(accu)/233.0)*100,"%";
    
    
with open('breast_cancer_data.csv','rb') as data:
    reader = csv.reader(data)
    
    input_train=[]
    input_test=[]
    output_train=[]
    output_test=[]
    iterator=1;
    for row in reader:
        if(iterator%3==0):
            
            input_test.append(row[1:10]);
            if(row[10]=="2"):
                output_test.append(-1);
            
            else:
                output_test.append(1);
        else:
            input_train.append(row[1:10]);
            if(row[10]=="2"):
                output_train.append(-1);
            
            else:
                output_train.append(1);
        iterator+=1;
            
            
     



    num_loop = 10
    C = 1

    w, res = PA_single(input_train, output_train, C, num_loop)
    PA_Single_test(input_test, output_test, w)

