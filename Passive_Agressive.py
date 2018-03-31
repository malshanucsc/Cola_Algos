
import sys
import numpy as np
from pylab import *
import csv

def PA(C,l_t,x_t):
    
    x = np.array(x_t);
    
    return l_t/(np.linalg.norm(x)**2)
def PA1(C, l_t, x_t):
    x = np.array(x_t);
    #print(C," : " , (l_t/(np.linalg.norm(x)**2)))
    return min(C,(l_t/(np.linalg.norm(x)**2)));
def PA2(C, l_t, x_t):
    
    x = np.array(x_t);
    
    return l_t/(((np.linalg.norm(x))**2)+(1.0/(2.0*C)));


def pA_Training(loop,input_vector,output_vector,C):

    X=input_vector;
    Y=output_vector;
    
    
    w=np.array([0.00000000000000000000]*(len(X[0])),np.float64)
    
    
    
    
    for i in range(loop):
        
        for j in range(0 , len(X)):
            
            x_t=X[j];
            y_t=Y[j];
            

            x_t = map(lambda x: float(x),x_t)
            
            signed_t=y_t*np.dot(w,x_t);
            
            loss_t=max(0,1-signed_t);

            if(loss_t>0):
                Tau_t=PA(C,loss_t,x_t);
                Tau_t1=PA1(C,loss_t,x_t);
                Tau_t2=PA2(C,loss_t,x_t);
                #print Tau_t," , ",Tau_t1," , ",Tau_t2
               
                
                #print x_t," : " ,Tau_t, " : ",y_t;
                w_delta=[k * Tau_t * y_t for k in x_t];
                
                #print(w_delta);
                
                for l in range(0,len(w)):
                    #print(w[i]," : ",w_delta[i])
                   
                    w[l]=np.around((w[l]),decimals=20)+w_delta[l]
                    
                #w=w+w_delta;
                

    
    return w
            
        
def pA_Testing(w,input_vector,output_vector):

    X=input_vector;
    Y=output_vector;
    wrong_cal=0;
    correct_cal=0;
    loop=len(Y);

    for i in range(loop):
        x_t=X[i];
        y_t=Y[i];

        x_t = map(lambda x: float(x),x_t)
        
        signed_t=y_t*np.dot(w,x_t);
        

        loss_t=max(0,1-signed_t);
        #print
        #print (signed_t," : ",y_t)
        if(signed_t>=0):
            correct_cal+=1
        else:
            wrong_cal+=1
            
    print(correct_cal, ":", wrong_cal   ) ;
    
with open('data.csv','rb') as data:
    reader = csv.reader(data)
    
    X_train=[]
    X_test=[]

    Y_train=[]
    Y_test=[]
    i=1;
    for row in reader:
        
        if(i%3==0):
            X_test.append(row[1:10]);
            if(row[10]=="2"):
                Y_test.append(-1);
            
            elif(row[10]=="4"):
            
                Y_test.append(1);
        else:
            X_train.append(row[1:10]);
            if(row[10]=="2"):
                Y_train.append(-1);
            
            elif(row[10]=="4"):
            
                Y_train.append(1);
        i+=1;
        


    
C=1.0;
loop=10;



w=pA_Training(loop,X_train,Y_train,C);



accuracy= pA_Testing(w,X_test,Y_test);


#print(w)

