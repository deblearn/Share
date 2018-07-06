import numpy as Math
import numpy as np
import time
import itertools
import random
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
import scipy.io
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from collections import Counter
import math


if __name__ == "__main__":

    final_output = np.zeros((5,4));

    for iii in range(1,6):

        x1 = scipy.io.loadmat('k_5_mean_windows_state_' + str(iii) + '.mat');
        x = x1['meanValues']

        y1 = scipy.io.loadmat('k_5_label_state_' + str(iii) + '.mat');
        y = y1['subLabel']

        # split data into four parts. At first first 25% will be test sets and rest one will be training. Then next 25% will be test cases and
        # rest one will be training sets
        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x, y, test_size=0.25, random_state=0, stratify=y)
        x_train_2_demo, x_test_2, y_train_2_demo, y_test_2 = train_test_split(x_train_1, y_train_1, test_size=0.33, random_state=0, stratify=y_train_1)
        x_train_3_demo, x_test_3, y_train_3_demo, y_test_3 = train_test_split(x_train_2_demo, y_train_2_demo, test_size=0.50, random_state=0, stratify=y_train_2_demo)

        x_test_4 = x_train_3_demo; y_test_4 = y_train_3_demo;
        dimensions = 1081;


        a1, b1 = np.shape(x_train_2_demo); c1, d1 = np.shape(x_test_1); total_length = a1 + c1;
        x_train_2 = np.zeros((total_length, dimensions)); y_train_2 =  np.zeros((total_length,1));

        x_train_2[0:a1, :] = x_train_2_demo; x_train_2[a1:total_length, :] = x_test_1;
        y_train_2[0:a1] = y_train_2_demo; y_train_2[a1:total_length] = y_test_1;



        a1, b1 = np.shape(x_train_3_demo); c1, d1 = np.shape(x_test_1); e1, f1 = np.shape(x_test_2); total_length = a1 + c1 + e1;
        x_train_3 = np.zeros((total_length, dimensions)); y_train_3 = np.zeros((total_length, 1));

        x_train_3[0:a1, :] = x_train_3_demo; x_train_3[a1:(a1+c1), :] = x_test_1; x_train_3[(a1+c1):total_length, :] = x_test_2;
        y_train_3[0:a1] = y_train_3_demo; y_train_3[a1:(a1+c1)] = y_test_1; y_train_3[(a1+c1):total_length] = y_test_2;


        a1, b1 = np.shape(x_test_1); c1, d1 = np.shape(x_test_2); e1, f1 = np.shape(x_test_3); total_length = a1 + c1 + e1;
        x_train_4 = np.zeros((total_length, dimensions)); y_train_4 = np.zeros((total_length, 1));

        x_train_4[0:a1, :] = x_test_1; x_train_4[a1:(a1 + c1), :] = x_test_2; x_train_4[(a1 + c1):total_length, :] = x_test_3;
        y_train_4[0:a1] = y_test_1; y_train_4[a1:(a1 + c1)] = y_test_2; y_train_4[(a1 + c1):total_length] = y_test_3;

        print("y_test1: %s y_test2: %s y_test3: %s  y_test4: %s" % (len(y_test_1), len(y_test_2),len(y_test_3),len(y_test_4)))
        print("y_train_1: %s y_train_2: %s y_train_3: %s  y_train_4: %s" % ( len(y_train_1), len(y_train_2), len(y_train_3), len(y_train_4)))



        # K-Fold for n_splits=3, n_repeats=10
        rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=36851234)

        first = -1;
        for iteration in range(0,4):
            if(iteration==0):
                x = x_train_1; y = y_train_1;
                x_test = x_test_1; y_test = y_test_1;

            if(iteration==1):
                x = x_train_2; y = y_train_2;
                x_test = x_test_2; y_test = y_test_2;

            if (iteration == 2):
                x = x_train_3; y = y_train_3;
                x_test = x_test_3; y_test = y_test_3;

            if (iteration == 3):
                x = x_train_4; y = y_train_4;
                x_test = x_test_4; y_test = y_test_4;


            c_value = np.arange(-3, -2, 0.01)
            sum = 0; max_accuracy = -1; first = -1;
            accuracy_Array = np.zeros((30,len(c_value)))

            for train_index, test_index in rskf.split(x, y):
                X_train, X_test = x[train_index], x[test_index]
                Y_train, Y_test = y[train_index], y[test_index]

                # Run one fold for different C values
                first = first + 1;
                j = 0; max_value = np.zeros(5);
                list1 = []

                for i in c_value:
                    val = math.pow(2, i)
                    clf = svm.SVC(kernel='linear', C=val).fit(X_train, Y_train)
                    accuracy = clf.score(X_test, Y_test)

                    accuracy_Array[first,j] = accuracy;
                    j = j + 1


            mean_Value = np.mean(accuracy_Array, axis=0)
            ind = np.argmax(mean_Value)

            optimal_c = math.pow(2, c_value[ind])

            #print('Accuracy in Iteration:', iteration)
            print('Optimal_C: ',optimal_c)

            clf = svm.SVC(kernel='linear', C= optimal_c).fit(x, y)
            final_accuracy = clf.score(x_test, y_test)
            #print("Accuracy: ", final_accuracy)
            final_output[(iii-1),iteration] = final_accuracy;

            #print(final_output)
        print('\n')

    print(final_output)
    scipy.io.savemat('/tmp/out.mat', mdict={'State_1_accuracy': final_output})
    print('Simulation done')




