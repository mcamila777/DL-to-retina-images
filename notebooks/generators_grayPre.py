import os
from keras.utils import np_utils
import numpy as np
from numpy import genfromtxt
import random
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
#Cutomized classes
import retinaMethods as rm
import paths_file

#Train data Generator -> Yields a specific amount of train images with an uniform classes distribution.
def BatchGenerator(batch_size, nb_classes):
    path = paths_file.train_processed_path
    all_imag = [f for f in os.listdir(path) if f.startswith('i')]
    all_label = [f for f in os.listdir(path) if f.startswith('l')]

    #Sort the files names to be according with the labeling file
    rm.sort_nicely(all_imag) 
    rm.sort_nicely(all_label) 


    #LOAD THE DATA (IMAGES AND LABELS)
    #Get the labels from the CSV file
    my_data = genfromtxt(paths_file.train_labels_path, delimiter=',')
    #print(my_data.shape)
    labelsNum = my_data[1:,1] #Remove the first row, and take the second column
    labelsNum = labelsNum.astype('uint8')

    prob = []
    print("Original data distribution:")
    for k in range(5):
        print(np.sum(labelsNum==k)/float(labelsNum.size)*100)
        prob.append(np.sum(labelsNum==k)/float(labelsNum.size))

    prob = np.array(prob)
    n2 = 1./np.min(prob)    #The maximum value that I can have is 1/minimum


    while 1: #In order to the generator can return images while the fit_generator requires it
        print("Training data")
        for i in xrange(len(all_imag)):
            
            #Read set of data
            X_train = np.load(os.path.join(path, all_imag[i]))
            y_train = np.load(os.path.join(path, all_label[i]))

            #-------------------------------------------
            # draw a new sample that ensures a uniform data distribution
            delInd = []

            for k in range(y_train.size):
                if not(np.random.rand() < 1./prob[y_train[k]]/n2): #divide for the maximum to normalize
                    delInd.append(k)

            X_train = np.delete(X_train, delInd , axis=0)
            y_train = np.delete(y_train, delInd, axis=0)
            
            #Validate if is necesary to append some previous images to complete a batch size
            try:
                X_train_old
            except NameError:
                pass#print '\nNOT defined'
            else:
                #print '\nsure, it was defined.'
                X_train = np.append(X_train, X_train_old , axis = 0)
                y_train = np.append(y_train, y_train_old , axis = 0)
                #print('\nShape after append: ' +str( X_train.shape[0]))

            print('\n'+ all_imag[i])
            
            #Validate if were choosen less images than the required by batch_size
            if X_train.shape[0] < batch_size:
                
                X_train_old = X_train
                y_train_old = y_train
                
                del(X_train)
                del(y_train)
                
                #print('\nShape to continue: ' + str( X_train_old.shape[0]))

                continue
            else:    
                
                if X_train.shape[0] > batch_size:
                    #Delete randomly the amount of images that surpasses the batch size
                    dif = X_train.shape[0] - batch_size
                    delInd = random.sample(range(X_train.shape[0]), dif)
                    X_train = np.delete(X_train, delInd , axis=0)
                    y_train = np.delete(y_train, delInd, axis=0)
                
                
                #----------------------------------------
                #Data augmentation
                
                #Transform data with a 50% probabilitie
                
                for i in np.arange(X_train.shape[0]):
                    #Rotate randomly all the images
                    X_train[i] = random_rotation(X_train[i], 360, fill_mode = 'constant', cval=128)
                    
                    #plt.imshow(np.rollaxis(X_train[i],0,3) ) , plt.xticks([]),plt.yticks([]) , plt.show()

                    #Choose a random between 6 numbers, whith the first 3 theres is a transformation but with the rest is not
                    tranf = random.randint(0, 5) 
                    if tranf == 0:
                        #Make horizontal flip
                #         print("Horizontal")
                        X_train[i] =  np.fliplr(X_train[i])
                    elif tranf == 1:
                        #Make vertical flip
                #         print("Vertical")
                        X_train[i] =  np.flipud(X_train[i])
                    elif tranf == 2:
                #         print("Horizontal-Vertical")
                        #make horizontal - vertical flip
                        X_train[i] =  np.fliplr(X_train[i])
                        X_train[i] =  np.flipud(X_train[i])

                #-------------------------------------------

                # convert class vectors to binary class matrices
                Y_train = np_utils.to_categorical(y_train, nb_classes)
                
                
                try:
                    del(X_train_old)
                    del(y_train_old)
                except NameError:
                    pass#print '\nNot deleted because not created'

                print('\nAmount of images before return: '+str( X_train.shape[0]))
                
                yield (X_train, Y_train)
                

                
#-------------------------------------------------------------
#Random rotation Keras methods
import scipy.ndimage as ndi

def random_rotation(x, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x
#-------------------------------------------------------------

#Train data Generator -> Yields a specific amount of test images with an uniform classes distribution.
def BatchValidationGenerator(batch_size, nb_classes):
    path = paths_file.test_processed_path
    all_imag = [f for f in os.listdir(path) if f.startswith('i')]
    all_label = [f for f in os.listdir(path) if f.startswith('l')]

    #Sort the files names to be according with the labeling file
    rm.sort_nicely(all_imag) 
    rm.sort_nicely(all_label) 


    #LOAD THE DATA (IMAGES AND LABELS)
    #Get the labels from the CSV file
    my_data = genfromtxt(paths_file.test_labels_path, delimiter=',')
    #print(my_data.shape)
    labelsNum = my_data[1:,1] #Remove the first row, and take the second column
    labelsNum = labelsNum.astype('uint8')

    prob = []
    print("Original data distribution:")
    for k in range(5):
        print(np.sum(labelsNum==k)/float(labelsNum.size)*100)
        prob.append(np.sum(labelsNum==k)/float(labelsNum.size))

    prob = np.array(prob)
    n2 = 1./np.min(prob)    #The maximum value that I can have is 1/minimum


    while 1: #In order to the generator can return images while the fit_generator requires it
        print("Validation Data")
        for i in xrange(int(len(all_imag)*0.3)): #take only 30% of the training data set for validation
            
            #Read set of data
            X_train = np.load(os.path.join(path, all_imag[i]))
            y_train = np.load(os.path.join(path, all_label[i]))

            #-------------------------------------------
            # draw a new sample that ensures a uniform data distribution
            delInd = []

            for k in range(y_train.size):
                if not(np.random.rand() < 1./prob[y_train[k]]/n2): #divide for the maximum to normalize
                    delInd.append(k)

            X_train = np.delete(X_train, delInd , axis=0)
            y_train = np.delete(y_train, delInd, axis=0)
            
            #Validate if is necesary to append some previous images to complete a batch size
            try:
                X_train_old
            except NameError:
                pass#print '\nNOT defined'
            else:
                #print '\nsure, it was defined.'
                X_train = np.append(X_train, X_train_old , axis = 0)
                y_train = np.append(y_train, y_train_old , axis = 0)
                #print('\nShape after append: ' +str( X_train.shape[0]))

            print('\nV: '+ all_imag[i])
            
            #Validate if were choosen less images than the required by batch_size
            if X_train.shape[0] < batch_size:
                
                X_train_old = X_train
                y_train_old = y_train
                
                del(X_train)
                del(y_train)
                
                #print('\nShape to continue: ' + str( X_train_old.shape[0]))

                continue
            else:    
                
                if X_train.shape[0] > batch_size:
                    #Delete randomly the amount of images that surpasses the batch size
                    dif = X_train.shape[0] - batch_size
                    delInd = random.sample(range(X_train.shape[0]), dif)
                    X_train = np.delete(X_train, delInd , axis=0)
                    y_train = np.delete(y_train, delInd, axis=0)
                    
                for i in np.arange(X_train.shape[0]):
                    #Rotate randomly all the images
                    X_train[i] = random_rotation(X_train[i], 360, fill_mode = 'constant', cval=128)
                                
                #-------------------------------------------

                # convert class vectors to binary class matrices
                Y_train = np_utils.to_categorical(y_train, nb_classes)
                
                
                try:
                    del(X_train_old)
                    del(y_train_old)
                except NameError:
                    pass#print '\nNot deleted because not created'

                #print('\nAmount of images before return: '+str( X_train.shape[0]))
                
                yield (X_train, Y_train)
                
                
#Test data Generator --> Yields a specific amount of test images without modifying the classes distribution
def ValidationGenerator(batch_test ,nb_classes):
    
    #Load validation data (Taken from test data)  
    #10% of the total amount of trainind data .. aprox 3k images
    testPath = paths_file.test_processed_path

    all_imag_test = [f for f in os.listdir(testPath) if f.startswith('i')]
    all_label_test = [f for f in os.listdir(testPath) if f.startswith('l')]

    #Sort the files names to be according with the labeling file
    rm.sort_nicely(all_imag_test) 
    rm.sort_nicely(all_label_test) 

    #Take the las 40 packages (~100imges/package)... 
    #Choosen this because this have a classes distribution similar to the test distribution
    all_imag_test = all_imag_test[-40:] 
    all_label_test = all_label_test[-40:] 
    
    
    while 1: #Infinite loop to repeat the for loop as the generator caller require it
        for i in xrange(len(all_imag_test)):
            
            #print(all_imag_test[i])
            #Load a package of images (~100imges)
            X_test_package = np.load(os.path.join(testPath, all_imag_test[i]))
            y_test_package = np.load(os.path.join(testPath, all_label_test[i]))

            #Return the images of this package in groups of batch_test
            stops = np.arange(0, X_test_package.shape[0], batch_test)

            for i in np.arange(1, len(stops)):
                #'new group'
                X_test = X_test_package[stops[i-1]:stops[i]]
                y_test = y_test_package[stops[i-1]:stops[i]]                
                
                
                #-------------------------------------------
                for i in np.arange(X_test.shape[0]):
                    #Rotate randomly all the images
                    X_test[i] = random_rotation(X_test[i], 360, fill_mode = 'constant', cval=128)
                    
                #-------------------------------------------
                
                # convert class vectors to binary class matrices
                Y_test = np_utils.to_categorical(y_test, nb_classes)
        
                #print X_test.shape
                
                #Return the required amount of images according with the batch size
                yield (X_test, Y_test)
                
#Test data Generator --> Yields a specific amount of images without modifying the classes distribution
              
def OriginalDataGenerator(batch_test ,nb_classes, test = 0):
    if test == 1:
        #Load validation data (Taken from test data)  
        #10% of the total amount of trainind data .. aprox 3k images
        testPath = paths_file.test_processed_path 
        print("Using testing data")
    else:
        testPath =  paths_file.train_processed_path
        print("Using training data")

    all_imag_test = [f for f in os.listdir(testPath) if f.startswith('i')]
    all_label_test = [f for f in os.listdir(testPath) if f.startswith('l')]
    #Sort the files names to be according with the labeling file
    rm.sort_nicely(all_imag_test) 
    rm.sort_nicely(all_label_test) 
    
    if test ==1:
        #Take the las 40 packages (~100imges/package)... 
        #Choosen this because this have a classes distribution similar to the test distribution
        all_imag_test = all_imag_test[-40:] 
        all_label_test = all_label_test[-40:] 
        
    
    while 1: #Infinite loop to repeat the for loop as the generator caller require it
        for i in xrange(len(all_imag_test)):
            
            #print(all_imag_test[i])
            #Load a package of images (~100imges)
            X_test_package = np.load(os.path.join(testPath, all_imag_test[i]))
            y_test_package = np.load(os.path.join(testPath, all_label_test[i]))

            #Return the images of this package in groups of batch_test
            stops = np.arange(0, X_test_package.shape[0], batch_test)

            for i in np.arange(1, len(stops)):
                #'new group'
                X_test = X_test_package[stops[i-1]:stops[i]]
                y_test = y_test_package[stops[i-1]:stops[i]]                
               
                #----------------------------------------
                #Data augmentation
                
                #Transform data with a 50% probabilitie
                for i in np.arange(X_test.shape[0]):
                    #Rotate randomly all the images
                    X_test[i] = random_rotation(X_test[i], 360, fill_mode = 'constant', cval=128)

                    #plt.imshow(np.rollaxis(X_test[i],0,3) ) , plt.xticks([]),plt.yticks([]) , plt.show()

                    #Choose a random between 6 num, whith the first 3 theres is a transformation but with the rest is not
                    tranf = random.randint(0, 5) 
                    if tranf == 0:
                        #Make horizontal flip
                        #         print("Horizontal")
                        X_test[i] =  np.fliplr(X_test[i])
                    elif tranf == 1:
                        #Make vertical flip
                        #         print("Vertical")
                        X_test[i] =  np.flipud(X_test[i])
                    elif tranf == 2:
                        #         print("Horizontal-Vertical")
                        #make horizontal - vertical flip
                        X_test[i] =  np.fliplr(X_test[i])
                        X_test[i] =  np.flipud(X_test[i])

                #----------------------------------------
                
                # convert class vectors to binary class matrices
                Y_test = np_utils.to_categorical(y_test, nb_classes)
        
                #print X_test.shape
                
                #Return the required amount of images according with the batch size
                yield (X_test, Y_test)
            