## Retina_proyect
Deep learning applied to human retina images for medical diagnosis support:

Use of digital image processing, machine learning tools and libraries such as Keras to develop a system to classify human retina images depending of the diabetic retinopathy signs on the image. The classification is made by a convolutional neural network implemented through python programming language.

## FOLDERS EXPLANATION

* Item 1
* Item 2
  * Sub Item 1
  * Sub Item 2

* Kaggle-data-labels: Train and test CVS files with the kaggle’s data labels.
* models-weights:
  * In the callBackModel folder are saved the weights for the two bests models models that were trained:
  ```diff
  - final_2016-11-17_kappa_test_0.70 
  ```
  and 
  ```diff
  - final_2016-11-25_kappa_test_0.752
  ```
    (the one of kappa = 0.752 was get making a fine tuning with the original data distribution at the model with kappa =0.70).
  * In modelJson are saved different networks architectures, the best was Json2016-11-17-BEST.
  * quiver-net-visualization: Contains a tool to visualize the outputs of the feature extraction layers of the CNN
  
* notebooks: All the scripts of the project
* Original-Retina-Data: Contains the a sample of the original kaggle images 
* Processed-retina-data: Numpy arrays that contains some of the processed retina images, training and test.

## TO RUN THE SCRIPTS

Run jupyter notebook from the notebooks folder


## RELEVANT SCRIPTS (notebooks folder):

>iPython notebooks

### Retina_gray_preProc: 
Performs the image pre processing to ingress the images to the network: cutting, scale Radius, subtract local mean color and mapping 50% gray, remove boundary effects, make a square and resize. It can be selected the train or the test folder of images with their corresponding labels.

### Try_preproc_gray: 
Performs the image preprocessing for a single chosen image showing all the steps.

### Retina_pre-processing: 
Was the script used to perform the initial kind of processing: cutting, make square and resize, histogram equalization, mean subtraction and normalization.  which was after replaced for the preprocessing of Retina_gray_preProc.

### Retina_grayCNN: 
Contains the network architecture and the code used to train the model with the generator. It saves the model at the beginning and the weights after each epoch, shows the graphs of accuracy, kappa and loss for training and validation data. At the end of the training calculates the kappa for a set of images and after for all the test images.

### Predict_retinopathy: 
Predicts a value of retinopathy for a set of input images using one of the two best models (final_2016-11-17_kappa_test_0.70.hdf5 or final_2016-11-25_kappa_test_0.752.hdf5). If the labels are given computes the kappa (for the hospital retina images is not possible get the kappa because there are not labels of this kind availables).

>Python notebooks

### paths_file.py: 
Contains all the relative paths to acces the data and the resources located in extern forlders. All the paths are referenced assuming the current path is the folder “notebooks”.

### generators_grayPre.py: 
Contains the generators that provides the batch of images to the network during the training, performing real-time data augmentation . The BatchGenerator method yields a specific amount of train images with an uniform classes distribution and the BatchValidationGenerator method does the same but for the validation images (a set of the test images).  The OriginalDataGenerator and the ValidationGenerator methods yields specific amount of images without modifying the classes distribution for training and validation respectively.

### retinaMethods.py: 
Contains methods that are continuously called in different classes, like sort alphabetically with the human perception of letters. Also contains some methods of the retina pre processing.

### kappa_methods.py: 
Contains the methods to calculate the Quadratic Weighted Kappa. With numpy arrays and with tensors (this used to calculate the kappa during the training and show it in the graph).

