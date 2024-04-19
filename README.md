# MACHINE LEARNING AND PATTERN RECOGNITION
#### Project task
## Lab 2 - Features loading and visualization
The project task consists of a binary classification problem. The goal is to perform fingerprint spoofing detection, i.e. to identify genuine vs counterfeit fingerprint images. The dataset consists of labelled samples corresponding to the genuine (True, label 1) class and the fake (False, label 0) class. The samples are computed by a feature extractor that summarizes high-level characteristics of a fingerprint image. The data is 6-dimensional.
The training files for the project are stored in file Project/trainData.txt. The format of the file is the same as for the Iris dataset, i.e. a csv file where each row represents a sample. The first 6 values of each row are the features, whereas the last value of each row represents the class (1 or 0). The samples are not ordered. 
Load the dataset and plot the histogram and pair-wise scatter plots of the different features. Analyze the plots:
1.	Analyze the first two features. What do you observe? Do the classes overlap? If so, where? Do the classes show similar mean for the first two features? Are the variances similar for the two classes? How many modes are evident from the histograms (i.e., how many “peaks” can be observed)?
2.	Analyze the third and fourth features. What do you observe? Do the classes overlap? If so, where? Do the classes show similar mean for these two features? Are the variances similar for the two classes? How many modes are evident from the histograms?
3.	Analyze the last two features. What do you observe? Do the classes overlap? If so, where? How many modes are evident from the histograms? How many clusters can you notice from the scatter plots for each class?
## Lab 3 - Dimensionality reduction
Apply PCA and LDA to the project data.
1. Start analyzing the effects of PCA on the features. Plot
the histogram of the projected features for the 6 PCA directions, starting from the principal (largest
variance). What do you observe? What are the effects on the class distributions? Can you spot the
different clusters inside each class?

2. Apply LDA (1 dimensional, since we have just two classes), and compute the histogram of the projected
LDA samples. What do you observe? Do the classes overlap? Compared to the histogram of the 6
features you computed in Laboratory 2, is LDA finding a good direction with little class overlap?

3. Try applying LDA as classifier. Divide the dataset in model training and validation sets (you can reuse
the previous function to split the dataset). Apply LDA, and select the threshold as in the previous
sections. Compute the predictions, and the error rate.

4. Now try changing the value of the threshold. What do you observe? Can you find values that improve
the classification accuracy?

5. Finally, try pre-processing the features with PCA. Apply PCA (estimated on the model training data
only), and then classify the validation data with LDA. Analyze the performance as a function of the
number of PCA dimensions m . What do you observe? Can you find values of m that improve the
accuracy on the validation set? Is PCA beneficial for the task when combined with the LDA classifier?
## Lab 4 - Gaussian density estimation
1. Try fitting uni-variate Gaussian models to the different features of the project dataset. For each component
of the feature vectors, compute the ML estimate for the parameters of a 1D Gaussian distribution.
2. Plot the distribution density (remember that you have to exponentiate the log-density) on top of the
normalized histogram (set density=True when creating the histogram, see Laboratory 2). What do
you observe? Are there features for which the Gaussian densities provide a good fit? Are there features
for which the Gaussian model seems significantly less accurate?  
*Note*: for this part of the project, since we are still performing some preliminary, qualitative analysis,
you can compute the ML estimates and the plots either on the whole training set. In the following labs
we will employ the densities for classification, and we will need to perform model selection, therefore we
will re-compute ML estimates on the model training portion of the dataset only (see Laboratory 3).