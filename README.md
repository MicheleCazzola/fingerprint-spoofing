# MACHINE LEARNING AND PATTERN RECOGNITION
#### Project task
## Lab 2
The project task consists of a binary classification problem. The goal is to perform fingerprint spoofing detection, i.e. to identify genuine vs counterfeit fingerprint images. The dataset consists of labelled samples corresponding to the genuine (True, label 1) class and the fake (False, label 0) class. The samples are computed by a feature extractor that summarizes high-level characteristics of a fingerprint image. The data is 6-dimensional.
The training files for the project are stored in file Project/trainData.txt. The format of the file is the same as for the Iris dataset, i.e. a csv file where each row represents a sample. The first 6 values of each row are the features, whereas the last value of each row represents the class (1 or 0). The samples are not ordered. 
Load the dataset and plot the histogram and pair-wise scatter plots of the different features. Analyze the plots:
1.	Analyze the first two features. What do you observe? Do the classes overlap? If so, where? Do the classes show similar mean for the first two features? Are the variances similar for the two classes? How many modes are evident from the histograms (i.e., how many “peaks” can be observed)?
2.	Analyze the third and fourth features. What do you observe? Do the classes overlap? If so, where? Do the classes show similar mean for these two features? Are the variances similar for the two classes? How many modes are evident from the histograms?
3.	Analyze the last two features. What do you observe? Do the classes overlap? If so, where? How many modes are evident from the histograms? How many clusters can you notice from the scatter plots for each class?
#### Answers
1. For each feature, both classes have a single modal distribution with normal shape and they overlap in the central part of their domain:
   - mean values are the same (about 0) for both features and their classes;
   - variances are different:  
     - about 0.6 for the *fake* class of feature 1 and the *genuine* class of feature 2;
     - about 1.4 for the *genuine* class of feature 1 and the *fake* class of feature 2.
   
   We can observe that, for each feature, the class with the higher variance has the highest modal frequency (peek value).


2. For each feature, both classes have a single modal distribution with normal shape and they overlap on the side of the normal distribution; for each pair of classes inside each feature:
   - mean values are opposite but almost equal in module (between 0.6 and 0.7);
   - variances are almost equal (between 0.5 and 0.6).
   
   We can observe that, for each feature, the classes have about the same modal frequency.


3. For each feature, the *fake* class has a single modal distribution and the *genuine* class has a double modal distribution:
   - for the *fake* class, modal values are opposites;
   - for the *genuine* class, modal value is about 0.

   We can observe that they overlap around the modal values of the *fake* class distribution, while in the central part of the domain the overlapping is almost null.  
   Moreover, the scatter plots highlight the presence of 4 clusters for each class.