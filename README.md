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
1. For each feature, both classes exhibit a unimodal distribution with Normal shape, and they overlap in the central part of their domain:
   - the mean values are the same (approximately 0) for both features and their respective classes;
   - the variances are different:  
     - approximately 0.6 for the *fake* class of feature 1 and the *genuine* class of feature 2;
     - approximately 1.4 for the *genuine* class of feature 1 and the *fake* class of feature 2.
   
   We observe that, for each feature, the class with the higher variance exhibits the highest modal frequency (peak value).

2. For each feature, both classes demonstrate a unimodal distribution with Normal shape, and they overlap on their respective sides; for each pair of classes within each feature:
   - the mean values are opposite but nearly equal in magnitude (between 0.6 and 0.7);
   - the variances are nearly equal (between 0.5 and 0.6).
   
   We observe that, for each feature, the classes display the similar modal frequencies.

3. For each feature, the *fake* class displays a unimodal distribution, while the *genuine* class exhibits a bimodal distribution:
   - for the *fake* class, the modal values are opposite;
   - for the *genuine* class, the modal value is approximately 0.

   We observe that they overlap around the modal values of the *fake* class distribution, while the overlapping is minimal in the central part of the domain.
   Furthermore, the scatter plots highlight the presence of four clusters for each class.