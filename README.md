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

## Lab 5 - Generative models I: Gaussian models
1. Apply the MVG model to the project data. Split the dataset in model training and validation subsets
(**important**: use the same splits for all models, including those presented in other laboratories), train the
model parameters on the model training portion of the dataset and compute LLRs: (with class True, label 1 on top of the
ratio) for the validation subset. Obtain predictions from LLRs assuming uniform class priors P(C = 1) = P(C = 0) = 0.5.
Compute the corresponding error rate (_suggestion_: in the next laboratories we will modify the way we compute
predictions from LLRs, we therefore recommend that you keep separated the functions that compute LLRs, those that
compute predictions from LLRs and those that compute error rate from predictions).

2. Apply now the tied Gaussian model, and compare the results with MVG and LDA. Which model seems
to perform better?

3. Finally, test the Naive Bayes Gaussian model. How does it compare with the previous two?

4. Let’s now analyze the results in light of the characteristics of the features that we observed in previous
laboratories. Start by printing the covariance matrix of each class (you can extract this from the MVG
model parameters). The covariance matrices contain, on the diagonal, the variances for the different
features, whereas the elements outside the diagonal are the feature co-variances. For each class,
compare the covariance of different feature pairs with the respective variances. What do you observe?
Are co-variance values large or small compared to variances? To better visualize the strength of covariances
with respect to variances we can compute, for a pair of features i, j, the Pearson correlation
coefficient, or, directly the covariance matrix. The correlation matrix has diagonal elements equal to 1,
whereas out-diagonal elements correspond to the correlation coefficients for all feature pairs; when Corr(i; j) = 0 the 
features (i, j) are uncorrelated, whereas values close to +-1 denote strong correlation.
Compute the correlation matrices for the two classes. What can you conclude on the features? Are the
features strongly or weakly correlated? How is this related to the Naive Bayes results?

5. The Gaussian model assumes that features can be jointly modeled by Gaussian distributions. The goodness
of the model is therefore strongly affected by the accuracy of this assumption. Although visualizing
6-dimensional distributions is unfeasible, we can analyze how well the assumption holds for single (or
pairs) of features. In Laboratory 4 we separately fitted a Gaussian density over each feature for each
class. This corresponds to the Naive Bayes model. What can you conclude on the goodness of the
Gaussian assumption? Is it accurate for all the 6 features? Are there features for which the assumptions
do not look good?
6. To analyze if indeed the last set of features negatively affects our classifier because of poor modeling
assumptions, we can try repeating the classification using only feature 1 to 4 (i.e., discarding the last 2
features). Repeat the analysis for the three models. What do you obtain? What can we conclude on
discarding the last two features? Despite the inaccuracy of the assumption for these two features, are
the Gaussian models still able to extract some useful information to improve classification accuracy?
7. In Laboratory 2 and 4 we analyzed the distribution of features 1-2 and of features 3-4, finding that for
features 1 and 2 means are similar but variances are not, whereas for features 3 and 4 the two classes
mainly differ for the feature mean, but show similar variance. Furthermore, the features also show limited
correlation for both classes. We can analyze how these characteristics of the features distribution affect
the performance of the different approaches. Repeat the classification using only features 1-2 (jointly),
and then do the same using only features 3-4 (jointly), and compare the results of the MVG and tied
MVG models. In the first case, which model is better? And in the second case? How is this related
to the characteristics of the two classifiers? Is the tied model effective at all for the first two features?
Why? And the MVG? And for the second pair of features?
8. Finally, we can analyze the effects of PCA as pre-processing. Use PCA to reduce the dimensionality of
the feature space, and apply the three classification approaches. What do you observe? Is PCA effective
for this dataset with the Gaussian models? Overall, what is the model that provided the best accuracy
on the validation set?