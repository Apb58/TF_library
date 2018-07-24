# TensorFlow Model and ML Project Library

A collection of ML models built with the TensorFlow api package for python3 (https://www.tensorflow.org) and other modeling tools such as `glmnet` package in R.

Models are in the main directory, while training and testing data is in the `/data` repo.

### Model Collection:

  1. **Diabetes Patient Classification**: application of the softmax regression model to group patients not diagnosed with diabetes, from those with either Chemical or Overt diabetes, based on several clinical measures. Training and testing obtained from the R dataset 'Diabetes' from the 'heplots' library (Original data from Reaven and Miller (1979)). 

  2. **LIHC Tumor Stage Classification using Viral Load**: application of multinomial regression in `glmnet` to create a model predicting patient tumor stage classification based on HBV and total viral load, as reported in by Cao et. Al (Cao, S et. al. 'Divergent viral presentation among human tumors and adjacent normal tissues'. *Sci. Rep.*, **6**,28294 (2016)). Tumor stage and clinical information retrieved from TCGA biobank (All LIHC patients).
     
