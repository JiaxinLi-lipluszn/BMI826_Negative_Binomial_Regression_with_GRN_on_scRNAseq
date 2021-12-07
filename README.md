# BMI826 Negative Binomial Regression with GRN on scRNA-seq 
## TASK 1
Train a negative binomial regression model for each gene on the expression level on training set (80% of the cells) with both the regulators of the gene and the sequencing depth of the cell. Then use the trained model to predict the expression level in validation set. This is a validation task to verify if incorporating GRN into negative binomial regression can benefit the model in term of caperturing the biological variance in single cell data.

## TASK 2
Incorporate NB regression into the gene regulatory network inference algorithms like MERLIN that assums a conditional Gaussian distribution on gene expression level in the dataset. 

## Implementations
* [X] Negative Binomial distribution with gene-wise alpha (NegativeBinomialRegression.py)
* [X] Parallel computing of Negative Binomial distribution (NegativeBinomialRegression_wrapper.py)
* [ ] Parameter sharing between genes, genes with similar expression profiles will share the parameters in negative binomial regression 
* [ ] Find the classes and functions that need to be modified in MERLIN.
* [ ] Incorporate Negative Binomial Regression into MERLIN
  - [ ]  double getPseudoLikelihood
  - [ ]  double getGaussianLikelihood

## Applications
* [ ] A2S; regress on only sequencing depth; 
  - [X] Predictions
  - [X] Validations
* [ ] A2S; regress on both sequencing depth and regulators;
  - [X] Predictions
  - [X] Validations
* [ ] pbmc; regress on only sequencing depth; 
  - [X] Predictions
  - [ ] Validations
* [ ] pbmc; regress on both sequencing depth and regulators;
  - [X] Predictions
  - [ ] Validations
