# Supervised Learning
- Ontlines:
    - Regression
        - regression.ipynb
        - regression_with_scikit-learn.ipynb
    - Classification
        - logistic_regression.ipynb
## Problem Description of regression.ipynb
Please load "data.mat" into your Python code, where you will find 𝑥, 𝑦 ∈ $𝑅^{1001}.$
And do the following procedures.
1. Compute the least square line $y=\theta_0+x\theta_1$ using the given data and overlay the line over the given data.
2. Using the same data from Question 1, compute the least square parabola (i.e. 
second order polynomial $y=\theta_0+x\theta_1+x^2\theta_2$) to fit the data.
3. Using the same data from Question 2, now we use the loss function (L1 Norm) 
below instead of least square based methods.
![lossfunc](./img/lossfunc.png)

## Problem Description of regression_with_scikit-learn.ipynb
Please analysis the relation between the house's attribute and the house price. And Report what you found.

## Problem Description of logistic_regression.ipynb
- In ‘train.mat,’ you can find 2-D points X=[x1, x2] and their corresponding labels Y=y. 
- Please use logistic regression ℎ(𝜽) = 1/ 1+𝑒−𝜽𝑇𝑥 to find the decision boundary (optimal 𝜽∗) based on ‘train.mat.” 
- Please use a gradient descent method to solve it and report the test error on the test dataset ‘test.mat.’ (percentage of misclassified test samples)

## Requirements:
- scikit-learn 
- seaborn (for data visulization)











