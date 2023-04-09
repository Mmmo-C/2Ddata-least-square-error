# 2Ddata-least-square-error
UW course work - EE399 - HW1
</p>
Xinqi Chen @05/04/2023 

## Table of Content
- [2Ddata-loss-landscape](#2ddata-loss-landscape)
  - [Abstract](#abstract)
  - [Overview](#overview)
  - [Theoretical Background](#theoretical-background)
  - [Algorithm Implementation and Development](#algorithm-implementation-and-development)
  - [Computational Results](#computational-results)
  - [Summary and Conclusions](#summary-and-conclusions)
  - [Acknowledgement](#acknowledgement)

## Abstract
This is an intro-level basic machine learning algorithm practice with a small sample space. The goal of the project is to explore simple concepts such as model fitting, loss function, error landscape, and model training. The project is done through Python packages including matplotlib, Sklearn, and Scipy.optimize. Based on the given dataset (size of 31), every aspect is handled in a 2-D plane, and Numpy arrays are used to hold and manipulate data.  

## Overview
The project concerns fitting a model to data with the least-squares error function. Specifically, a set of data points is given, represented by the arrays X and Y, and we want to find the parameters A, B, C, and D that minimize the error of a model of the form:
$$f (x) = A cos(Bx) + Cx + D$$

Once the optimal parameters is found, then the behavior of the model is investigated by fixing two of the parameters and sweeping through values of the other two parameters to generate a 2D loss (error) landscape. By doing this, we can visualize the regions of parameter space where the model performs well and identify any local minima or other features that may be of interest.

Finally, the performance of the model is evaluated on new data by dividing the original data into a training set and a test set. The model is fitted to the training set and is computed the least-squares error for each of three different models: a linear model, a quadratic model, and a 19th-degree polynomial model. After that, the errors are compared on the test set for each of these models to determine which one performs best on unseen data. 

## Theoretical Background
The problem involves fitting a function to a given set of data points, specifically a least-squares fit of the model function $$f(x) = A cos(Bx) + Cx + D$$ to the given data X and Y.

The least-squares fitting involves minimizing the sum of squared errors between the predicted values of the model and the actual values of the data. In this case, the error is defined as the difference between the predicted value of the model and the actual data point value for each x.
$$f(x) = \sqrt{(1/n)* \sum_{j=1}^{n} (f(x_j)-y_j)^2}$$

To solve for the optimal parameters A, B, C, and D that minimize the error, the optimization techniques is used such as Scipy.optimize.minimize. It iteratively adjust the values of the parameters until the error is minimized.

In part (ii) of the problem, the loss landscape represents the variation of the loss (or error) with respect to the parameters, and it allows us to visualize the structure of the optimization problem. By examining the landscape, we can identify the presence of multiple local minima, which may affect the convergence of optimization algorithms.

In parts (iii) and (iv), by comparing the errors of these models, we can evaluate their performance and choose the best one for the given data. This involves fitting the models to the training data and evaluating their performance on the test data, which are a set of data points that were not used during the training process. This helps to prevent overfitting of the model to the training data and provides an estimate of the model's generalization performance.

## Algorithm Implementation and Development 
### Part i
The minimize function from the scipy.optimize module with Nelder-Mead method is used to minimize the least-squares error between the model and the data.
```ruby
result = minimize(model_func, x0=[3, 1*np.pi/4, 2/3, 32], args=(X, Y), method='Nelder-Mead')
y_fit = result.x[0]*np.cos(result.x[1]*X) + result.x[2]*X + result.x[3]
```

## Computational Results

## Summary and Conclusions

## Acknowledgement
- [ChatGPT](https://platform.openai.com/)
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [Scipy Documentation](https://docs.scipy.org/doc/scipy/)
