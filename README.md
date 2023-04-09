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

### Part ii
To generate the 2D loss landscape, we can fix two of the parameters and sweep through values of the other two parameters, evaluating the least-squares error for each combination of parameters. We can then plot the resulting error values using the pcolor function from the matplotlib module to visualize the loss landscape. The algorithm can be explained as below in the example of fixing parameters C and D, while looping through A and B:

First, two arrays C_vals and D_vals are created using the np.linspace() function, which generates linearly spaced values between the given start and end points. Then, two empty 2-D arrays A_vals and B_vals are created with the shape of (50, 50), which will be used to store the error values and B parameter values respectively for each combination of C and D values.
```ruby
C_vals = np.linspace(-0.005, 0.005, 50)
D_vals = np.linspace(10, 10.01, 50)
A_vals = np.zeros((len(C_vals), len(D_vals)))
B_vals = np.zeros((len(C_vals), len(D_vals)))
```

Next, a nested loop is used to iterate through each combination of C and D values. For each combination, the model parameters params are created using the optimal values of A and B obtained from the previous optimization. The model_func() is then used to compute the error value for this combination of C and D values.The computed error value is stored in the corresponding location in A_vals, and the optimal B value is stored in the corresponding location in B_vals. 
```ruby
for i, C in enumerate(C_vals):
    for j, D in enumerate(D_vals):
        params = [result.x[0], result.x[1], C, D]
        A_vals[i, j] = model_func(params, X, Y)
        B_vals[i, j] = result.x[1]
```

Finally, A_vals is plotted as a 2-D landscape using plt.pcolor() function, where the C_vals and D_vals are used as the x and y coordinates, and the error values stored in A_vals are used as the color values. Part ii is finished by appying this algorithm to all combinations of two-variable sets.

### Part iii & iv
This question requires two sets of training data come from different part of the data set. The first group of training data is the first 20 data points, the second group of training data is the first 10 and the last 10 data points. 

1. Line Fit

The line fit to the training data using the least-squares method. It starts by creating a matrix A which stacks the features of the training data X_train as the first column, and a vector of ones as the second column. This is done to obtain the coefficients of the line equation y = mx + b, where m is the slope and b is the y-intercept.
```ruby
A = np.vstack([X_train, np.ones(len(X_train))]).T
m, b = np.linalg.lstsq(A, Y_train, rcond=None)[0]
```

The least-squares method is then used to find the values of m and b that minimize the sum of the squared errors between the predicted values of the line Y_line_train and the actual training values Y_train. Once the coefficients are obtained, the line is used to predict the values of the test data X_test, and the squared errors between these predicted values and the actual test values Y_test are calculated.
```ruby
Y_line_train = m*X_train + b
Y_line_test = m*X_test + b
```

The line_error_train variable represents the sum of the squared errors between the predicted values of the line and the actual training values, while line_error_test represents the sum of the squared errors between the predicted values of the line and the actual test values. These errors can be used to compare the performance of the line model on the training and test data, where a smaller error indicates a better fit.
```ruby
line_error_train = np.sqrt(np.sum((Y_train - Y_line_train)**2)/len(X_train))
line_error_test = np.sqrt(np.sum((Y_test - Y_line_test)**2)/len(X_test))
```

2. Parabola Fit

The parabola fit algorithm is simmilar with line fit, with replacing the linear expression to a parabola expression y = a*x^2 + b*x + c as the training function.

3. 19th Degree Polynomial Fit

To fit a 19th degree polynomial to the training data, first, a matrix A of size (n, 20) is created where n is the number of training data points. Each column of A represents a power of X_train from 0 to 19. The coefficients of the polynomial are then calculated using least squares regression by solving the equation A.T * A * coeffs = A.T * Y_train, where A.T is the transpose of A, Y_train is the vector of training data outputs, and coeffs is the vector of polynomial coefficients.
```ruby
A = np.zeros((len(X_train), 20))
for i in range(20):
    A[:, i] = X_train**i
coeffs = np.linalg.lstsq(A, Y_train, rcond=None)[0]
Y_poly_train = np.zeros(len(X_train))
Y_poly_test = np.zeros(len(X_test))
```

Next, the polynomial function is evaluated at each point in the training and test sets using the calculated coefficients. This is done by iterating over each power of X_train and X_test, multiplying it by its corresponding coefficient, and summing the products. The resulting values are stored in the Y_poly_train and Y_poly_test vectors.
```ruby
for i in range(20):
    Y_poly_train += coeffs[i]*X_train**i
    Y_poly_test += coeffs[i]*X_test**i
```

Finally, the least square errors for the polynomial fit are computed by comparing the predicted outputs to the actual outputs in both the training and test sets. The error is calculated as the sum of the squared differences between the predicted and actual outputs. The resulting values are stored in the poly_error_train and poly_error_test variables, respectively.

## Computational Results

## Summary and Conclusions

## Acknowledgement
- [ChatGPT](https://platform.openai.com/)
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [Scipy Documentation](https://docs.scipy.org/doc/scipy/)
