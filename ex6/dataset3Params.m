function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
lowest_error_rate = 10000000;
best_C = 0;
best_sigma = 0;
%hyper parameters to choose from
hyper_param_choices = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
for i = 1:length(hyper_param_choices)
  c_choice = hyper_param_choices(i);
  for j = 1:length(hyper_param_choices)
    sigma_choice = hyper_param_choices(j);
    model = svmTrain(X,y, c_choice, @(x1, x2) gaussianKernel(x1, x2, sigma_choice));
    predictions = svmPredict(model, Xval);
    error_rate = double(mean(predictions~=yval));
    if error_rate <lowest_error_rate
      lowest_error_rate = error_rate;
      best_C = c_choice;
      best_sigma = sigma_choice;
    endif
  endfor;
endfor;

lowest_error_rate
C = best_C;
sigma = best_sigma;

% =========================================================================

end
