source("FunctionsNN.R")


#Test initialize_bw and loss_grad_scores with synthetic data:

# Test initialization function
params <- initialize_bw(p = 3, hidden_p = 2, K = 3, scale = 0.01, seed = 1)
print(params)

# Check structure and dimensions
stopifnot(length(params$b1) == 2)
stopifnot(length(params$b2) == 3)
stopifnot(dim(params$W1)[1] == 3 && dim(params$W1)[2] == 2)
stopifnot(dim(params$W2)[1] == 2 && dim(params$W2)[2] == 3)

# Test scores, true labels, and loss_grad_scores function
scores <- matrix(c(1.5, 0.5, 1.0, 
                   0.7, 1.2, 0.8, 
                   0.3, 0.4, 1.5), nrow = 3, byrow = TRUE)
y <- c(0, 1, 2)  # True labels

results <- loss_grad_scores(y, scores, K = 3)
print(results)

# Check if loss, gradient, and error make sense
stopifnot(is.numeric(results$loss))
stopifnot(is.matrix(results$grad))
stopifnot(length(results$error) == 1)



#Test 2: initialize_bw and loss_grad_scores with unifrom scores:
# Create uniform scores
scores_uniform <- matrix(1, nrow = 4, ncol = 3)
y_uniform <- c(0, 1, 2, 1)

results_uniform <- loss_grad_scores(y_uniform, scores_uniform, K = 3)
print(results_uniform$probs)
stopifnot(all(abs(results_uniform$probs - expected_prob) < 1e-3))



#Test 3: One-Hot Encoding Check
# One class is predicted with very high certainty
scores_confident <- matrix(c(10, -1, -2, 
                             -2, 10, -3, 
                             -4, -5, 15), nrow = 3, byrow = TRUE)
y_confident <- c(0, 1, 2)

# Run function
results_confident <- loss_grad_scores(y_confident, scores_confident, K = 3)
print(results_confident)

# Check if gradient and loss are consistent with highly confident predictions
stopifnot(results_confident$loss > 0)


#Test 4: large input test
# Generate larger input data
set.seed(42)
scores_large <- matrix(rnorm(1000 * 5, mean = 0, sd = 1), nrow = 1000, ncol = 5)
y_large <- sample(0:4, 1000, replace = TRUE)

# Run function
results_large <- loss_grad_scores(y_large, scores_large, K = 5)

# Check for errors in computation and reasonable output
stopifnot(!is.na(results_large$loss))
stopifnot(all(results_large$error >= 0 & results_large$error <= 100))






#Test 1: basic functionality check
# Simple synthetic test data
set.seed(42)
X_test <- matrix(rnorm(10 * 3), nrow = 10, ncol = 3)  # 10 samples, 3 features
y_test <- sample(0:2, 10, replace = TRUE)  # 3 classes (0 to 2)
W1_test <- matrix(rnorm(3 * 5, mean = 0, sd = 0.1), nrow = 3, ncol = 5)  # 3 input features, 5 hidden units
b1_test <- rep(0, 5)
W2_test <- matrix(rnorm(5 * 3, mean = 0, sd = 0.1), nrow = 5, ncol = 3)  # 5 hidden units, 3 output classes
b2_test <- rep(0, 3)
lambda_test <- 0.1

# Run the function
results_test <- one_pass(X_test, y_test, K = 3, W1_test, b1_test, W2_test, b2_test, lambda_test)
print(results_test)

# Checks for expected outputs
stopifnot(is.numeric(results_test$loss))
stopifnot(length(results_test$error) == 1)
stopifnot(is.matrix(results_test$grads$dW1) && all(dim(results_test$grads$dW1) == dim(W1_test)))
stopifnot(is.vector(results_test$grads$db1) && length(results_test$grads$db1) == length(b1_test))
stopifnot(is.matrix(results_test$grads$dW2) && all(dim(results_test$grads$dW2) == dim(W2_test)))
stopifnot(is.vector(results_test$grads$db2) && length(results_test$grads$db2) == length(b2_test))


#Test 2:
