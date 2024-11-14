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


#Test 2: edge case with zero weights and biases
# Zero-initialized weights and biases
W1_zero <- matrix(0, nrow = 3, ncol = 5)
b1_zero <- rep(0, 5)
W2_zero <- matrix(0, nrow = 5, ncol = 3)
b2_zero <- rep(0, 3)

# Run the function
results_zero <- one_pass(X_test, y_test, K = 3, W1_zero, b1_zero, W2_zero, b2_zero, lambda_test)
print(results_zero)

# Ensure loss is calculated and gradients have expected structure
stopifnot(results_zero$loss > 0)
stopifnot(is.matrix(results_zero$grads$dW1) && all(dim(results_zero$grads$dW1) == dim(W1_zero)))
stopifnot(is.matrix(results_zero$grads$dW2) && all(dim(results_zero$grads$dW2) == dim(W2_zero)))


#Test 3: Gradient check with small input
# Function for numerical gradient checking
numerical_gradient_check <- function(X, y, K, W1, b1, W2, b2, lambda, epsilon = 1e-5) {
  # Analytical gradient computation
  res <- one_pass(X, y, K, W1, b1, W2, b2, lambda)
  grad_analytical <- res$grads
  
  # Check W1 gradients
  num_grad_W1 <- matrix(0, nrow = nrow(W1), ncol = ncol(W1))
  for (i in 1:nrow(W1)) {
    for (j in 1:ncol(W1)) {
      W1_pos <- W1; W1_pos[i, j] <- W1_pos[i, j] + epsilon
      W1_neg <- W1; W1_neg[i, j] <- W1_neg[i, j] - epsilon
      
      loss_pos <- one_pass(X, y, K, W1_pos, b1, W2, b2, lambda)$loss
      loss_neg <- one_pass(X, y, K, W1_neg, b1, W2, b2, lambda)$loss
      
      num_grad_W1[i, j] <- (loss_pos - loss_neg) / (2 * epsilon)
    }
  }
  
  # Print numerical and analytical gradients for comparison
  print("Numerical and analytical gradients comparison for W1:")
  print(cbind(num_grad_W1, grad_analytical$dW1))
  
  # Repeat for other gradients (W2, b1, b2) similarly if needed
}

# Run numerical gradient check
numerical_gradient_check(X_test, y_test, K = 3, W1_test, b1_test, W2_test, b2_test, lambda_test)
