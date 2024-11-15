# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  set.seed(seed)
  
  # [ToDo] Initialize intercepts as zeros
  b1 <- rep(0, hidden_p)
  b2 <- rep(0, K)
  
  # [ToDo] Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
  
  W1 <- matrix(rnorm(p * hidden_p, mean = 0, sd = scale), nrow = p, ncol = hidden_p)
  W2 <- matrix(rnorm(hidden_p * K, mean = 0, sd = scale), nrow = hidden_p, ncol = K)
  
  # Return
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K){

  # Number of samples
  n <- length(y)
  
  # Apply softmax to scores to get probabilities
 
  exp_scores <- exp(scores)
  probs <- exp_scores / rowSums(exp_scores)
  
  
  # Convert y to a one-hot encoded matrix
  y_one_hot <- matrix(0, n, K)
  for (i in 1:n) {
    y_one_hot[i, y[i] + 1] <- 1
  }
  
  
  # [ToDo] Calculate loss when lambda = 0
  # loss = ...
  loss <- -sum(y_one_hot * log(probs)) / n
  
  # Predict class labels from scores
  predicted_labels <- max.col(probs) - 1  # max.col returns 1-indexed, subtract 1 for 0-indexed
  
  
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  # error = ...
  # Calculate misclassification error rate
  error <- mean(predicted_labels != y) * 100  # in percentage
  
  # Calculate gradient of loss with respect to scores (when lambda = 0)
  grad <- (probs - y_one_hot) / n
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  # when lambda = 0
  # grad = ...

  
  # Return loss, gradient and misclassification error on training (in %)
  return(list(loss = loss, grad = grad, error = error, probs = probs))
  
  }

# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations


#hidden_output <- matrix(pmax(0, hidden_input), nrow = n, ncol = ncol(hidden_input))

one_pass <- function(X, y, K, W1, b1, W2, b2, lambda){
  n <- nrow(X)
  
  # Forward pass
  hidden_input <- X %*% W1 + matrix(rep(b1, each = n), nrow = n, byrow = TRUE)
  hidden_output <- matrix(pmax(0, hidden_input), nrow = n, ncol = ncol(hidden_input))
  
  scores <- hidden_output %*% W2 + matrix(rep(b2, each = n), nrow = n, byrow = TRUE)
  
  # Debug: Print intermediate values
  print("Hidden input:")
  print(hidden_input)
  print("Hidden output:")
  print(hidden_output)
  print("Scores:")
  print(scores)
  
  # Calculate probabilities and loss using the function previously created
  loss_grad_result <- loss_grad_scores(y, scores, K)
  
  # Debug: Check what loss_grad_result returns
  print("Loss and Gradient Result:")
  print(loss_grad_result)
  
  # Correct the access to the gradient
  grad_scores <- loss_grad_result$grad
  
  # Ensure grad_scores is a numeric matrix
  if (!is.matrix(grad_scores) && !is.null(grad_scores)) {
    grad_scores <- as.matrix(grad_scores)
  } else if (is.null(grad_scores)) {
    stop("Gradient returned by loss_grad_scores is NULL, cannot proceed.")
  }
  
  # Debug: Print grad_scores to check its structure
  print("Grad Scores:")
  print(grad_scores)
  
  loss <- loss_grad_result$loss
  
  # Step 1: Gradient with respect to W2 and b2
  grad_W2 <- t(hidden_output) %*% grad_scores + lambda * W2
  grad_b2 <- colSums(grad_scores)
  
  # Debug: Print gradients for W2 and b2
  print("Gradient W2:")
  print(grad_W2)
  print("Gradient b2:")
  print(grad_b2)
  
  # Step 2: Backpropagate gradient to hidden layer
  grad_hidden <- grad_scores %*% t(W2)
  grad_hidden[hidden_input <= 0] <- 0  # Apply ReLU derivative
  
  # Debug: Print gradient at hidden layer
  print("Gradient Hidden Layer:")
  print(grad_hidden)
  
  # Step 3: Gradient with respect to W1 and b1
  grad_W1 <- t(X) %*% grad_hidden + lambda * W1
  grad_b1 <- colSums(grad_hidden)
  
  # Debug: Print gradients for W1 and b1
  print("Gradient W1:")
  print(grad_W1)
  print("Gradient b1:")
  print(grad_b1)
  
  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(loss = loss_grad_result$loss, error = loss_grad_result$error, grads = list(dW1 = grad_W1, db1 = grad_b1, dW2 = grad_W2, db2 = grad_b2)))
}

# Set seed for reproducibility
set.seed(42)

# Generate a small synthetic dataset
X <- matrix(rnorm(20), nrow = 5, ncol = 4)  # 5 samples, 4 features
y <- sample(0:2, 5, replace = TRUE)  # Class labels for 3 classes (0, 1, 2)

# Initialize weights and biases
p <- ncol(X)  # Number of input features
h <- 3        # Number of hidden units
K <- 3        # Number of classes

W1 <- matrix(rnorm(p * h), nrow = p, ncol = h)
b1 <- rnorm(h)
W2 <- matrix(rnorm(h * K), nrow = h, ncol = K)
b2 <- rnorm(K)

# Set a small lambda for ridge regularization
lambda <- 0.01

# Run the one_pass function
result <- one_pass(X, y, K, W1, b1, W2, b2, lambda)

# Print results
print(paste("Loss:", result$loss))
print("Gradient W1:")
print(result$grads$dW1)
print("Gradient b1:")
print(result$grads$db1)
print("Gradient W2:")
print(result$grads$dW2)
print("Gradient b2:")
print(result$grads$db2)



# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
evaluate_error <- function(Xval, yval, W1, b1, W2, b2){
  nval <- nrow(Xval)
  
  # [ToDo] Forward pass to get scores on validation data
  hidden_input <- Xval %*% W1 + matrix(rep(b1, each = nval), nrow = nval)
  hidden_output <- matrix(pmax(0, hidden_input), nrow = nval, ncol = ncol(hidden_input))
  scores <- hidden_output %*% W2 + matrix(rep(b2, each = nval), nrow = nval)
  
  
  # [ToDo] Evaluate error rate (in %) when 
  # comparing scores-based predictions with true yval
  predicted_classes <- max.col(scores) - 1 # Subtract 1 for 0-based indexing
  error <- mean(predicted_classes != yval) * 100
  
  
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
NN_train <- function(X, y, Xval, yval, lambda = 0.01,
                     rate = 0.01, mbatch = 20, nEpoch = 100,
                     hidden_p = 20, scale = 1e-3, seed = 12345){
  # Get sample size and total number of batches
  n = length(y)
  nBatch = floor(n/mbatch)
  
  # [ToDo] Initialize b1, b2, W1, W2 using initialize_bw with seed as seed,
  # and determine any necessary inputs from supplied ones
  
  p <- ncol(X)
  K <- length(unique(y))
  init_params <- initialize_bw(p, hidden_p, K, scale, seed)
  W1 <- init_params$W1
  b1 <- init_params$b1
  W2 <- init_params$W2
  b2 <- init_params$b2
  
  # Initialize storage for error to monitor convergence
  error = rep(NA, nEpoch)
  error_val = rep(NA, nEpoch)
  
  # Set seed for reproducibility
  set.seed(seed)
  # Start iterations
  for (i in 1:nEpoch){
    # Allocate bathes
    batchids = sample(rep(1:nBatch, length.out = n), size = n)
    # [ToDo] For each batch
    #  - do one_pass to determine current error and gradients
    #  - perform SGD step to update the weights and intercepts
    
    # Track training error for this epoch
    epoch_loss <- 0
    for (batch in 1:nBatch) {
      # Select batch indices
      batch_indices <- which(batchids == batch)
      X_batch <- X[batch_indices, , drop = FALSE]
      y_batch <- y[batch_indices]
      
      # Run one pass to get current loss, error, and gradients
      pass_result <- one_pass(X_batch, y_batch, K, W1, b1, W2, b2, lambda)
      
      # Accumulate the loss for averaging later
      epoch_loss <- epoch_loss + pass_result$loss
      
      # Extract gradients
      dW1 <- pass_result$grads$dW1
      db1 <- pass_result$grads$db1
      dW2 <- pass_result$grads$dW2
      db2 <- pass_result$grads$db2
      
      # SGD update step
      W1 <- W1 - rate * dW1
      b1 <- b1 - rate * db1
      W2 <- W2 - rate * dW2
      b2 <- b2 - rate * db2
    }
    
    # [ToDo] In the end of epoch, evaluate
    # - average training error across batches
    # - validation error using evaluate_error function
    
    
    # Calculate average training error across batches
    error[i] <- epoch_loss / nBatch
    
    # Evaluate validation error at the end of the epoch
    error_val[i] <- evaluate_error(Xval, yval, W1, b1, W2, b2)
    
  }
  # Return end result
  return(list(error = error, error_val = error_val, params =  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)))
}