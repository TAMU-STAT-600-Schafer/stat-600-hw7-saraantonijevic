# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  set.seed(seed)
  
  # Initialize intercepts as zeros
  b1 <- rep(0, hidden_p)
  b2 <- rep(0, K)
  
  # Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
  W1 <- matrix(rnorm(p * hidden_p, mean = 0, sd = scale), nrow = p, ncol = hidden_p)
  W2 <- matrix(rnorm(hidden_p * K, mean = 0, sd = scale), nrow = hidden_p, ncol = K)
  
  # Return initialized parameters
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
# Function to calculate the loss, gradient, and misclassification error
# when lambda = 0 for multi-class logistic regression
loss_grad_scores <- function(y, scores, K) {
  # Number of samples
  n <- length(y)
  
  # Apply softmax to scores to get probabilities
  exp_scores <- exp(scores)
  probs <- exp_scores / rowSums(exp_scores)
  
  # Create one-hot encoded matrix using matrix indexing
  y_one_hot <- matrix(0, n, K)
  y_one_hot[cbind(1:n, y + 1)] <- 1
  
  # Calculate loss
  loss <- -sum(y_one_hot * log(probs)) / n
  
  # Predict class labels from scores
  predicted_labels <- max.col(probs) - 1  # max.col returns 1-indexed, subtract 1 for 0-indexed
  
  # Calculate misclassification error rate
  error <- mean(predicted_labels != y) * 100  # in percentage
  
  # Calculate gradient of loss with respect to scores
  grad <- (probs - y_one_hot) / n
  
  # Return loss, gradient, and misclassification error
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
# One pass function with regularization term added to the loss
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda){
  n <- nrow(X)
  
  # Forward pass
  hidden_input <- X %*% W1 + matrix(rep(b1, each = n), nrow = n, byrow = TRUE)
  hidden_output <- matrix(pmax(0, hidden_input), nrow = n, ncol = ncol(hidden_input))
  
  scores <- hidden_output %*% W2 + matrix(rep(b2, each = n), nrow = n, byrow = TRUE)
  
  # Calculate probabilities and loss using the function previously created
  loss_grad_result <- loss_grad_scores(y, scores, K)
  loss <- loss_grad_result$loss
  
  # Add regularization term to the loss
  reg_term <- (lambda / 2) * (sum(W1^2) + sum(W2^2))
  loss <- loss + reg_term
  
  # Correct the access to the gradient
  grad_scores <- loss_grad_result$grad
  
  # Ensure grad_scores is a numeric matrix
  if (!is.matrix(grad_scores) && !is.null(grad_scores)) {
    grad_scores <- as.matrix(grad_scores)
  } else if (is.null(grad_scores)) {
    stop("Gradient returned by loss_grad_scores is NULL, cannot proceed.")
  }
  
  # Step 1: Gradient with respect to W2 and b2
  grad_W2 <- t(hidden_output) %*% grad_scores + lambda * W2
  grad_b2 <- colSums(grad_scores)
  
  # Step 2: Backpropagate gradient to hidden layer
  grad_hidden <- grad_scores %*% t(W2)
  grad_hidden[hidden_input <= 0] <- 0  # Apply ReLU derivative
  
  # Step 3: Gradient with respect to W1 and b1
  grad_W1 <- t(X) %*% grad_hidden + lambda * W1
  grad_b1 <- colSums(grad_hidden)
  
  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(loss = loss, error = loss_grad_result$error, grads = list(dW1 = grad_W1, db1 = grad_b1, dW2 = grad_W2, db2 = grad_b2)))
}


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


# Full training function for the neural network
NN_train <- function(X, y, Xval, yval, lambda = 0.01,
                     rate = 0.01, mbatch = 20, nEpoch = 100,
                     hidden_p = 20, scale = 1e-3, seed = 12345) {
  # Get sample size and total number of batches
  n <- length(y)
  nBatch <- floor(n / mbatch)
  
  # Initialize weights and biases using the provided initialize_bw function
  init_params <- initialize_bw(p = ncol(X), hidden_p = hidden_p, K = length(unique(y)), scale = scale, seed = seed)
  W1 <- init_params$W1
  b1 <- init_params$b1
  W2 <- init_params$W2
  b2 <- init_params$b2
  
  # Initialize storage for training and validation errors to monitor convergence
  error <- rep(NA, nEpoch)
  error_val <- rep(NA, nEpoch)
  
  # Set seed for reproducibility in shuffling
  set.seed(seed)
  
  # Start training iterations
  for (i in 1:nEpoch) {
    # Shuffle and create batch assignments
    batch_ids <- sample(rep(1:nBatch, length.out = n), size = n)
    
    # Variable to store total batch error for averaging
    total_batch_error <- 0
    
    # Loop over each batch
    for (batch in 1:nBatch) {
      # Extract the indices for the current batch
      batch_indices <- which(batch_ids == batch)
      X_batch <- X[batch_indices, , drop = FALSE]
      y_batch <- y[batch_indices]
      
      # Perform one pass to get current error and gradients
      results <- one_pass(X_batch, y_batch, K = length(unique(y)), W1, b1, W2, b2, lambda)
      
      # Update weights and biases using SGD with the learning rate
      W1 <- W1 - rate * results$grads$dW1
      b1 <- b1 - rate * results$grads$db1
      W2 <- W2 - rate * results$grads$dW2
      b2 <- b2 - rate * results$grads$db2
      
      # Accumulate batch error for averaging
      total_batch_error <- total_batch_error + results$error
    }
    
    # Average training error for the epoch
    error[i] <- total_batch_error / nBatch
    
    # Evaluate validation error at the end of each epoch
    error_val[i] <- evaluate_error(Xval, yval, W1, b1, W2, b2)
    
    
  }
  
  # Return the final parameters and recorded errors
  return(list(error = error, error_val = error_val, params = list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)))
}



