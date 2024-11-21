# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345) {
  # Set the seed for reproducibility
  set.seed(seed)
  
  # Initialize intercepts as zeros
  b1 <- rep(0, hidden_p)  # Intercept for the hidden layer
  b2 <- rep(0, K)         # Intercept for the output layer
  
  # Initialize weights by drawing them iid from Normal with mean zero and scale as sd
  W1 <- matrix(rnorm(p * hidden_p, mean = 0, sd = scale), nrow = p, ncol = hidden_p)
  W2 <- matrix(rnorm(hidden_p * K, mean = 0, sd = scale), nrow = hidden_p, ncol = K)
  
  # Return
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}


# Function to calculate loss, error, and gradient strictly based on scores with lambda = 0
#############################################################
# loss_grad_scores <- function(y, scores, K) {
#   # Number of samples (n)
#   n <- nrow(scores)
#   
#   # Apply softmax to scores to get probabilities
#   exp_scores <- exp(scores)
#   probs <- exp_scores / rowSums(exp_scores)  # Normalize to get probabilities
#   
#   # Create one-hot encoding for true labels
#   true_labels <- matrix(0, n, K)
#   true_labels[cbind(1:n, y + 1)] <- 1  # Adding 1 for R's 1-based indexing
#   
#   # Calculate cross-entropy loss when lambda = 0
#   # f(B) = -1/n * sum( 1(y_i = k) * log(p_k(x_i)) )
#   loss <- -sum(true_labels * log(probs)) / n
#   
#   # Calculate misclassification error rate (%)
#   # Predict the class by taking the argmax of probabilities
#   predicted_labels <- apply(probs, 1, which.max) - 1  # Convert back to 0-based labels
#   error <- mean(predicted_labels != y) * 100
#   
#   # Calculate gradient of loss with respect to scores
#   # grad = (probs - true_labels) / n
#   grad <- (probs - true_labels) / n
#   
#   # Return loss, gradient, and misclassification error on training (in %)
#   return(list(loss = loss, grad = grad, error = error, probs = probs))
# }



# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer) # y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K) {
  # Number of samples (n)
  n <- nrow(scores)
  
  # Apply softmax to scores to get probabilities
  exp_scores <- exp(scores)
  probs <- exp_scores / rowSums(exp_scores)  # Normalize to get probabilities
  
  # Create one-hot encoding for true labels
  true_labels <- matrix(0, n, K)
  true_labels[cbind(1:n, y + 1)] <- 1  # Adding 1 for R's 1-based indexing
  
  # Calculate cross-entropy loss when lambda = 0
  # f(β) = -1/n * sum( 1(y_i = k) * log(p_k(x_i)) )
  loss <- -sum(true_labels * log(probs)) / n
  
  # Calculate misclassification error rate (%)
  # Predict the class by taking the argmax of probabilities
  predicted_labels <- max.col(probs, ties.method = "first") - 1  # Convert back to 0-based labels
  error <- mean(predicted_labels != y) * 100
  
  # Calculate gradient of loss with respect to scores
  # grad = (probs - true_labels) / n
  grad <- (probs - true_labels) / n
  
  # Return loss, gradient, and misclassification error on training (in %)
  return(list(loss = loss, grad = grad, error = error, probs = probs))
}






# One pass function for forward and backward propagation
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda) {
  # Number of samples (n)
  n <- nrow(X)
  
  # [Forward pass]
  # From input to hidden layer
  hidden_input <- X %*% W1 + matrix(b1, n, length(b1), byrow = TRUE)
  
  # Apply ReLU activation
  hidden_output <- pmax(hidden_input, 0)
  
  # From hidden to output scores
  scores <- hidden_output %*% W2 + matrix(b2, n, length(b2), byrow = TRUE)
  
  # Get loss, error, gradient at current scores using loss_grad_scores function
  out <- loss_grad_scores(y, scores, K)
  
  # [Backward pass]
  # Gradient for W2 and b2 (output layer)
  dW2 <- t(hidden_output) %*% out$grad + lambda * W2
  db2 <- colSums(out$grad)
  
  # Gradient propagation to hidden layer
  hidden_grad <- out$grad %*% t(W2)
  hidden_grad[hidden_input <= 0] <- 0  # Backprop through ReLU (gradient is zero where input <= 0)
  
  # Gradients for W1 and b1 (input to hidden layer)
  dW1 <- t(X) %*% hidden_grad + lambda * W1
  db1 <- colSums(hidden_grad)
  
  # Return output (loss, error from forward pass, hidden output, gradients)
  return(list(
    loss = out$loss,
    error = out$error,
    hidden_output = hidden_output,
    grads = list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)
  ))
}

# Function to evaluate validation set error
####################################################
evaluate_error <- function(Xval, yval, W1, b1, W2, b2) {
  # Number of validation samples (nval)
  nval <- nrow(Xval)
  
  # [Forward pass]
  # From input to hidden layer
  hidden_input_val <- Xval %*% W1 + matrix(b1, nval, length(b1), byrow = TRUE)
  
  # Apply ReLU activation
  hidden_output_val <- pmax(hidden_input_val, 0)
  
  # From hidden to output scores
  scores_val <- hidden_output_val %*% W2 + matrix(b2, nval, length(b2), byrow = TRUE)
  
  # Apply softmax to scores to get probabilities
  exp_scores_val <- exp(scores_val)
  probs_val <- exp_scores_val / rowSums(exp_scores_val)
  
  # [Evaluate error rate]
  # Predict the class by taking the argmax of probabilities
  predicted_labels_val <- max.col(probs_val, ties.method = "first") - 1  # Adjusting to 0-based index
  
  # Calculate error rate (%)
  error <- mean(predicted_labels_val != yval) * 100
  
  return(error)
}


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



