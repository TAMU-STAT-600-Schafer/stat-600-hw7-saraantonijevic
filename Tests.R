source("FunctionsNN.R")

#Do I calculate the objective function correctly?

# Simple dataset
X <- matrix(c(1, 0, 0, 1), nrow = 2)  # Two samples, two features
y <- c(0, 1)                          # Two classes (0-based indexing)

# Initialize parameters
params <- initialize_bw(p = 2, hidden_p = 2, K = 2)

# Perform one forward pass
results <- one_pass(X, y, K = 2, W1 = params$W1, b1 = params$b1, W2 = params$W2, b2 = params$b2, lambda = 0)

# Manually compute the expected loss and compare
scores <- results$hidden_output %*% params$W2 + matrix(params$b2, 2, 2, byrow = TRUE)
exp_scores <- exp(scores)
probs <- exp_scores / rowSums(exp_scores)
true_labels <- matrix(0, 2, 2)
true_labels[cbind(1:2, y + 1)] <- 1
expected_loss <- -sum(true_labels * log(probs)) / 2

# Output results
cat("Calculated Loss:", results$loss, "\nExpected Loss:", expected_loss, "\n")


#What happens if I use two normal populations?

# Generate data from two normal populations
set.seed(42)
X <- rbind(matrix(rnorm(50, mean = 2), nrow = 25),
           matrix(rnorm(50, mean = -2), nrow = 25))
y <- c(rep(0, 25), rep(1, 25))

#X <- rbind(matrix(rnorm(50, mean = 0, sd = 2), nrow = 25),
 #          matrix(rnorm(50, mean = 0, sd = 2), nrow = 25))
#y <- c(rep(0, 25), rep(1, 25))

# Validation set
Xval <- rbind(matrix(rnorm(20, mean = 2), nrow = 10),
              matrix(rnorm(20, mean = -2), nrow = 10))
yval <- c(rep(0, 10), rep(1, 10))

# Train the neural network
results <- NN_train(X, y, Xval, yval, lambda = 0.01, rate = 0.1, mbatch = 10, nEpoch = 50, hidden_p = 5)

# Output training and validation errors
cat("Final Training Error:", results$error[50], "\nFinal Validation Error:", results$error_val[50], "\n")


#Does the objective value improve across iterations?

# Generate simple linearly separable data
set.seed(42)
X <- rbind(matrix(rnorm(50, mean = 2), nrow = 25),
           matrix(rnorm(50, mean = -2), nrow = 25))
y <- c(rep(0, 25), rep(1, 25))

# Validation set
Xval <- rbind(matrix(rnorm(20, mean = 2), nrow = 10),
              matrix(rnorm(20, mean = -2), nrow = 10))
yval <- c(rep(0, 10), rep(1, 10))

# Train the neural network with loss tracking
results <- NN_train(X, y, Xval, yval, lambda = 0.01, rate = 0.1, mbatch = 10, nEpoch = 50, hidden_p = 5)

# Plot the objective value (loss) across epochs
plot(1:length(results$error), results$error, type = "l", col = "blue",
     xlab = "Epoch", ylab = "Training Error",
     main = "Training Error Across Iterations")


# How does the classification error change across iterations?

