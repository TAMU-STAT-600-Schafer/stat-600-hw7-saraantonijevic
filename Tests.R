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



#Does the objective value improve across iterations?



# How does the classification error change across iterations?