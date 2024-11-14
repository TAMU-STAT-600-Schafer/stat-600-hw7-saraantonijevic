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







#Test initialize_bw and loss_grad_scores with unifrom scores:
# Create uniform scores
scores_uniform <- matrix(1, nrow = 4, ncol = 3)
y_uniform <- c(0, 1, 2, 1)

# Run function


results_uniform <- loss_grad_scores(y_uniform, scores_uniform, K = 3)
print(results_uniform$probs)
stopifnot(all(abs(results_uniform$probs - expected_prob) < 1e-3))
