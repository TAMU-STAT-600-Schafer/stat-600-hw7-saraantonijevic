# Load the data
install.packages("microbenchmark")
library(microbenchmark)
# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Update training to set last part as validation
id_val = 1801:2000
Yval = Y[id_val]
Xval = X[id_val, ]
Ytrain = Y[-id_val]
Xtrain = X[-id_val, ]

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

# Source the NN function
source("FunctionsNN.R")
# [ToDo] Source the functions from HW3 (replace FunctionsLR.R with your working code)
source("FunctionsLR.R")


# Recall the results of linear classifier from HW3
# Add intercept column
Xinter <- cbind(rep(1, nrow(Xtrain)), Xtrain)
Xtinter <- cbind(rep(1, nrow(Xt)), Xt)

#  Apply LR (note that here lambda is not on the same scale as in NN due to scaling by training size)
out <- LRMultiClass(Xinter, Ytrain, Xtinter, Yt, lambda = 1, numIter = 150, eta = 0.1)
plot(out$objective, type = 'o')
plot(out$error_train, type = 'o') # around 19.5 if keep training
plot(out$error_test, type = 'o') # around 25 if keep training


# Apply neural network training with default given parameters
out2 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 100, scale = 1e-3, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70))
lines(1:length(out2$error_val), out2$error_val, col = "red")


# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error # 16.1

# Print the training error
cat("Final Training Error =", out2$error[length(out2$error)], "%\n")
cat("Final Validation Error =", out2$error_val[length(out2$error_val)], "%\n")
# Load the microbenchmark package
library(microbenchmark)

# Microbenchmark the NN_train function call that produces `out2`
benchmark_out2 <- microbenchmark(
  out2 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                  rate = 0.1, mbatch = 50, nEpoch = 150,
                  hidden_p = 100, scale = 1e-3, seed = 12345),
  times = 5  # Number of repetitions to ensure robust timing
)

# Print the benchmarking results for the out2 training
print(benchmark_out2)



# [ToDo] Try changing the parameters above to obtain a better performance,
# this will likely take several trials

# Train the neural network with improved parameters
out4 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                lambda = 0.001,
                rate = 0.15, 
                mbatch = 75, 
                nEpoch = 150,
                hidden_p = 125, 
                scale = 5e-3, 
                seed = 12345)

# Plot training and validation error curves
plot(1:length(out4$error), out4$error, ylim = c(0, 70), type = "o", col = "blue", xlab = "Epoch", ylab = "Error", main = "Training and Validation Errors")
lines(1:length(out4$error_val), out4$error_val, type = "o", col = "red")
legend("topright", legend = c("Training Error", "Validation Error"), col = c("blue", "red"), lty = 1)

# Evaluate test error
test_error = evaluate_error(Xt, Yt, out4$params$W1, out4$params$b1, out4$params$W2, out4$params$b2)

# Print results
cat("Final Training Error =", out4$error[length(out4$error)], "%\n")
cat("Final Validation Error =", out4$error_val[length(out4$error_val)], "%\n")
cat("Test Error =", test_error, "%\n")



# Train the neural network with further fine-tuned parameters
out5 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                lambda = 0.0008,
                rate = 0.18, 
                mbatch = 80, 
                nEpoch = 150,
                hidden_p = 150, 
                scale = 8e-3, 
                seed = 12345)

# Plot training and validation error curves
plot(1:length(out5$error), out5$error, ylim = c(0, 70), type = "o", col = "blue", xlab = "Epoch", ylab = "Error", main = "Training and Validation Errors")
lines(1:length(out5$error_val), out5$error_val, type = "o", col = "red")
legend("topright", legend = c("Training Error", "Validation Error"), col = c("blue", "red"), lty = 1)

# Evaluate test error
test_error = evaluate_error(Xt, Yt, out5$params$W1, out5$params$b1, out5$params$W2, out5$params$b2)

# Print results
cat("Final Training Error =", out5$error[length(out5$error)], "%\n")
cat("Final Validation Error =", out5$error_val[length(out5$error_val)], "%\n")
cat("Test Error =", test_error, "%\n")
