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
  times = 10  # Number of repetitions to ensure robust timing
)

# Print the benchmarking results for the out2 training
print(benchmark_out2)



# [ToDo] Try changing the parameters above to obtain a better performance,
# this will likely take several trials
# Train the neural network with revised parameters
out6 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                lambda = 0.0004,
                rate = 0.12, 
                mbatch = 90, 
                nEpoch = 150,
                hidden_p = 175, 
                scale = 6e-3, 
                seed = 12345)

# Plot training and validation error curves
plot(1:length(out6$error), out6$error, ylim = c(0, 70), type = "o", col = "blue", xlab = "Epoch", ylab = "Error", main = "Training and Validation Errors")
lines(1:length(out6$error_val), out6$error_val, type = "o", col = "red")
legend("topright", legend = c("Training Error", "Validation Error"), col = c("blue", "red"), lty = 1)

test_error = evaluate_error(Xt, Yt, out6$params$W1, out6$params$b1, out6$params$W2, out6$params$b2)
cat("Final Training Error =", out6$error[length(out6$error)], "%\n") # better training error ~ 4.555556 %
cat("Final Validation Error =", out6$error_val[length(out6$error_val)], "%\n")
cat("Test Error =", test_error, "%\n") #higher test error ~ 15.97778 %




out7 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                lambda = 0.0005,
                rate = 0.12, 
                mbatch = 80, 
                nEpoch = 150, 
                hidden_p = 175, 
                scale = 0.006, 
                seed = 12345)

# Plot training and validation error curves
plot(1:length(out7$error), out7$error, ylim = c(0, 70), type = "o", col = "blue", xlab = "Epoch", ylab = "Error", main = "Training and Validation Errors")
lines(1:length(out7$error_val), out7$error_val, type = "o", col = "red")
legend("topright", legend = c("Training Error", "Validation Error"), col = c("blue", "red"), lty = 1)

# Evaluate error
test_error = evaluate_error(Xt, Yt, out7$params$W1, out7$params$b1, out7$params$W2, out7$params$b2)
cat("Final Training Error =", out7$error[length(out7$error)], "%\n")
cat("Final Validation Error =", out7$error_val[length(out7$error_val)], "%\n")
cat("Test Error =", test_error, "%\n")


# Train the neural network with the second-best parameters (Trial 116)
out8 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                lambda = 0.0005,
                rate = 0.12, 
                mbatch = 80, 
                nEpoch = 150, 
                hidden_p = 175, 
                scale = 0.008, 
                seed = 12345)

# Plot training and validation error curves
plot(1:length(out8$error), out8$error, ylim = c(0, 70), type = "o", col = "blue", xlab = "Epoch", ylab = "Error", main = "Training and Validation Errors")
lines(1:length(out8$error_val), out8$error_val, type = "o", col = "red")
test_error = evaluate_error(Xt, Yt, out8$params$W1, out8$params$b1, out8$params$W2, out8$params$b2)
cat("Final Training Error =", out8$error[length(out8$error)], "%\n")
cat("Final Validation Error =", out8$error_val[length(out8$error_val)], "%\n")
cat("Test Error =", test_error, "%\n")

# Train the neural network with the third-best parameters (Trial 27)
out9 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                lambda = 0.0007,
                rate = 0.15, 
                mbatch = 90, 
                nEpoch = 150, 
                hidden_p = 175, 
                scale = 0.006, 
                seed = 12345)

# Plot training and validation error curves
plot(1:length(out9$error), out9$error, ylim = c(0, 70), type = "o", col = "blue", xlab = "Epoch", ylab = "Error", main = "Training and Validation Errors")
lines(1:length(out9$error_val), out9$error_val, type = "o", col = "red")
legend("topright", legend = c("Training Error", "Validation Error"), col = c("blue", "red"), lty = 1)

# Evaluate error
test_error = evaluate_error(Xt, Yt, out9$params$W1, out9$params$b1, out9$params$W2, out9$params$b2)
cat("Final Training Error =", out9$error[length(out9$error)], "%\n")
cat("Final Validation Error =", out9$error_val[length(out9$error_val)], "%\n")
cat("Test Error =", test_error, "%\n")



# Unified benchmarking for all top 5 configurations
benchmark_results <- microbenchmark(
  out7 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                  lambda = 0.0005, rate = 0.12, mbatch = 80, 
                  nEpoch = 150, hidden_p = 175, scale = 0.006, seed = 12345),
  out8 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                  lambda = 0.0005, rate = 0.12, mbatch = 80, 
                  nEpoch = 150, hidden_p = 175, scale = 0.008, seed = 12345),
  out9 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                  lambda = 0.0007, rate = 0.15, mbatch = 90, 
                  nEpoch = 150, hidden_p = 175, scale = 0.006, seed = 12345),
  out10 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                   lambda = 0.0007, rate = 0.15, mbatch = 90, 
                   nEpoch = 150, hidden_p = 175, scale = 0.005, seed = 12345),
  out11 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                   lambda = 0.0005, rate = 0.12, mbatch = 80, 
                   nEpoch = 150, hidden_p = 150, scale = 0.006, seed = 12345),
  times = 5  # Number of repetitions for each configuration
)

# Print the benchmarking results
print(benchmark_results)
