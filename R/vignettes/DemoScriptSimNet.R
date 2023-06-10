rm(list = ls())

library(mvtnorm)

library(MASS)


n <- 200
p <- 6

sigma1.inv <- matrix(c(1, 0.3, -0.3, 0.3, 0, 0,
                       0.3, 1,  0,  0, 0.3, 0,
                       -0.3, 0, 1, 0.3, 0, 0.3,
                       0.3, 0, 0.3, 1, 0, 0, 
                       0, 0.3, 0, 0, 1, 0,
                       0, 0, 0.3, 0, 0, 1), nrow = 6, ncol = 6, byrow = TRUE)

sigma2.inv <- matrix(c(1, 0.3,-0.3, 0, 0, 0,
                       0.3, 1,  0,  0, 0.3, 0,
                       -0.3, 0, 1, 0, 0, 0.3,
                       0, 0, 0, 1, -0.5, 0.3,
                       0, 0.3, 0, -0.5, 1, 0,
                       0, 0, 0.3, 0.3, 0, 1), nrow = 6, ncol = 6, byrow = TRUE)


sigma1 <- ginv(sigma1.inv)
sigma2 <- ginv(sigma2.inv)

data1 <- rmvnorm(n, sigma=sigma1)
data2 <- rmvnorm(n, sigma=sigma2)

scaling.factor1 <- numeric()
scaling.factor2 <- numeric()
for (i in 1:nrow(sigma1)) {
  s <- sqrt(sum(data1[,i] ^ 2))
  scaling.factor1 <- c(scaling.factor1, s)
  data1[,i] <- data1[,i] / s
  s <- sqrt(sum(data2[,i] ^ 2))
  scaling.factor2 <- c(scaling.factor2, s)
  data2[,i] <- data2[,i] / s
}

print(scaling.factor1)
print(scaling.factor2)

data <- cbind(rbind(data1, matrix(0, nrow = nrow(data2), ncol = ncol(data1)) ),
              rbind(matrix(0, nrow=nrow(data1), ncol=ncol(data2)), data2))

lambda1 <- 0.1
lambda2 <- 0.05

obj.func <- function(beta, y, X, lambda1, lambda2) {
  p <- ncol(X) / 2
  f <- 0.5 * sum((y - X %*% beta) ^ 2) + lambda1 * sum(abs(beta)) + lambda2 * sum(abs(beta[1:p] - beta[(p+1):(2*p)]))
  f
}

node.name <- c("A", "B", "C", "D", "E", "F")

for (node in 1:nrow(sigma1)) {

  y <- c(as.vector(data1[,node]), as.vector(data2[,node]))

  X <- data[,c(-node, -(node+nrow(sigma1)))]

  cat("\n-----------------------------\n")
  cat(" Node: ", node.name[node])
  
  beta <- bcd(y, X, lambda1, lambda2)
 
  cat("\nSolution by bcd: \n")
  print(beta)
  cat(" Optimal value = ", obj.func(beta, y, X, lambda1, lambda2), "\n")

  if (node == 1) {
    beta1 <- c(0, beta[1:(p-1)])
    beta2 <- c(0, beta[p:(p*2-2)])
  }
  if (node == p) {
    beta1 <- c(beta[1:(p-1)], 0)
    beta2 <- c(beta[p:(p*2-2)], 0)
  }
  if (node > 1 && node < p) {
    beta1 <- c(beta[1:(node-1)], 0, beta[node:(p-1)])
    beta2 <- c(beta[p:(p+node-2)], 0, beta[(p+node-1):(p*2-2)])
  }


  cat(" Neighbors of Node ", node.name[node], "\n")
  neighor <- node.name[which(beta1 != 0)]
  cat("   Condition 1 ", neighor, "\n")
  neighor <- node.name[which(beta2 != 0)]
  cat("   Condition 2 ", neighor, "\n")
  
  

  ## beta0 <- rep(0, times=ncol(X))
  ## result <- optim(beta0, obj.func, y=y, X=X, lambda1=lambda1, lambda2=lambda2, method="BFGS")
  ## beta <- result$par

  ## cat("\nSolution by optim: \n")
  ## print(beta)
  ## cat(" Optimal value = ", result$val, "\n")


}



