ddn <- function(data1, data2, lambda1, lambda2) {

  p <- dim(data1)[2]
  n <- dim(data1)[1]

  data1 <- data1
  data2 <- data2


  ## Normalization
  for (i in 1:p) {
    data1[,i] <- data1[,i] - mean(data1[,i])
    data2[,i] <- data2[,i] - mean(data2[,i])
  }
  
  
  scaling.factor1 <- numeric()
  scaling.factor2 <- numeric()
  for (i in 1:p) {
    s <- sqrt(sum(data1[,i] ^ 2))
    scaling.factor1 <- c(scaling.factor1, s)
    data1[,i] <- data1[,i] / s
    s <- sqrt(sum(data2[,i] ^ 2))
    scaling.factor2 <- c(scaling.factor2, s)
    data2[,i] <- data2[,i] / s
  }


  
  data <- cbind(rbind(data1, matrix(0, nrow = nrow(data2), ncol = ncol(data1)) ),
                rbind(matrix(0, nrow=nrow(data1), ncol=ncol(data2)), data2))

  obj.func <- function(beta, y, X, lambda1, lambda2) {
    p <- ncol(X) / 2
    f <- 0.5 * sum((y - X %*% beta) ^ 2) + lambda1 * sum(abs(beta)) + lambda2 * sum(abs(beta[1:p] - beta[(p+1):(2*p)]))
    f
  }


  beta.matrix <- matrix(0, ncol=0, nrow=p*2)
  
  for (node in 1:p) {

    y <- c(as.vector(data1[,node]), as.vector(data2[,node]))

    X <- data[,c(-node, -(node+p))]

    beta <- bcd(y, X, lambda1, lambda2)
    
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

    beta <- c(beta1, beta2)
    beta.matrix <- cbind(beta.matrix, beta)
  }


  beta.matrix

}


