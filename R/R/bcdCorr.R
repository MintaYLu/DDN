bcdCorr <- function(n1,n2, CurrIdx,CorrMtx1, CorrMtx2, lambda1, lambda2,beta=rep(0, ncol(CorrMtx1)*2)) {
  p <- ncol(CorrMtx1)
  beta1 <- rep(0, p)
  beta2 <- rep(0, p)
  # beta1 <- beta[1:p]
  # beta2 <- beta[(p+1):p*2]
  
  solve2dRho <- function(rho1, rho2, lambda1, lambda2) {
    beta1 <- 0
    beta2 <- 0
    if (rho2 <= (rho1 + 2*lambda2) && rho2 >= (rho2 - 2*lambda2)
        && rho2 >= (2*lambda1 - rho1)) {
      beta1 <- (rho1 + rho2)/2 - lambda1
      beta2 <- (rho1 + rho2)/2 - lambda1
    }
    else if (rho2 > (rho1 + 2*lambda2) && rho1 >= (lambda1 - lambda2)) {
      beta1 <- rho1 - lambda1 + lambda2
      beta2 <- rho2 - lambda1 - lambda2
    }
    
    else if (rho1 < (lambda1 - lambda2) && rho1 >= -(lambda1 + lambda2)
             && rho2 >= (lambda1 + lambda2)) {
      beta1 <- 0
      beta2 <- rho2 - lambda1 - lambda2
    }
    else if (rho1 < -(lambda1 + lambda2) && rho2 >= (lambda1 + lambda2)) {
      beta1 <- rho1 + lambda1 + lambda2
      beta2 <- rho2 - lambda1 - lambda2
    }
    else if (rho1 < -(lambda1 + lambda2) && rho2 < (lambda1 + lambda2)
             && rho2 >= -(lambda1 - lambda2)) {
      beta1 <- rho1 + lambda1 + lambda2
      beta2 <- 0
    }
    else if (rho2 < -(lambda1 - lambda2) && rho2 >= (rho1 + 2*lambda2)) {
      area.index <- 6
      beta1 <- rho1 + lambda1 + lambda2
      beta2 <- rho2 + lambda1 - lambda2
    }
    else if (rho2 >= (rho1 - 2*lambda2) && rho2 < (rho1 + 2*lambda2)
             && rho2 <= (-2*lambda1 - rho1)) {
      beta1 <- (rho1 + rho2)/2 + lambda1
      beta2 <- (rho1 + rho2)/2 + lambda1
    }
    else if (rho2 < (rho1 - 2*lambda2) && rho1 <= -(lambda1 - lambda2)) {
      beta1 <- rho1 + lambda1 - lambda2
      beta2 <- rho2 + lambda1 + lambda2
    }
    else if (rho1 <= (lambda1 + lambda2) && rho1 >= -(lambda1 - lambda2)
             && rho2 <= -(lambda1 + lambda2)) {
      beta1 <- 0
      beta2 <- rho2 + lambda1 + lambda2
    }
    else if (rho1 > (lambda1 + lambda2) && rho2 <= -(lambda1 + lambda2)) {
      beta1 <- rho1 - lambda1 - lambda2
      beta2 <- rho2 + lambda1 + lambda2
    }
    else if (rho2 > -(lambda1 + lambda2) && rho2 <= (lambda1 - lambda2)
             && rho1 >= (lambda1 + lambda2)) {
      beta1 <- rho1 - lambda1 - lambda2
      beta2 <- 0
    }
    else if (rho2 > (lambda1 - lambda2) && rho2 < (rho1 - 2*lambda2)) {
      beta1 <- rho1 - lambda1 - lambda2
      beta2 <- rho2 - lambda1 + lambda2
    }
    beta <- c(beta1, beta2)
    beta
  }
  
  isStop <- FALSE
  r <- 0
  while (!isStop) {
    beta1.old <- beta1
    beta2.old <- beta2
    
    for (i in 1:p) {
      if (i==CurrIdx) {
        next
      }
        
      r <- r + 1
      k <- i
      
      betaBar1 <- -beta1
      betaBar2 <- -beta2
      betaBar1[k] <- 0
      betaBar2[k] <- 0
      betaBar1[CurrIdx] <- 1
      betaBar2[CurrIdx] <- 1
      
      rho1 <- sum(betaBar1 %*% (CorrMtx1[,k]))
      rho2 <- sum(betaBar2 %*% (CorrMtx2[,k]))
      
      beta2d <- solve2dRho(rho1, rho2, lambda1, lambda2)
      beta1[k] <- beta2d[1]
      beta2[k] <- beta2d[2]
    }
    betaerr <- sum(abs( c(beta1 - beta1.old, beta2 - beta2.old) ))
    if ( (betaerr< 0.001*p*2) || r>10000) {
      isStop <- TRUE
    }
  }

  beta <- c(beta1,beta2)
  beta
}
