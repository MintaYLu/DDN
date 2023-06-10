#' bcd
#'
#' solve the Lasso with bcd - Block coordinate descent
#'
#' @param y the expression values of dependent variable
#' @param X the expression values of input variables
#' @param lambda1 parameter used to assure a sparse common network structure
#' @param lambda2 parameter used to assure a sparse differential network rewiring
#' @param beta network coefficient parameter
#'
#' @return beta (network coefficient parameter)
#'
#' @examples
#' # Give an example here that can run independently
#'
#' @export
bcd <- function(y, X, lambda1, lambda2, beta=rep(0, ncol(X))) {

  p <- ncol(X) / 2
  n <- nrow(X)
  isStop <- FALSE
  r <- 0

  if (p == 1) {
    beta <- solve2d(y, X[, 1], X[, 2], lambda1, lambda2)
  }else {

    while (!isStop) {
      beta.old <- beta
      for (i in 1:p) {
        r <- r + 1
        k <- r %% p
        if (k==0) {k <- p}
        x1 <- X[, k]
        x2 <- X[, p+k]

        y.residual <- y - X[, c(-k, -p-k)] %*% matrix(beta[c(-k, -p-k)], nrow=2*p-2, ncol=1)
        beta2d <- solve2d(y.residual, x1, x2, lambda1, lambda2)
        beta[k] <- beta2d[1]
        beta[p+k] <- beta2d[2]
      }

      if (sum(abs(beta - beta.old)) < 0.001*p*2) {
        isStop <- TRUE;
      }
    }

  } 

  beta
  
}
