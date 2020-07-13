library(evclass)
library(FNN)

#la fonction
EkNNval_uncertain_label <- function(xtrain, ytrain, y_uncertainty, xtst, K, ytst = NULL, param = NULL){
  
  xtst   <- as.matrix(xtst)
  xtrain <- as.matrix(xtrain)
  ytrain <- y <- as.integer(as.factor(ytrain))
  if(!is.null(ytst)) ytst <- y <- as.integer(as.factor(ytst))
  
  if(is.null(param)) param <- EkNNinit(xtrain, ytrain)
  
  Napp <- nrow(xtrain)
  M    <- max(ytrain)
  N    <- nrow(xtst)
  
  knn         <- get.knnx(xtrain, xtst, k=K)
  knn$nn.dist <- knn$nn.dist^2
  is          <- t(knn$nn.index)
  ds          <- t(knn$nn.dist)
  uncertainty <- t(knn$nn.dist)
  for (i in 1 : nrow(uncertainty)) {
    for (j in 1 : ncol(uncertainty)) {
      uncertainty[i,j] <- y_uncertainty[is[i,j]]
    }
  }
  
  m = rbind(matrix(0, M, N), rep(1, N))
  
  for(i in 1 : N){
    for(j in 1 : K){
      m1                   <- rep(0, M + 1)
      m1[ytrain[is[j, i]]] <- param$alpha * exp( - param$gamma[ytrain[is[j, i]]]^2 * ds[j, i]) * (1 - uncertainty[j, i])
      m1[M + 1]            <- 1 - m1[ytrain[is[j, i]]]
      m[1 : M, i]          <- m1[1 : M] * m[1 : M, i] + m1[1 : M] * m[M + 1, i] + m[1 : M, i] * m1[M + 1]
      m[M + 1, i]          <- m1[M + 1] * m[M + 1, i]
      m                    <- m/matrix(colSums(m), M + 1, N, byrow = TRUE)
    }
  }
  m     <- t(m)
  ypred <- max.col(m[, 1 : M])
  if(!is.null(ytst)) err <- length(which(ypred != ytst))/N else err <- NULL
  
  return(list(m = m, ypred = ypred, err = err))
}

# Synthetic example:
df <- data.frame(x = c(0, 0, 2, 2, 2, 1), 
                 y = c(0, 2, 0, 1, 2, 1), 
                 label = c("A","A","B","B","B", "?"), 
                 uncertainty_level = c(0.1, 0.1, 0.9, 0.9, 0.9, NA))
plot(df[, -c(3, 4)], type ='p', pch = 16, cex = 0)
text(df$x, df$y, labels=df$label, cex= 2)

# 2 neighbours labeled "A" with high confidence level (uncertainty = 0.1) and 
# 3 neighbours labeled "B" with low confidence level (uncertainty = 0.9) 

EkNNval_uncertain_label(df[1:5, 1:2], df[-6, 'label'], df[-6, 'uncertainty_level'], df[6,1:2], K = 5)
# -> predictive mass function= m(A)=0.51, m(B)=0,04, m(A, B)=0.45
