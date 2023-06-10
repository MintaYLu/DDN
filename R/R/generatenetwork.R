
rm(list = ls())


source("bcd.R")

source("ddn.R")

source("solve2d.R")

load("GRNsimData.RData")

data1 <- t(data1)
data2 <- t(data2)
lambda1=0.26
lambda2=0.12285

p <- dim(data1)[2]
n <- dim(data1)[1]


node.name <- gene.name

beta.matrix <- ddn(data1, data2, lambda1, lambda2)  

edge.list1 <- matrix(character(), ncol=3, nrow=0)
edge.list2 <- matrix(character(), ncol=3, nrow=0)

for (node in 1:ncol(beta.matrix)) {
  beta <- as.vector(beta.matrix[, node])
  beta1 <- beta[1:p]
  beta2 <- beta[(p+1):(p*2)]
  neighor <- node.name[which(beta1 != 0)]
  neighor <- node.name[which(beta2 != 0)]
  diff.edge.set1 <- node.name[setdiff(which(beta1 != 0), which(beta2 != 0))]
  diff.edge.set2 <- node.name[setdiff(which(beta2 != 0), which(beta1 != 0))]
  
  if (length(diff.edge.set1) >= 1) {
    for (i in 1:length(diff.edge.set1)) {
      print(c(node.name[node], "1", diff.edge.set1[i]))
      if (node.name[node] < diff.edge.set1[i]) {
        edge.list1 <- rbind(edge.list1, c(node.name[node], "1", diff.edge.set1[i]))
      } else {
        edge.list1 <- rbind(edge.list1, c(diff.edge.set1[i], "1", node.name[node]))
      }
      
      
    }
  }
  
  if (length(diff.edge.set2) >= 1) {
    for (i in 1:length(diff.edge.set2)) {
      print(c(node.name[node], "2", diff.edge.set2[i]))
      if (node.name[node] < diff.edge.set2[i]) {
        edge.list2 <- rbind(edge.list2, c(node.name[node], "2", diff.edge.set2[i]))
      } else {
        edge.list2 <- rbind(edge.list2, c(diff.edge.set2[i], "2", node.name[node]))
      }
      
    }
  }
  
}

#library(igraph)
#library(influential)
data.to.write_initial <- rbind(edge.list1, edge.list2)
dup_idx <- duplicated(data.to.write_initial)
data.to.write <- data.to.write_initial[!dup_idx, ]
write(t(data.to.write), file="SimGRNNetwork.sif", sep="\t", ncol=3)

g1 <- graph_from_data_frame(as.data.frame(data.to.write[, c(1, 3, 2)]), directed = FALSE)
plot(g1, layout = layout.circle(g1), edge.color=c("blue", "blue", "blue", "blue", "red", "red", "red"))
