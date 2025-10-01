tfile <- "output.svg"
png("output.png",width=800,height=800,res=150,pointsize=12)
library(qcc)
mu = 100
sigma_W = 10
epsilon = rnorm(500)
x = matrix(mu + sigma_W*epsilon, ncol=10, byrow=TRUE)
q = qcc(x, type="xbar")
dev.off()
