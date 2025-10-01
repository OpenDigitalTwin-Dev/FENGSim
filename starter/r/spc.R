
tfile <- "output.svg";svg(width=6,height=6,pointsize=10,filename=tfile); library(qcc,lib.loc='/home/jiping/FENGSim/toolkit/PS/install/r_install/lib/R/library');mu = 100;sigma_W = 10;epsilon = rnorm(500);x = matrix(mu + sigma_W*epsilon, ncol=10, byrow=TRUE);q = qcc(x, type="xbar");dev.off()
