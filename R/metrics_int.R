####################################################################
# Internal validation                                              
# 
# Please cite:
# J.Bridge et al. Development and Validation of a Mxed-Effects 
# Deep Learning Model to Diagnose COVID-19 from
# CT Imaging. 2021
#
# For the desicion curve code, visit:
# www.decisioncurveanalysis.org
# 
# joshua.bridge@liverpool.ac.uk
# github.com/JTBridge/ME-COVID19
# Apache License 2.0
#
# Packages may be subject to their own licences.
###################################################################

install.packages("remotes")
remotes::install_github("BavoDC/CalibrationCurves")
library("CalibrationCurves")
library(pROC)
library(reportROC)
library(rms)
library(caret)
library(epiR)


# Load predictions
TRUE_int = read.csv('results/internal/true.csv', header = FALSE)

ME_int = read.csv('results/internal/ME.csv', header = FALSE)
ME_int = cbind(TRUE_int, ME_int)
colnames(ME_int)[1] <- "true"
colnames(ME_int)[2] <- "pred"

bai_int = read.csv('results/internal/Bai.csv', header = FALSE)
bai_int = cbind(TRUE_int, bai_int)
colnames(bai_int)[1] <- "true"
colnames(bai_int)[2] <- "pred"

covinet_int = read.csv('results/internal/covinet_int.csv', header = FALSE)
covinet_int = cbind(TRUE_int, covinet_int)
colnames(covinet_int)[1] <- "true"
colnames(covinet_int)[2] <- "pred"

covnet_int = read.csv('results/internal/covnet.csv', header = FALSE)
covnet_int = cbind(TRUE_int, covnet_int)
colnames(covnet_int)[1] <- "true"
colnames(covnet_int)[2] <- "pred"

true_factor = ifelse(ME_int$true>0.5, 1, 0)
true_factor = factor(true_factor, levels=c(0,1))

# Mixed-Effects
ME_roc = roc(ME_int, true, pred)
auc(ME_roc)
ci.auc(ME_roc)
ME_30 = factor(ifelse(ME_int$pred>0.30,1,0), levels=c(0,1))
test30 = epi.tests(confusionMatrix(ME_30, true_factor)$table)
ME_40 = factor(ifelse(ME_int$pred>0.40,1,0), levels=c(0,1))
test40 = epi.tests(confusionMatrix(ME_40, true_factor)$table)
ME_50 = factor(ifelse(ME_int$pred>0.50,1,0), levels=c(0,1))
test50 = epi.tests(confusionMatrix(ME_50, true_factor)$table)
ME_60 = factor(ifelse(ME_int$pred>0.60,1,0), levels=c(0,1))
test60 = epi.tests(confusionMatrix(ME_60, true_factor)$table)
ME_70 = factor(ifelse(ME_int$pred>0.70,1,0), levels=c(0,1))
test70 = epi.tests(confusionMatrix(ME_70, true_factor)$table)

round(test70$detail$se, 3)
round(test70$detail$sp, 3)
round(test70$detail$pv.pos, 3)
round(test70$detail$pv.neg, 3)



# Bai
bai_roc = roc(bai_int, true, pred)
auc(bai_roc)
ci.auc(bai_roc)
roc.test(ME_roc, bai_roc)
bai_30 = factor(ifelse(bai_int$pred>0.30,1,0), levels=c(0,1))
test30 = epi.tests(confusionMatrix(bai_30, true_factor)$table)
bai_40 = factor(ifelse(bai_int$pred>0.40,1,0), levels=c(0,1))
test40 = epi.tests(confusionMatrix(bai_40, true_factor)$table)
bai_50 = factor(ifelse(bai_int$pred>0.50,1,0), levels=c(0,1))
test50 = epi.tests(confusionMatrix(bai_50, true_factor)$table)
bai_60 = factor(ifelse(bai_int$pred>0.60,1,0), levels=c(0,1))
test60 = epi.tests(confusionMatrix(bai_60, true_factor)$table)
bai_70 = factor(ifelse(bai_int$pred>0.70,1,0), levels=c(0,1))
test70 = epi.tests(confusionMatrix(bai_70, true_factor)$table)

round(test70$detail$se, 3)
round(test70$detail$sp, 3)
round(test70$detail$pv.pos, 3)
round(test70$detail$pv.neg, 3)

# Covinet
covinet_roc = roc(covinet_int, true, pred)
auc(covinet_roc)
ci.auc(covinet_roc)
covinet_30 = factor(ifelse(covinet_int$pred>0.30,1,0), levels=c(0,1))
test30 = epi.tests(confusionMatrix(covinet_30, true_factor)$table)
covinet_40 = factor(ifelse(covinet_int$pred>0.40,1,0), levels=c(0,1))
test40 = epi.tests(confusionMatrix(covinet_40, true_factor)$table)
covinet_50 = factor(ifelse(covinet_int$pred>0.50,1,0), levels=c(0,1))
test50 = epi.tests(confusionMatrix(covinet_50, true_factor)$table)
covinet_60 = factor(ifelse(covinet_int$pred>0.60,1,0), levels=c(0,1))
test60 = epi.tests(confusionMatrix(covinet_60, true_factor)$table)
covinet_70 = factor(ifelse(covinet_int$pred>0.70,1,0), levels=c(0,1))
test70 = epi.tests(confusionMatrix(covinet_70, true_factor)$table)

round(test70$detail$se, 3)
round(test70$detail$sp, 3)
round(test70$detail$pv.pos, 3)
round(test70$detail$pv.neg, 3)

# Covnet
covnet_roc = roc(covnet_int, true, pred)
auc(covnet_roc)
ci.auc(covnet_roc)
covnet_30 = factor(ifelse(covnet_int$pred>0.30,1,0), levels=c(0,1))
test30 = epi.tests(confusionMatrix(covnet_30, true_factor)$table)
covnet_40 = factor(ifelse(covnet_int$pred>0.40,1,0), levels=c(0,1))
test40 = epi.tests(confusionMatrix(covnet_40, true_factor)$table)
covnet_50 = factor(ifelse(covnet_int$pred>0.50,1,0), levels=c(0,1))
test50 = epi.tests(confusionMatrix(covnet_50, true_factor)$table)
covnet_60 = factor(ifelse(covnet_int$pred>0.60,1,0), levels=c(0,1))
test60 = epi.tests(confusionMatrix(covnet_60, true_factor)$table)
covnet_70 = factor(ifelse(covnet_int$pred>0.70,1,0), levels=c(0,1))
test70 = epi.tests(confusionMatrix(covnet_70, true_factor)$table)

round(test70$detail$se, 3)
round(test70$detail$sp, 3)
round(test70$detail$pv.pos, 3)
round(test70$detail$pv.neg, 3)

# ROC Curves
roc.list=list("Mixed-Effects"=ME_roc, "Bai et al."=bai_roc, "Covinet"=covinet_roc, "Covnet"=covnet_roc)
ci.list <- lapply(roc.list, ci.se, specificities = seq(0, 1, l = 743))
dat.ci.list <- lapply(ci.list, function(ciobj) 
  data.frame(x = as.numeric(rownames(ciobj)),
             lower = ciobj[, 1],
             upper = ciobj[, 3]))
p = ggroc(roc.list, aes = c("color"), "size"=1.2) +
  labs(x="Specificity", y="Sensitivity", color="Model")+
  ylim(0,1) +
  xlim(1,0)+
  theme(legend.position="none")+
  theme_minimal()
for(i in 1:4) {
  p <- p + geom_ribbon(
    data = dat.ci.list[[i]],
    aes(x = x, ymin = lower, ymax = upper),
    fill = i+1,
    alpha = 0.2,
    inherit.aes = F) 
} 
p + theme(text = element_text(size = 20))  
ggsave('figures/fig4/fig4a.eps', device=cairo_ps)

# Calibration Curves

cairo_ps('figures/fig5/fig5a.eps', pointsize = 20)
bai_int$pred[bai_int$pred==0]=1e-5
bai_int$pred[bai_int$pred==1]=1-1e-5
val.prob.ci.2(bai_int$pred, bai_int$true,  dostats = F, riskdist = FALSE)
dev.off()

cairo_ps('figures/fig5/fig5b.eps', pointsize = 20)
covinet_int$pred[covinet_int$pred==0]=1e-5
covinet_int$pred[covinet_int$pred==1]=1-1e-5
val.prob.ci.2(covinet_int$pred, covinet_int$true,  dostats = F, riskdist = FALSE)
dev.off()

cairo_ps('figures/fig5/fig5c.eps', pointsize = 20)
covnet_int$pred[covnet_int$pred==0]=1e-5
covnet_int$pred[covnet_int$pred==1]=1-1e-5
val.prob.ci.2(covnet_int$pred, covnet_int$true,  dostats = F, riskdist = FALSE)
dev.off()

cairo_ps('figures/fig5/fig5d.eps', pointsize = 20)
ME_int$pred[ME_int$pred==0]=1e-5
ME_int$pred[ME_int$pred==1]=1-1e-5
val.prob.ci.2(ME_int$pred, ME_int$true,  dostats = F, riskdist = FALSE)
dev.off()

# Decision curves
df1 = data.frame(true=as.numeric(ME_int$true), "Mixed-Effects"=as.numeric(ME_int$pred))
cairo_ps('figures/fig6/fig6.eps')
dca(data=df1, outcome="true", predictors="Mixed.Effects", probability = TRUE)
dev.off()

