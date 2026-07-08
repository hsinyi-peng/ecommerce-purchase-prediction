# ===============================
# E-commerce Purchase Prediction
# Predicting purchase completion from session behavior with logistic
# regression, CART, and random forest
# ===============================
#
# Run from the project root, e.g.:
#   Rscript analysis/purchase_prediction_analysis.R
#
# Reads:  data/ecommerce_customer_behavior.csv
# Writes: outputs/*.png

# --------- 0) Setup ---------
pkg_needed <- c("dplyr","ggplot2","ROCR","rpart","rpart.plot","randomForest")
pkg_to_install <- pkg_needed[!(pkg_needed %in% installed.packages()[,"Package"])]
if (length(pkg_to_install) > 0) install.packages(pkg_to_install, dependencies = TRUE)

library(dplyr)
library(ggplot2)
library(ROCR)
library(rpart)
library(rpart.plot)
library(randomForest)

dir.create("outputs", showWarnings = FALSE)

# --------- 1) Load Data ---------
df <- read.csv("data/ecommerce_customer_behavior.csv")

# Summary of the dataset
summary(df)

# --------- 2) Feature Selection & Cleaning ---------
df_clean <- df %>%
  select(purchased, unit_price, discount_percent, pages_viewed,
         time_on_site_sec, device_type, user_type, marketing_channel) %>%
  na.omit()
df_clean$purchased         <- as.factor(df_clean$purchased)
df_clean$device_type       <- as.factor(df_clean$device_type)
df_clean$marketing_channel <- as.factor(df_clean$marketing_channel)
df_clean$user_type         <- as.factor(df_clean$user_type)

# --------- 3) Exploratory Data Analysis (EDA) ---------
png("outputs/plot_price_distribution.png", width = 900, height = 600)
hist(df$unit_price, main="Distribution of Product Prices",
     col="skyblue", xlab="Price ($)")
dev.off()

p_density <- ggplot(df_clean, aes(x = time_on_site_sec, fill = purchased)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("grey", "orange"),
                    labels = c("Not Purchased (0)", "Purchased (1)")) +
  labs(title = "Density of Time on Site by Purchase Outcome",
       subtitle = "Evidence of Purchases Occurring in the 'Long Tail'",
       x = "Time on Site (Seconds)",
       y = "Density",
       fill = "Outcome") +
  theme_minimal()
ggsave("outputs/plot_time_on_site_density.png", p_density, width = 9, height = 6)

png("outputs/plot_purchases_by_rating.png", width = 900, height = 600)
counts <- table(df$purchased, df$rating)
barplot(counts, main="Purchases vs Rating",
        xlab="Rating", col=c("grey", "orange"),
        legend = c("Not Purchased", "Purchased"), beside=TRUE)
dev.off()

rating_table <- table(df$purchased, df$rating)
rownames(rating_table) <- c("Not Purchased (0)", "Purchased (1)")
colnames(rating_table) <- c("1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars")
print(rating_table)

png("outputs/plot_time_on_site_by_purchase.png", width = 900, height = 600)
boxplot(time_on_site_sec ~ purchased, data=df,
        main="Time on Site by Purchase Outcome",
        xlab="Purchased (0=No, 1=Yes)", ylab="Seconds", col=c("white", "lightgreen"))
dev.off()

# --------- 4) Train/Test Split (80/20) ---------
set.seed(123)
idx <- sample(nrow(df_clean), nrow(df_clean)*0.8)
train_data <- df_clean[idx, ]
test_data  <- df_clean[-idx, ]

# --------- 5) Logistic Regression ---------
log_model <- glm(purchased ~ ., family=binomial, data=train_data)

# --------- 6) CART (Classification Tree) ---------
dt_model <- rpart(purchased ~ ., data = train_data, method = "class",
                  control = rpart.control(cp = 0.002))
png("outputs/plot_decision_tree.png", width = 1000, height = 700)
prp(dt_model, type = 2, extra = 104,
    main="Pruned Decision Tree: Purchase Intent Logic")
dev.off()
prob_dt <- predict(dt_model, newdata = test_data, type = "prob")[,2]

# --------- 7) Random Forest ---------
rf_model <- randomForest(purchased ~ ., data=train_data, ntree=500, nodesize=1)
png("outputs/plot_rf_varimp.png", width = 900, height = 600)
varImpPlot(rf_model, main = "Variable Importance in Purchase Prediction")
dev.off()
prob_log <- predict(log_model, newdata=test_data, type="response")
prob_rf  <- predict(rf_model, newdata=test_data, type="prob")[,2]

# --------- 8) ROC / AUC Comparison ---------
pred_log <- prediction(prob_log, test_data$purchased)
perf_log <- performance(pred_log, "tpr", "fpr")
auc_log  <- unlist(slot(performance(pred_log, "auc"), "y.values"))

pred_dt  <- prediction(prob_dt, test_data$purchased)
perf_dt  <- performance(pred_dt, "tpr", "fpr")
auc_dt   <- unlist(slot(performance(pred_dt, "auc"), "y.values"))

pred_rf  <- prediction(prob_rf, test_data$purchased)
perf_rf  <- performance(pred_rf, "tpr", "fpr")
auc_rf   <- unlist(slot(performance(pred_rf, "auc"), "y.values"))

png("outputs/plot_roc_comparison.png", width = 900, height = 700)
plot(perf_log, col="blue", lwd=2, main="ROC Comparison: 3-Model Evaluation")
plot(perf_rf, col="red", add=TRUE, lwd=2)
plot(perf_dt, col="darkgreen", add=TRUE, lwd=2)
abline(0, 1, lty=2, col="gray")
text(0.6, 0.4, paste("Logistic AUC =", round(auc_log, 4)), col="blue", adj=0)
text(0.6, 0.3, paste("Decision Tree AUC =", round(auc_dt, 4)), col="darkgreen", adj=0)
text(0.6, 0.2, paste("Random Forest AUC =", round(auc_rf, 4)), col="red", adj=0)
legend("bottomright",
       legend=c("Logistic Regression", "Decision Tree (CART)", "Random Forest"),
       col=c("blue", "darkgreen", "red"), lwd=2)
dev.off()

# --------- 9) Misclassification Rates ---------
get_misrate <- function(model_prob, actual, cutoff=0.3) {
  preds <- ifelse(model_prob > cutoff, 1, 0)
  mean(preds != actual)
}

log_train_prob <- predict(log_model, type = "response")
log_test_prob  <- predict(log_model, newdata = test_data, type = "response")
log_tr_mis <- get_misrate(log_train_prob, train_data$purchased)
log_te_mis <- get_misrate(log_test_prob, test_data$purchased)

dt_train_prob <- predict(dt_model, type = "prob")[,2]
dt_test_prob  <- predict(dt_model, newdata = test_data, type = "prob")[,2]
dt_tr_mis <- get_misrate(dt_train_prob, train_data$purchased)
dt_te_mis <- get_misrate(dt_test_prob, test_data$purchased)

rf_train_prob <- predict(rf_model, type = "prob")[,2]
rf_test_prob  <- predict(rf_model, newdata = test_data, type = "prob")[,2]
rf_tr_mis <- get_misrate(rf_train_prob, train_data$purchased)
rf_te_mis <- get_misrate(rf_test_prob, test_data$purchased)

results_table <- data.frame(
  Model = c("Logistic (GLM)", "Decision Tree (CART)", "Random Forest"),
  Train_Misrate = c(log_tr_mis, dt_tr_mis, rf_tr_mis),
  Test_Misrate = c(log_te_mis, dt_te_mis, rf_te_mis),
  AUC = c(auc_log, auc_dt, auc_rf)
)
cat("\n--- Final Model Comparison (Threshold = 0.3) ---\n")
print(results_table)

# --------- 10) Sensitivity to Cutoff ---------
pcuts <- c(0.2, 0.3, 0.4, 0.5, 0.6)
costs_log <- sapply(pcuts, function(p) get_misrate(log_test_prob, test_data$purchased, p))
costs_dt  <- sapply(pcuts, function(p) get_misrate(dt_test_prob,  test_data$purchased, p))
costs_rf  <- sapply(pcuts, function(p) get_misrate(rf_test_prob,  test_data$purchased, p))
cost_comparison <- data.frame(
  Cutoff = pcuts,
  GLM_Misrate = costs_log,
  DT_Misrate  = costs_dt,
  RF_Misrate  = costs_rf
)
cat("\n--- Misclassification Rate across Different Thresholds ---\n")
print(cost_comparison)

cat("\nAll done. Figures saved to outputs/.\n")
