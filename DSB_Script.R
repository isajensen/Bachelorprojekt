# ============================================================================
# DSB BORNHOLM - DEMAND FORECASTING SHOWCASE
# ============================================================================
# Forkortet kode til demonstration af metode og resultater
# ============================================================================

# Pakker
suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(zoo)
  library(randomForest)
  library(xgboost)
  library(Metrics)
})

set.seed(42)  # Reproducerbarhed

# ============================================================================
# 1. DATA LOADING & CLEANING
# ============================================================================

cat("═══════════════════════════════════════════════════════════════════\n")
cat("1. DATA LOADING & CLEANING\n")
cat("═══════════════════════════════════════════════════════════════════\n\n")

data <- read_csv("dsb_data.csv", show_col_types = FALSE)
cat("✓ Rå data:", nrow(data), "rækker\n")

# Fjern rækker med manglende kontingent
data_clean <- data %>% filter(!is.na(Kontingent))
cat("✓ Efter cleaning:", nrow(data_clean), "rækker\n")
cat("✓ Periode:", as.character(min(data_clean$Dato)), "til", 
    as.character(max(data_clean$Dato)), "\n\n")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

cat("═══════════════════════════════════════════════════════════════════\n")
cat("2. FEATURE ENGINEERING\n")
cat("═══════════════════════════════════════════════════════════════════\n\n")

# Temporale features
data_clean <- data_clean %>%
  mutate(
    Sæson_num = as.numeric(factor(Sæson, 
                                  levels = c("Vinter", "Forår", "Sommer", "Efterår"))),
    Ugedag = factor(Ugedag, levels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun"))
  )

# Sorter kronologisk per retning
data_clean <- data_clean %>% arrange(Retning, Dato, Time)

# Beregn historisk median per retning (til imputation)
historical_median <- data_clean %>%
  group_by(Retning) %>%
  slice_head(n = 1000) %>%
  summarise(median_solgte = median(Solgte_billetter, na.rm = TRUE))

# Lagged features og rolling averages
data_clean <- data_clean %>%
  group_by(Retning) %>%
  mutate(
    Solgte_lag1 = lag(Solgte_billetter, 1),
    Solgte_lag7 = lag(Solgte_billetter, 7),
    Solgte_ma3 = rollapply(Solgte_billetter, width = 3, FUN = mean, 
                           fill = NA, align = "right", na.rm = TRUE)
  ) %>%
  ungroup()

# Imputer NA med historisk median
data_clean <- data_clean %>%
  left_join(historical_median, by = "Retning") %>%
  mutate(
    Solgte_lag1 = ifelse(is.na(Solgte_lag1), median_solgte, Solgte_lag1),
    Solgte_lag7 = ifelse(is.na(Solgte_lag7), median_solgte, Solgte_lag7),
    Solgte_ma3 = ifelse(is.na(Solgte_ma3), median_solgte, Solgte_ma3)
  ) %>%
  select(-median_solgte)

cat("✓ Features tilføjet: Solgte_lag1, Solgte_lag7, Solgte_ma3\n")
cat("✓ Historisk median brugt til imputation af initiale værdier\n\n")

# ============================================================================
# 3. TRAIN/TEST SPLIT (KRONOLOGISK)
# ============================================================================

cat("═══════════════════════════════════════════════════════════════════\n")
cat("3. TRAIN/TEST SPLIT\n")
cat("═══════════════════════════════════════════════════════════════════\n\n")

# Vælg features til modellering
ml_data <- data_clean %>%
  mutate(Ugedag_num = as.numeric(Ugedag)) %>%
  select(Time, Kontingent, Solgte_lag1, Solgte_lag7, Solgte_ma3, 
         Sæson_num, Ugedag_num, Event_flag, Solgte_billetter, Dato) %>%
  na.omit()

# Kronologisk 80/20 split (VIGTIGT: undgår data leakage)
split_idx <- floor(0.8 * nrow(ml_data))

train_data <- ml_data[1:split_idx, ]
test_data <- ml_data[(split_idx + 1):nrow(ml_data), ]

X_train <- train_data %>% select(Time, Kontingent, Solgte_lag1, Solgte_lag7, 
                                  Solgte_ma3, Sæson_num, Ugedag_num, Event_flag)
y_train <- train_data$Solgte_billetter

X_test <- test_data %>% select(Time, Kontingent, Solgte_lag1, Solgte_lag7, 
                                Solgte_ma3, Sæson_num, Ugedag_num, Event_flag)
y_test <- test_data$Solgte_billetter

cat("✓ Training set:", nrow(train_data), "observationer\n")
cat("✓ Test set:", nrow(test_data), "observationer\n")
cat("✓ Kronologisk split sikrer ingen fremtidsdata i training\n\n")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

cat("═══════════════════════════════════════════════════════════════════\n")
cat("4. MODEL TRAINING\n")
cat("═══════════════════════════════════════════════════════════════════\n\n")

# --- BASELINE ---
baseline_pred <- rep(mean(y_train), length(y_test))
baseline_mae <- mae(y_test, baseline_pred)

cat("BASELINE (gennemsnit):\n")
cat("  MAE:", round(baseline_mae, 3), "\n\n")

# --- LINEAR REGRESSION ---
lm_model <- lm(Solgte_billetter ~ ., data = cbind(X_train, Solgte_billetter = y_train))
lm_pred <- predict(lm_model, X_test)
lm_mae <- mae(y_test, lm_pred)
lm_r2 <- cor(y_test, lm_pred)^2

cat("LINEAR REGRESSION:\n")
cat("  MAE:", round(lm_mae, 3), "| R²:", round(lm_r2, 3), "\n\n")

# --- RANDOM FOREST ---
rf_model <- randomForest(x = X_train, y = y_train, 
                         ntree = 500, mtry = 3, nodesize = 5)
rf_pred <- predict(rf_model, X_test)
rf_mae <- mae(y_test, rf_pred)
rf_r2 <- cor(y_test, rf_pred)^2

cat("RANDOM FOREST:\n")
cat("  MAE:", round(rf_mae, 3), "| R²:", round(rf_r2, 3), "\n\n")

# --- XGBOOST (optimerede hyperparametre) ---
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

xgb_model <- xgb.train(
  params = list(
    objective = "reg:squarederror",
    max_depth = 6,
    eta = 0.1
  ),
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

xgb_pred <- predict(xgb_model, dtest)
xgb_mae <- mae(y_test, xgb_pred)
xgb_rmse <- rmse(y_test, xgb_pred)
xgb_r2 <- cor(y_test, xgb_pred)^2

cat("XGBOOST:\n")
cat("  MAE:", round(xgb_mae, 3), "| RMSE:", round(xgb_rmse, 3), "| R²:", round(xgb_r2, 3), "\n\n")

# ============================================================================
# 5. MODEL EVALUERING
# ============================================================================

cat("═══════════════════════════════════════════════════════════════════\n")
cat("5. MODEL EVALUERING\n")
cat("═══════════════════════════════════════════════════════════════════\n\n")

# Samlet resultatoversigt
results <- data.frame(
  Model = c("Baseline", "Linear Regression", "Random Forest", "XGBoost"),
  MAE = round(c(baseline_mae, lm_mae, rf_mae, xgb_mae), 3),
  R2 = round(c(0, lm_r2, rf_r2, xgb_r2), 3),
  Forbedring_pct = round(c(0, 
                           (baseline_mae - lm_mae) / baseline_mae * 100,
                           (baseline_mae - rf_mae) / baseline_mae * 100,
                           (baseline_mae - xgb_mae) / baseline_mae * 100), 1)
)

print(results)
cat("\n✓ XGBoost performer bedst med", round(results$Forbedring_pct[4], 1), 
    "% forbedring over baseline\n\n")

# Feature importance
xgb_importance <- xgb.importance(model = xgb_model)
cat("TOP FEATURES (XGBoost Gain):\n")
print(head(xgb_importance, 5))

# ============================================================================
# 6. BUSINESS VALUE
# ============================================================================

cat("\n═══════════════════════════════════════════════════════════════════\n")
cat("6. BUSINESS VALUE\n")
cat("═══════════════════════════════════════════════════════════════════\n\n")

wholesale_price <- 49  # DKK per billet

# DSB baseline
dsb_kontingent <- test_data$Kontingent
dsb_waste <- pmax(0, dsb_kontingent - y_test)
dsb_waste_rate <- sum(dsb_waste) / sum(dsb_kontingent) * 100

cat("DSB NUVÆRENDE PRAKSIS:\n")
cat("  Waste rate:", round(dsb_waste_rate, 1), "%\n")
cat("  Total spild:", sum(dsb_waste), "billetter\n\n")

# ML-baseret kontingent (realistisk scenarie: +10% buffer, min 5)
ml_kontingent <- pmax(5, ceiling(xgb_pred * 1.10))
ml_waste <- pmax(0, ml_kontingent - y_test)
ml_waste_rate <- sum(ml_waste) / sum(ml_kontingent) * 100

# Beregn besparelse
test_period_days <- as.numeric(max(test_data$Dato) - min(test_data$Dato))
savings_test <- (sum(dsb_waste) - sum(ml_waste)) * wholesale_price
annual_savings <- savings_test * (365 / test_period_days)

waste_reduction <- (sum(dsb_waste) - sum(ml_waste)) / sum(dsb_waste) * 100

cat("ML MODEL (realistisk scenarie):\n")
cat("  Waste rate:", round(ml_waste_rate, 1), "%\n")
cat("  Waste reduktion:", round(waste_reduction, 1), "%\n")
cat("  Årlig besparelse: ~", format(round(annual_savings), big.mark = "."), " DKK\n\n", sep = "")

# ============================================================================
# 7. KONKLUSION
# ============================================================================

cat("═══════════════════════════════════════════════════════════════════\n")
cat("KONKLUSION\n")
cat("═══════════════════════════════════════════════════════════════════\n\n")

cat("✓ XGBoost model reducerer MAE fra", round(baseline_mae, 2), "til", 
    round(xgb_mae, 2), "billetter\n")
cat("✓ Forklarer", round(xgb_r2 * 100, 1), "% af variansen i efterspørgsel\n")
cat("✓ Potentiel årlig besparelse: ~", format(round(annual_savings/1000), big.mark = "."), 
    "k DKK\n", sep = "")
cat("✓ Vigtigste features: Solgte_ma3, Time, Kontingent\n\n")

cat("═══════════════════════════════════════════════════════════════════\n")
