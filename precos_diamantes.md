## Este é meu projeto final da materia de machine learning
library(tidyverse)
library(glmnet)
library(workflows)
library(parsnip)
library(recipes)
library(rsample)
library(dplyr)
library(kernlab)
library(ranger)
library(ggplot2)  
library(dplyr)
library(rlang)
library(tidymodels) 
library(tidyverse)
library(yardstick)
library(devtools)

data("diamonds")
set.seed(123)

# Divisão dos dados: Treinamento (70%), Teste (15%) e Validação (15%)
diamonds_dividido <- initial_split(diamonds, prop = 0.7, strata = price)
diamonds_treino <- training(diamonds_dividido)

diamonds_teste <- testing(diamonds_dividido)

# Dividindo o conjunto de teste em 50% para teste e 50% para validação
diamonds_teste_dividido <- initial_split(diamonds_teste, prop = 0.5, strata = price)
diamonds_teste <- training(diamonds_teste_dividido)
diamonds_validacao <- testing(diamonds_teste_dividido)


cat("Dimensões do conjunto de treinamento:", dim(diamonds_treino), "\n")
cat("Dimensões do conjunto de teste:", dim(diamonds_teste), "\n")
cat("Dimensões do conjunto de validação:", dim(diamonds_validacao), "\n")
cat("\nTamanho do conjunto de treinamento:", nrow(diamonds_treino), "\n")
cat("Tamanho do conjunto de teste:", nrow(diamonds_teste), "\n")
cat("Tamanho do conjunto de validação:", nrow(diamonds_validacao), "\n")

# Pré-processamento dos dados (normalização e transformação de variáveis categóricas)
diamonds_receita <- recipe(price ~ ., data = diamonds_treino) %>%
  step_normalize(all_numeric_predictors()) %>%  
  step_dummy(all_nominal_predictors())          

# Definição dos modelos
lasso_model <- linear_reg(penalty = 0.1, mixture = 1) %>% set_engine("glmnet")
rf_model <- rand_forest(trees = 500, mode = "regression") %>% set_engine("ranger")
svr_model <- svm_rbf(cost = 1, rbf_sigma = 0.1, mode = "regression") %>% set_engine("kernlab")

# Definição dos workflows (modelos + pré-processamento)
lasso_wf <- workflow() %>% add_model(lasso_model) %>% add_recipe(diamonds_receita)
rf_wf <- workflow() %>% add_model(rf_model) %>% add_recipe(diamonds_receita)
svr_wf <- workflow() %>% add_model(svr_model) %>% add_recipe(diamonds_receita)

# Ajustar os modelos
lasso_fit <- fit(lasso_wf, data = diamonds_treino)
rf_fit <- fit(rf_wf, data = diamonds_treino)
svr_fit <- fit(svr_wf, data = diamonds_treino)

# Previsões nos dados de teste
lasso_preds <- predict(lasso_fit, new_data = diamonds_teste) %>% bind_cols(diamonds_teste)
rf_preds <- predict(rf_fit, new_data = diamonds_teste) %>% bind_cols(diamonds_teste)
svr_preds <- predict(svr_fit, new_data = diamonds_teste) %>% bind_cols(diamonds_teste)

# Desempenho dos modelos
lasso_metrics <- metrics(lasso_preds, truth = price, estimate = .pred)
rf_metrics <- metrics(rf_preds, truth = price, estimate = .pred)
svr_metrics <- metrics(svr_preds, truth = price, estimate = .pred)

cat("\nDesempenho do modelo Lasso:\n")
(lasso_metrics)
cat("\nDesempenho do modelo Random Forest:\n")
print(rf_metrics)
cat("\nDesempenho do modelo SVM:\n")
print(svr_metrics)

# Gráficos

# Preço dos diamantes
ggplot(diamonds_treino, aes(x = price)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Distribuição do Preço dos Diamantes", x = "Preço", y = "Contagem") +
  theme_minimal()

# Modelos x Preço real (Gráfico de dispersão)
ggplot() +
  geom_point(data = lasso_preds, aes(x = price, y = .pred, color = "Lasso"), alpha = 0.5) +
  geom_point(data = rf_preds, aes(x = price, y = .pred, color = "Random Forest"), alpha = 0.5) +
  geom_point(data = svr_preds, aes(x = price, y = .pred, color = "SVM"), alpha = 0.5) +
  labs(title = "Previsões vs Preço Real", x = "Preço Real", y = "Preço Previsto") +
  scale_color_manual(values = c("Lasso" = "blue", "Random Forest" = "green", "SVM" = "red")) +
  theme_minimal() +
  theme(legend.title = element_blank(), legend.position = "bottom")


# Erro absoluto dos modelos
lasso_preds <- lasso_preds %>%
  mutate(lasso_error = abs(price - .pred))
rf_preds <- rf_preds %>%
  mutate(rf_error = abs(price - .pred))
svr_preds <- svr_preds %>%
  mutate(svr_error = abs(price - .pred))

# Comparando os erros absolutos
ggplot() +
  geom_boxplot(data = lasso_preds, aes(y = lasso_error, x = "Lasso", fill = "Lasso"), alpha = 0.5) +
  geom_boxplot(data = rf_preds, aes(y = rf_error, x = "Random Forest", fill = "Random Forest"), alpha = 0.5) +
  geom_boxplot(data = svr_preds, aes(y = svr_error, x = "SVM", fill = "SVM"), alpha = 0.5) +
  labs(title = "Distribuição do Erro Absoluto por Modelo", x = "Modelo", y = "Erro Absoluto") +
  scale_fill_manual(values = c("Lasso" = "blue", "Random Forest" = "green", "SVM" = "red")) +
  theme_minimal() +
  theme(legend.position = "none")

