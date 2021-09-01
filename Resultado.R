setwd("~/R/PEP 2")

#Este script entrena el modelo elegido e imprime resultados en csv

library(ISLR)
library(tidyverse)
require(car)
library(caret)
library(keras)
library(neuralnet)
library(Hmisc)
require(glmnet) 
require(lmtest)
library(rgdal)
library(raster)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(corrgram)
library(corrplot)
library(caTools)

# Se extraen los ceros de la variable respuesta 
data = read.csv("Datos.csv") %>%
  filter(G3 != 0)

test_data = read.csv("Test.csv")

#Data numérica (286 filas)

data_number = data[,c("G3","G2","G1",
                      "age","failures",
                      "absences","studytime",
                      "traveltime")]


test_data = test_data %>%
  dplyr::select(c("G2","G1",
                  "age","failures",
                  "absences","studytime",
                  "traveltime"))
# Scaling Train

x = data.frame(data_number) %>%
  dplyr::select(-c("G3"))
y = data.frame(data_number) %>%
  dplyr::select(c("G3"))

mean_x = apply(x, 2, mean)
mean_y = apply(y, 2, mean)
sd_x = apply(x, 2, sd)
sd_y = apply(y, 2, sd)

x_scaled = scale(x, center = mean_x, scale = sd_x) %>%
  data.frame()
y_scaled = scale(y, center = mean_y, scale = sd_y) %>%
  data.frame()

data_scaled = cbind(x_scaled, y_scaled)

# Scaling Test

x_test = data.frame(test_data)

mean_test_x = apply(x_test, 2, mean)
sd_test_x = apply(x_test, 2, sd)

x_test_scaled = scale(x_test, center = mean_test_x, scale = sd_test_x) %>%
  data.frame()


# Modelo 1: OLS

mod_1 = lm(G3 ~ .,
           data = data_scaled)

pred.ols = predict(mod_1, newdata = x_test_scaled)*sd_y + mean_y 

# Modelo 2 Elastic Net 

grid = expand.grid(alpha = seq(from = 0.1, 
                               to = 0.9,
                               length.out = 5),
                   lambda = seq(from = 0.001, 
                                to = 0.1,
                                length.out = 5))

control = trainControl(search = "grid",
                       method = "cv",
                       number = 10)

set.seed(123)

(elastic_cv = train(G3 ~ .,
                    data = data_scaled,
                    method = "glmnet", 
                    trControl = control,
                    metric = "RMSE",
                    tuneGrid = grid))

elastic_cv$bestTune

elastic_model = glmnet(x = x_scaled %>%
                         as.matrix(),
                       y = y_scaled %>%
                         as.matrix(),
                       intercept = TRUE,
                       alpha = elastic_cv$bestTune$alpha,
                       lambda = elastic_cv$bestTune$lambda)

pred.elas = predict(elastic_model, newx = x_test_scaled %>% as.matrix())*sd_y + mean_y


# Modelo 3: Random Forest

set.seed(123)

mod.rf <- train(G3 ~ ., method = "rf", data = data_scaled)
pred.rf <- predict(mod.rf, x_test_scaled)*sd_y + mean_y

p = data.frame("OLS" = pred.ols,
               "Elstc" = pred.elas, 
               "Rndm_Frst" = pred.rf)

colnames(p) = c("OLS","Elstc","Rndm_Frst")


# KNN
# LO VAMOS A PROBAR 

# Ensemble 

Ensemble = p %>%
  rowwise() %>%
  mutate(Ensemble = mean(c(OLS,
                           Elstc,
                           Rndm_Frst)))  %>%
  ungroup()%>%
  dplyr::select(,c("Ensemble"))

p = data.frame("OLS" = pred.ols,
               "Elstc" = pred.elas, "Rndm_Frst" = pred.rf,
               "Ensemble" = Ensemble)

colnames(p) = c("OLS","Elstc","Rndm_Frst","Ensemble")

cor(p)
corrgram(p, lower.panel=panel.shade, upper.panel=panel.pie, order=TRUE)


pred = p %>%
  dplyr::select(,c("Ensemble"))


# PREGUNTAR  
write.csv(pred, file = "", row.names = FALSE)


aa = read.csv("Grupo_8.csv")


