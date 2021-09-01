setwd("~/R/PEP 2/modelos_predictivos")

#Este script contiene el an?lisis de variables 

#Funciones
RMSE <- function(error) { sqrt(mean(error^2)) }
rmse2 = function(actual, predicted) {
  a= sqrt(mean((actual - predicted) ^ 2))
  #print(paste0("Ra?Z Error Cuadr?tico Medio: ", a))
}
err_por = function(actual, predicted) {
  a = mean((abs(actual -predicted)/actual)*100)
  # print(paste0("Error porcentual:  ", a))
}

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

#16 num 16 fact
#Data num?rica (286 filas)

data_number = data[,c("G3","G2","G1","Walc","goout",
                      "Medu","Fedu","famrel","Dalc",
                      "freetime","health","age","failures",
                      "absences","studytime","traveltime")]


# Entrenamiento y Testeo
set.seed(123)
sample <- sample.split(data_number$G3, SplitRatio = 0.7)
train <- data_number[sample, ]
test <- data_number[!sample, ]

# Correlaci?n entre variables
cor(data_number)
corrplot(cor(data_number), method="square",title="Regression between numeric columns")
corrgram(data_number, lower.panel=panel.shade, upper.panel=panel.pie, order=TRUE)


# Boxplot data_num?rica

# Se extraen las ausencias para observar
# debido a que tiene muchos outliers

sin_ab= data_number %>%
  dplyr::select(,-c("absences"))
boxplot(sin_ab,
        main = "Data Sin Escalar")
boxplot(scale(sin_ab),
        main = "Data Escalada")


# Boxplot factores 

boxplot(data$G3 ~ data$guardian,
       xlab = "", ylab = "G3",
      las = 2)
points(tapply(data$G3, data$guardian, mean), pch = "x")
abline(h = mean(data$G3),
      col = "red",
     lty = 4)


# Scaling Train

x = data.frame(train) %>%
  dplyr::select(-c("G3"))
y = data.frame(train) %>%
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
x_test = data.frame(test) %>%
  dplyr::select(-c("G3"))
y_test = data.frame(test) %>%
  dplyr::select(c("G3"))

mean_test_x = apply(x_test, 2, mean)
mean_test_y = apply(y_test, 2, mean)
sd_test_x = apply(x_test, 2, sd)
sd_test_y = apply(y_test, 2, sd)

x_test_scaled = scale(x_test, center = mean_test_x, scale = sd_test_x) %>%
  data.frame()
y_test_scaled = scale(y_test, center = mean_test_y, scale = sd_test_y) %>%
  data.frame()

data_test_scaled = cbind(x_test_scaled, y_test_scaled)


# Modelamiento
# Solo con variables num?ricas 

# Modelo 1: OLS

mod_1 = lm(G3 ~ .,
           data = data_scaled)

summary(mod_1)

pred.ols = predict(mod_1, newdata = x_test_scaled)*sd_test_y + mean_test_y 
p = data.frame("G3" = y_test,"OLS" = pred.ols)

Anova(mod_1,
      data = data_number,
      type = 2)


# RMSE Test
ols = rmse2(p$G3,p$OLS)  

# RMSE Train
ols.tr = RMSE(mod_1$residuals) 

#Error % 
err_ols = err_por(p$G3,p$OLS)

plot(mod_1$residuals) 
plot(mod_1)

shapiro.test(mod_1$residuals) #residuos normales
bptest(mod_1) # residuos homo
dwtest(mod_1,
       alternative = "two.sided", 
       iterations = 100) #residuos no correla
bgtest(mod_1,
       order = 1) 
bgtest(mod_1,
       order = 2)


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

pred.elas = predict(elastic_model, newx = x_test_scaled %>% as.matrix())*sd_test_y + mean_test_y

p = data.frame("G3" = y_test,"OLS" = pred.ols,"Elstc" = pred.elas)
colnames(p) = c("G3","OLS","Elstc")

# RMSE Test
els = rmse2(p$G3,p$Elstc)  

# RMSE Train
els.tr = min(elastic_cv$results$RMSE)

#Error % 
err_elas = err_por(p$G3,p$Elstc)


# Modelo 3: Random Forest 

set.seed(123)

mod.rf <- train(G3 ~ ., method = "rf", data = data_scaled)
pred.rf <- predict(mod.rf, x_test_scaled)*sd_test_y + mean_test_y

p = data.frame("G3" = y_test,"OLS" = pred.ols,
               "Elstc" = pred.elas, "Rndm_Frst" = pred.rf)

colnames(p) = c("G3","OLS","Elstc","Rndm_Frst")

# RMSE Train
frst.tr = min(mod.rf$results$RMSE)

# RMSE Test
frst = rmse2(p$G3,p$Rndm_Frst)

#Error % 
err_frst = err_por(p$G3,p$Rndm_Frst)


#Correlaci?n entre modelos
cor(p)
corrplot(cor(p), method="square",title="Regression between numeric columns")
corrgram(p, lower.panel=panel.shade, upper.panel=panel.pie, order=TRUE)


#Modelo Final: Ensemble 

Ensemble = p %>%
  rowwise() %>%
  mutate(Ensemble = mean(c(OLS,
                           Elstc,
                           Rndm_Frst)))  %>%
  ungroup()%>%
  dplyr::select(,c("Ensemble"))

p = data.frame("G3" = y_test,"OLS" = pred.ols,
               "Elstc" = pred.elas, "Rndm_Frst" = pred.rf,
               "Ensemble" = Ensemble)

colnames(p) = c("G3","OLS","Elstc","Rndm_Frst","Ensemble")

# RMSE Train
nsmbl = rmse2(p$G3,p$Ensemble)
#Error %
err_nsmbl= err_por(p$G3,p$Ensemble)


#Comparaci?n de m?tricas 

cor(p)

compar = data.frame("RMSE" = c("Train",
                               "Test",
                               "Error %"),
                    "OLS" =c(ols.tr,ols,err_ols),
                    "Elastic" =c(els.tr,els,err_elas),
                    "Forest" =c(frst.tr,frst,err_frst))
                   # ,"Ensemble" = c(mean(ols.tr,els.tr,frst.tr),
                    #               nsmbl,err_nsmbl))


#Grafico Pred v/s G3_test
p %>%
  gather(key = "method", value = "value",
         -G3) %>%
  ggplot() +
  geom_point(aes(x = G3,
                 y = value,
                 colour = method),
             alpha = 0.8) +
  theme_classic() +
  theme(legend.position = "bottom",
        panel.border = element_rect(fill = NA, colour = "black")) +
  labs(y = "Predicted",
       colour = "Method",
       title = paste("Elstc", round(els,3),
                     ", OLS", round(ols,3),
                     ", Rndm", round(frst,3),
                     ", Ensmbl", round(nsmbl,3)))



# Finally 
# It's done 
