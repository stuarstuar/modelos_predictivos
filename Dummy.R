setwd("~/R/PEP 2")

#Funciones
RMSE <- function(error) { sqrt(mean(error^2)) }
rmse2 = function(actual, predicted) {
  a= sqrt(mean((actual - predicted) ^ 2))
  #print(paste0("RaíZ Error Cuadrático Medio: ", a))
}
err_por = function(actual, predicted) {
  a = mean((abs(actual -predicted)/actual)*100)
 # print(paste0("Error porcentual:  ", a))
}

install.packages('fastDummies')
library(ISLR)
library(tidyverse)
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
library(fastDummies)

# Se extraen los ceros de la variable respuesta 
data = read.csv("Datos.csv") %>%
  filter(G3 != 0)

#16 num 16 fact
#Data numérica (286 filas)
data <- dummy_cols(data, select_columns = 'sex')
data <- dummy_cols(data, select_columns = 'Mjob')
data <- dummy_cols(data, select_columns = 'Fjob')
data <- dummy_cols(data, select_columns = 'address')
data <- dummy_cols(data, select_columns = 'famsize')
data <- dummy_cols(data, select_columns = 'Pstatus')
data <- dummy_cols(data, select_columns = 'reason')
data <- dummy_cols(data, select_columns = 'guardian')
data <- dummy_cols(data, select_columns = 'schoolsup')
data <- dummy_cols(data, select_columns = 'famsup')
data <- dummy_cols(data, select_columns = 'paid')
data <- dummy_cols(data, select_columns = 'activities')
data <- dummy_cols(data, select_columns = 'internet')
data <- dummy_cols(data, select_columns = 'nursery')
data <- dummy_cols(data, select_columns = 'higher')
data <- dummy_cols(data, select_columns = 'romantic')

data = data[,-c(1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22)]

data_number = data

#[,c("G3","G2","G1","age",
 #                     "studytime","failures",
  #                    "traveltime","absences",
   #                   "Medu","Fedu","famrel",
    #                  "freetime","goout","Dalc"
     #                 ,"Walc","health")]

#Data con factores 
#data_all =

#Entrenamiento y Testeo
set.seed(123)
sample <- sample.split(data_number$G3, SplitRatio = 0.7)
train <- data_number[sample, ]
test <- data_number[!sample, ]

#Correlación entre variables
cor(data_number)
corrplot(cor(data_number[,c(16,47:57)]), method="square",title="Regression between numeric columns")
corrgram(data_number, lower.panel=panel.shade, upper.panel=panel.pie, order=TRUE)

# Sobre G3
# Cor + alta con G1,G2
# Un poco más baja con Fedu y Medu
# Cor - con failures,absences,age,Walc,goout

# A priori se esperaría un modelo:
# G3 = b + bi(g1g2fedumedu) - bj(failabsageWalcgoout) : i[1,4]j[5,9]

# Cabe mencionar que:
# 1. Cor + alta entre Fedu y Medu
# 2. Cor + entre study y G1 es mayor que 2y3
# 3. Cor + ligera entre age y fail,abs 
# 4. Cor - medalta entre study y Walc,Dalc,Freetime
# 5. Cor + alta entre Walc,gout,Dalc


# Boxplot data_numérica

# Se extraen las ausencias para observar
# debido a que tiene muchos outliers

sin_ab= data_number[,-c(8)]
boxplot(sin_ab)
boxplot(scale(sin_ab))

# Escaladas y sin escalar se ve un buen comportamiento
# en cuanto a distribución de sus datos, con pocos outliers 

# Boxplot data_all

boxplot(data_all$G3 ~ data_all$sex,
        xlab = "", ylab = "G3",
        las = 2)
points(tapply(data_all$G3, data_all$sex, mean), pch = "x")
abline(h = mean(data_all$G3),
       col = "red",
       lty = 4)

# Distribución de cada factor de forma independiente

# En general no hay ninguno que presente un comportamiento 
# que haga pensar que pueden influir de manera significativa en G3

# De todas formas hay algunos que tienen una ligera tendencia (higher)
# y se podría hacer un análisis más exhaustivo sobre su significancia  
# pero por temas de tiempo no se trabajará con factores  

# Esto no incluye los factores que están denotados por una escala numérica,
# ya que esos entregan información en la correlación y en boxplot 
#que sí muestran una ligera significancia o tendencia en relación a G3
#(Fedu,Medu,traveltime,goout,Walc)


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
# Solo con variables numéricas 

# Modelo 1: OLS

mod_1 = lm(G3 ~ .,
           data = data_scaled)

summary(mod_1)

pred.ols = predict(mod_1, newdata = x_test_scaled)*sd_test_y + mean_test_y 
p = data.frame("G3" = y_test,"OLS" = pred.ols)

# RMSE Test
ols = rmse2(p$G3,p$OLS)  

# RMSE Train
RMSE(mod_1$residuals) 

#Error % 
err_por(p$G3,p$OLS)

# Residuos parecen ser homocedásticos
plot(mod_1$residuals) 
# Pero el resto no sabría decir 
plot(mod_1)

# Análisis OLS w data_number

# R2 0.9362-p mod ***-RMSE t 0.2455 - RMSE T 0.7923 - Error 6.1234%

# Para verificar su funcionamiento habría que validar
# los test estadísticos que verifiqué en otro script 
# y no los cumple

# G1yG2 muestran ***
# healthyfamrel *
# absences .

# Para mejorar habría que ir sacando variables y 
# analizando como cambia R2 y significancia de las otras 

# Lo cual luego de mucha prueba y error, solo con una o dos
# variables se logra significancia completa
# mas no ayuda a validar supuestos ni a predecir mejor

# Por lo que para mejorar la predicción, se probará con 
# modelos de optimización iterativos


# Modelo 2 Elastic net 

grid = expand.grid(alpha = seq(from = 0.1, # 1 Lasso, 0 Ridge
                               to = 0.9,
                               length.out = 5),
                   lambda = seq(from = 0.001, # penal param
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
min(elastic_cv$results$RMSE)

#Error % 
err_por(p$G3,p$Elstc)

# RMSE t 0.255520 - RMSE T 0.8240 - Error 6.3621%
# Ambos son peores que los del modelo lineal 


# Modelo 3: Random Forest 

set.seed(123)

mod.rf <- train(G3 ~ ., method = "rf", data = data_scaled)
pred.rf <- predict(mod.rf, x_test_scaled)*sd_test_y + mean_test_y

p = data.frame("G3" = y_test,"OLS" = pred.ols,
               "Elstc" = pred.elas, "Rndm_Frst" = pred.rf)

colnames(p) = c("G3","OLS","Elstc","Rndm_Frst")

# RMSE Train
min(mod.rf$results$RMSE)

# RMSE Test
frst = rmse2(p$G3,p$Rndm_Frst)

#Error % 
err_por(p$G3,p$Rndm_Frst)


#RMSE t 0.28476 - RMSE T 0.464885 - Error 3.52 %

# Train es mayor que los anteriores, Test es mucho menor, Error es el menor


# Modelo 4: Artificial Neural Network 

set.seed(123)

nn=neuralnet(G3 ~ .,data=data_scaled, hidden=10,act.fct = "logistic",
             linear.output = TRUE,stepmax=10^5,threshold = 0.01)

Predict=compute(nn,x_test_scaled)

pred.nn = Predict$net.result*sd_test_y+mean_test_y

p = data.frame("G3" = y_test,"OLS" = pred.ols,
               "Elstc" = pred.elas, "Rndm_Frst" = pred.rf,
               "Nrl_Ntwrk" = pred.nn)

colnames(p) = c("G3","OLS","Elstc","Rndm_Frst","Nrl_Ntwrk")

# RMSE Train
ann = rmse2(p$G3,p$Nrl_Ntwrk)

#Error %
err_por(p$G3,p$Nrl_Ntwrk)

# RMSE T 0,698235 - Error 5.459%
# El mejor sigue siendo el random en testeo

# Red neuronal
plot(nn)


#Modelo Final: Ensemble 

Ensemble = p %>%
  rowwise() %>%
  mutate(Ensemble = mean(c(OLS,
                           Elstc,
                           Rndm_Frst)))  %>%
                           #,Nrl_Ntwrk))) %>%
  ungroup()%>%
  dplyr::select(,c("Ensemble"))

p = data.frame("G3" = y_test,"OLS" = pred.ols,
               "Elstc" = pred.elas, "Rndm_Frst" = pred.rf,
               "Nrl_Ntwrk" = pred.nn, "Ensemble" = Ensemble)

colnames(p) = c("G3","OLS","Elstc","Rndm_Frst","Nrl_Ntwrk","Ensemble")

# RMSE Train
nsmbl = rmse2(p$G3,p$Ensemble)

#Error %
err_por(p$G3,p$Ensemble)


#Comparación de RMSE Train 

raices = data.frame("OLS" = ols,
                    "Elstc" = els, 
                    "Rndm_Frst" = frst,
                    "Nrl_Ntwrk" = ann, 
                    "Ensemble" = nsmbl)

#Grafico de Pred v/s G3_test
p[,-c(5)] %>%
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
       title = paste("Elastic", round(els,3),
                     ", OLS", round(ols,3),
                     ", Rndm", round(frst,3)))



