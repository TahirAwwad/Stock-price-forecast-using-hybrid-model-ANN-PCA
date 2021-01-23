### Tahir Awwad
## Final Project ( Stock price forecast using hybrid model of ANN & PCA)
# Date :"5/29/2020"

# Load Packages
pckgs <- c("tidyverse","tidyr",'lubridate',"dplyr","ggplot2","tidyquant","factoextra","FactoMineR","fpp2",'tseries',"xts","neuralnet","nnfor")
#lapply(pckgs ,install.packages)
lapply(pckgs ,library,c=T)
if (!require(BatchGetSymbols)) install.packages('BatchGetSymbols')

# Acquire the EUROIndex50 daily Price

ndx50_raw <- tq_get("^STOXX50E", from = "2007-03-30", to = "2020-06-20")

tickers <- c("FRE.DE","PHIA.AS","ORA.PA","OR.PA",
             "ASML.AS","IBE.MC","SU.PA","BN.PA",
             "DTE.DE","SAN.PA","ITX.MC","AI.PA",
             "ENEL.MI","ENGI.PA","G.MI","CA.PA",
             "ALV.DE","BMW.DE","MC.PA","DPW.DE",
             "ABI.BR","SAF.PA","BAYN.DE","DBK.DE","INGA.AS",
             "BNP.PA","ENI.MI",	"BBVA.MC","AIR.PA","EI.PA")

ndx_comp <- BatchGetSymbols(tickers = tickers, 
                            first.date = "2007-03-30",
                            last.date = "2020-06-20", 
                            freq.data = "daily")

#################

# Data Cleansing

ndx50 <- ndx50_raw[,c(-1,-7,-8)]
ndx50$date <- as.Date(as.character(ndx50$date),"%Y-%m-%d")
ndx50 <- ndx50[order(ndx50$date) , ]
ndx50<- ndx50[complete.cases(ndx50),]


#################

# ARIMA Assumptions Validation

# test stationarity visually
tsdisplay(ts(ndx50$close)) # actual price
# test stationarity via ADF test
adf.test(ndx50$close) # can not reject Null Hypothesis

# take the first difference
tsdisplay(ts(diff(ndx50$close)))
# check normall distribution of price
checkresiduals(ts(diff(ndx50$close)))
# test unit-root via augmented DF test
adf.test(ts(diff(ndx50$close))) #  Reject Null Hypothesis of non-stationarity


#################


# Split data
trainset <- ndx50 %>% filter(date<="2016-06-17") # 70% observations
testset <- ndx50 %>%  filter(date>"2016-06-17")  # 30% observations


#################

# ARIMA
set.seed(123)
# log(closing price)
model_arima_log <- auto.arima(ts(log(trainset$close)),
                              stepwise = FALSE,
                              approximation = FALSE,
                              trace = T) 

summary(model_arima_log) # ARIMA(1,1,3) or ARIMA(5,1,0) both give AICC 12647.5

# Forecast Arima for length period of test set 
frcst_model_arima_log <- forecast(model_arima_log,
                                  h=length(testset$close))
#plot forecast
plot(frcst_model_arima_log)
# accuracy in log price
acc_arima<- accuracy(frcst_model_arima_log,
                     log(testset$close)) 
acc_arima # RMSE 0.18093454, MAPE 1.99983793

# fitted residuals
checkresiduals(frcst_model_arima_log$residuals)


#################


# Multilayer Perceptron for time series forecasting

library(nnfor)
fit1 <- mlp(ts(log(trainset$close)),difforder = 0,outplot = TRUE,hd=5)
print(fit1)
plot(fit1)
# forecast 984 values
mlp_frcst<- forecast(fit1,h=length(ts(testset$close,start = 2296)))
plot(mlp_frcst)
lines(ts(log(testset$close),start = 2296),col="purple", cex=0.1)
# accuracy log price
acc_mlp<- accuracy(mlp_frcst,ts(log(testset$close),start = 2296)) # RMSE 0.2479085 , MAPE 2.853002
acc_mlp

#################

# get data ready PCA ANALYSIS
stock_comp <- ndx_comp$df.tickers[,c(1:4,7,8)]
stock_comp$ticker <- as.factor(stock_comp$ticker)
bytickers <- stock_comp[,c(-1,-2,-3)]
# group tickers by date and ticker symbol just in case
bytickers_grouped <- bytickers %>%  group_by(ref.date,ticker)
# transform dataframe into wide format giving each ticker a vector
bytickers_grouped_wide <- bytickers_grouped %>% pivot_wider(names_from = ticker, values_from=price.close)
### we join the index dataframe with tickers dataframe
bytickers_grouped_wide <- rename(bytickers_grouped_wide,date=ref.date)
df_all <- inner_join(ndx50[,c(1,5)],bytickers_grouped_wide,by="date")
df_all <- rename(df_all,index_close="close")

# Implement PCA on Index Component stocks


# conduct PCA on raw_data
res.pca<- PCA(df_all[ ,c(-1,-2)],graph = T)

# extract eigenvalues ( retaines information in the data for each componenet, explained variance)
eig.val <- get_eigenvalue(res.pca)
eig.val
fviz_eig(res.pca,
         addlabels = TRUE,
         main = "Principal Components by Percentage of Variance Explianed",
         xlab = "Principal Components",
         barfill ="grey",
         barcolor="grey",
         choice = "variance")
# graph and information from variables
pcvar <- get_pca_var(res.pca)
pcvar
# Contributions to each principal component
head(pcvar$contrib)
ticker_contrib <- as.data.frame(pcvar$contrib)
#
ticker_contrib_dim1 <- ticker_contrib %>%
  filter(Dim.1 >= 1) %>% select("Dim.1")
#
ticker_contrib_dim2 <- ticker_contrib %>%
  filter(Dim.2 >= 1) %>% select("Dim.2")
#
ticker_contrib_dim3 <- ticker_contrib %>%
  filter(Dim.3 >= 1) %>% select("Dim.3")

# choose tickers of PC1
ticker_contrib_dim1 %>% rownames()
pc1_tickers_names <- c("FRE.DE","PHIA.AS","ORA.PA","OR.PA",
                       "ASML.AS","SU.PA","BN.PA","DTE.DE","SAN.PA",
                       "ITX.MC","AI.PA","ENGI.PA","CA.PA",
                       "ALV.DE", "BMW.DE","MC.PA","DPW.DE","ABI.BR",
                       "SAF.PA","BAYN.DE","DBK.DE","ENI.MI","BBVA.MC","AIR.PA")
# choose tickers of PC2
ticker_contrib_dim2 %>% rownames()
pc2_tickers_names <- c("PHIA.AS","ORA.PA","IBE.MC","DTE.DE","ENEL.MI", "ENGI.PA" ,"G.MI","CA.PA","ALV.DE","DPW.DE","DBK.DE","INGA.AS", "BNP.PA","ENI.MI","BBVA.MC")
# choose tickers of PC3
ticker_contrib_dim3 %>% rownames()
pc3_tickers_names <- c("FRE.DE","PHIA.AS", "OR.PA","ASML.AS" ,"IBE.MC","SAN.PA","ITX.MC","ENEL.MI","CA.PA","BMW.DE","MC.PA","DPW.DE","ABI.BR" , "SAF.PA" , "BAYN.DE", "DBK.DE","INGA.AS", "BNP.PA" , "ENI.MI" , "BBVA.MC","AIR.PA")



#################
# MLP & PC1

# Split combined data sets
train.set <- df_all %>% filter(date<="2016-06-17") # 70% observations
test.set <- df_all %>%  filter(date>"2016-06-17")  # 30% observations
#join all independent stocks in a matrix total observations
pc1_xreg_dfall1 <- df_all[,pc1_tickers_names] %>%
  log() %>%
  na.omit() %>%
  as.matrix()

# calculate model
fit1_xreg1 <- mlp(ts(log(train.set$index_close[1:2278])),
                  xreg =pc1_xreg_dfall1,
                  difforder = 0,outplot = TRUE)
# Univariate lags: (1,4) 16 regressors included.
print(fit1_xreg1)
plot(fit1_xreg1)
# forecast
mlp_frcst1<- forecast(fit1_xreg1,
                      h=976,
                      xreg=pc1_xreg_dfall1)
plot(mlp_frcst1)
# accuracy log price
acc_pcnn<- accuracy(mlp_frcst1,
                    ts(log(test.set$index_close),start = 2296)) # RMSE 0.1201713 , MAPE 1.291922
acc_pcnn
# actual price accuracy
#accuracy(mlp_frcst1$mean,
#         ts(log(test.set$index_close),start = 2296)) # RMSE 0.1201713 , MAPE 1.291922

# compbine all accuracy results
acc_all <- cbind(acc_pcnn,acc_mlp,acc_arima)
#write.csv(acc_all,file="accuracyResults.csv")





# PLOT
Arima_mean <- frcst_model_arima_log$mean %>% as.numeric()
# Start = 2296  End = 3279 , class numeric, length 984 observations
test_price <- testset$close %>% as.numeric()
# Start = 2296  End = 3279 , class numeric, length 984 observations
train_price <- trainset$close %>% as.numeric()
# Start = 1  End = 2295 , class numeric, length 2295 observations
dayy <- as.Date(ndx50$date)
# length 3279 observations , class date
#nnar_mean <- frcst_model_nnar$mean
# neural net autoregressive predictions
mlp_mean <- mlp_frcst$mean
#multi layer pecptron
pc1_mean <- mlp_frcst1$mean
# mlp wihth pc1 external regressors
############################# 

df1 <- data.frame(ndx50$date,ndx50$close)
# begining of test set
testset$date[1] # "2016-06-20" 
# end of test set
testset$date[1003] # "2020-06-19"
df2 <- data.frame(ndx50$date[2319:3321],testset$close)
# begining and end of ARIMA predictions
df3 <- data.frame(ndx50$date[2319:3321],Arima_mean)

#df4 <- data.frame(ndx50$date[2319:3321],nnar_mean)

df5 <- data.frame(ndx50$date[2319:3321],mlp_mean)

df6 <- data.frame(ndx50$date[2319:3294],pc1_mean)

glimpse(df1)
glimpse(df2)
glimpse(df3)
#glimpse(df4)
glimpse(df5)
glimpse(df6)


colnames(df1) <- c("DateStamp","closingPrice")
colnames(df2) <- c("DateStamp","testPrice")
colnames(df3) <- c("DateStamp","pred_arima")
colnames(df4) <- c("DateStamp","pred_nnar")
colnames(df5) <- c("DateStamp","mlp_mean")
colnames(df6) <- c("DateStamp","pc1_mean")


df_merge <- merge(df2,df3, by="DateStamp", all = T)
df_merge2 <- merge(df1,df_merge, by="DateStamp", all = T)
df_merge3 <- merge(df_merge2,df5, by="DateStamp", all = T)
df_merge4 <- merge(df_merge3,df6, by="DateStamp", all = T)
df_merge5 <- merge(df_merge4,df4, by="DateStamp", all = T)



#choose colors from https://www.schemecolor.com/
stock_colors <- c("closingPrice" = "#6d6e71", 
                  "testPrice" = "#bf5b20", 
                  "pred_arima" = "#EB4C42",
                  #"pred_nnar" = "#8c8700", 
                  "mlp_mean" = "#ad5f7d", 
                  "pc1_mean" = "#8DB601")

theme_bare <- theme(panel.background = element_blank(), 
                    panel.border = element_blank(), 
                    axis.title = element_blank(), 
                    axis.ticks = element_blank(), 
                    panel.grid = element_blank()) 

df_merge4$DateStamp <- as.Date(as.character(df_merge4$DateStamp))

g <- ggplot(df_merge4, aes(x = DateStamp),type="l") +
  geom_line(aes(y = closingPrice), colour="#6d6e71",size=0.5) + 
  geom_line(aes(y = testPrice), colour = "#bf5b20",size=0.5) +
  geom_line(aes(y = exp(pred_arima)), colour = "#EB4C42",size=1) +
  #geom_line(aes(y = exp(pred_nnar)), colour = "#8c8700",size=2) +
  geom_line(aes(y = exp(mlp_mean)), colour = "#ad5f7d",size=2) +
  geom_line(aes(y = exp(pc1_mean)), colour = "#8DB601",size=1.5)

g <- g + theme_bare + coord_fixed(ratio = 0.5) + 
  theme(axis.text = element_text(face = "bold", size = rel(1.5)), 
        legend.position = "none")
g <- g + labs(title = "EURO STOXX 50 Index Price Forecast") 
              #,caption = "Data Source: Yahoo Finance, Forecasts: Researcher Own Analysis")

g <- g + annotate("text", 
                  x = c(as.Date('2010', "%Y"),
                        as.Date('2016', "%Y"),
                        as.Date('2017', "%Y"),
                        #as.Date('2018', "%Y"),
                        as.Date('2019', "%Y"),
                        as.Date('2019', "%Y")), 
                  y = c(3400,3900,3000,2300,4100), 
                  label = c("Training Price","Test Price","Arima","AR-NN","PC_AR-NN"), 
                  color = stock_colors,
                  size=6)
print(g)

ggsave(filename = "Model Compaerission.png", plot = g)

#remove(g)
#######
