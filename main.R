library(data.table)
library(forecast)
library(ggplot2)

setwd("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge")
d = fread("train_2011_2012_2013.csv", select = c("DATE", "CSPL_RECEIVED_CALLS", "ASS_ASSIGNMENT"))
d = aggregate(d$CSPL_RECEIVED_CALLS ,list( ASSIGNMENT = d[,ASS_ASSIGNMENT], DATE = d[,DATE]), sum)

library(timeDate)
time_to_predict = c("2012-12-28 00:00:00.000", "2013-02-02 00:00:00.000", "2013-03-06 00:00:00.000",
                    "2013-04-10 00:00:00.000", "2013-05-13 00:30:00.000", "2013-06-12 00:00:00.000",
                    "2013-07-16 00:00:00.000", "2013-08-15 00:00:00.000", "2013-09-14 00:00:00.000",
                    "2013-10-18 00:00:00.000", "2013-11-20 00:00:00.000", "2013-12-22 00:00:00.000")

time_to_predict = strptime(time_to_predict, format = "%Y-%m-%d %H:%M:%S.000")

time_past = time_to_predict - 60*60*24*42

train = d[d$ASSIGNMENT == "Téléphonie", c("DATE", "x")]
for (i in 1:length(time_to_predict)){
  train_d = train[train$DATE < time_to_predict[i],]
  train_d = train_d[train_d$DATE >= time_past[i],]
  train_d$x = train_d$x/sd(train_d$x)
  period = 48*7
  data_ts = msts(train_d$x, seasonal.periods=c(48, period))
  fit = HoltWinters(data_ts, seasonal = "add")
  plot(forecast(fit, period))
}
