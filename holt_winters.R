library(forecast)
library(ggplot2)

setwd("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge")
data_test = read.csv("rcvcall_tel.csv", header = FALSE, col.names = c("Dates", "RCV_CALLS"))
data_test$Dates = strptime(data_test$Dates, "%Y-%m-%d %H:%M:%S")
n = 44
n_before = 3
error = data.frame(ind = 1:n, e = rep(0,n))
period = 48*7

for (i in 1:n){
  deb = period*i
  n = period*(i+n_before)
  data_ts = msts(data_test[deb:n,]$RCV_CALLS, seasonal.periods=period)
  #plot(data_ts)
  
  fit = HoltWinters(data_ts, seasonal = "add")
  error[i,]$e = fit$SSE
  #plot(forecast(fit, p))
}

ggplot(error) + aes(x = ind, y = e) + geom_point()
plot(forecast(fit, period))
