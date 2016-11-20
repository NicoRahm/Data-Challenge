# Data-Challenge
AXA Data Challenge for the Machine Learning I course at Ecole Polytechnique 

## 1. Project description
Whether in a contact center or bank branch environment, workforce managers face the constant challenge of balancing the priorities of service levels and labour costs. In the case where the demand (inbound calls, outbound calls, emails, web chats, etc.) is greater than supply (the agents themselves), the price, in the form of reduced service levels, falling customer satisfaction and poor agent morale, rises. On the other side, when supply is greater than demand, service levels tend to improve, but at the cost of idle and unproductive agents. The key to optimising the bottom line performance of a contact centre is to find a harmonious balance between supply and demand. This bottom line performance is directly impacted by the direct costs of hiring and employing your agents, but it is also influenced by client satisfaction, agent morale and other factors. Taking all the above into consideration, the basis of any good staffing plan is an accurate workload forecast. An accurate forecast gives us the opportunity to predict workload in order to get the right number of staff in place to handle it.
The specific project constitutes an AXA data challenge, where its purpose is to apply data mining
and machine learning techniques for the development of an inbound call forecasting system. The forecasting system should be able to predict the number of incoming calls for the AXA call center in France, on a per “half-hour” time slot basis. The prediction is for seven days ahead in time.

## 2. Chosen approach

At first, we chosed to implement the XGBoost algorithm to forecast the number of calls in each centers. First, we process the data to extract the features of interest, then we train one model for each assignement center (the program displays cross-validation errors to assess the liability of the models) and finally we test the models with unseen data from a submission file.

## 3. Required libraries 

- xgboost
- pandas
- datetime
- numpy
- sklearn
- workalendar
