# Marketing-Analytics

## [Customer Churn: Hurts Sales, Hurts Company](https://github.com/mengyjia/Marketing-Analytics/blob/master/Customer_Churn_Prediction.ipynb)

Customer churn refers to the situation when a customer ends their relationship with a company, and it’s a costly problem. 

<br>Customers are the fuel that powers a business. Loss of customers impacts sales. Further, it’s much more difficult and costly to gain new customers than it is to retain existing customers. As a result, organizations need to focus on reducing customer churn.

<br>This notebook will show you how to predict which customers are going to leave using machine learning.
### Data Description
The dataset used for this notebook is __[IBM Watson Telco Dataset](https://www.ibm.com/communities/analytics/watson-analytics-blog/predictive-insights-in-the-telco-customer-churn-data-set/)__. According to IBM, the business challenge is…

A telecommunications company Telco is concerned about the number of customers leaving their landline business for cable competitors. They need to understand who is leaving. Imagine that you’re an analyst at this company and you have to find out who is leaving and why.

The dataset includes information about:

Customers who left within the last month: The column is called Churn
Services that each customer has signed up for: phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Customer account information: how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers: gender, age range, and if they have partners and dependents
### Brief
This notebook is a great example of adopting machine learning to solve business problems. <br>

1.Deal with imbalanced data. Most business datasets are imbalanced, so it is critical to use oversampling or undersampling methods to reduce the imbalance. However, many data analysts fail to use these techniques because they conduct them in a wrong order. The consequence of wrong order is to "bleed" information to validation data and thus over estimate the performance. In this notebook, I use __SMOTE__ - Synthetic Minority Over-sampling Technique to deal with the imbalanced data, and make __pipelines__ to do __gridsearch__ with 10 fold cross-validation. <br>

2.Various feature engineering. For different kind of variables, we should use different transformation methods. In this notebook, I take __binning__ and __log transformation__ to deal with numerical variables. Also, it is important to test whether your transformation methods improve the correlation with classes.<br>

3.Evaluate models based on __business context__. There are various metrics to assess models. In this notebook, I demonstrate how to choose right metrics which fit your business context most to pick up a suitable model. For this business question, the most important thing is to identify potential churn customers. Therefore, we should improve the ratio of right prediction of churn customers. However, the actions to retain customers might cause costs, so we also need to avoid to give wrong alerts. Technically, we want a model with higher recall rate and precision rate for "1" . Recall (also true positive rate or specificity) is when the actual value is “yes” or "no", how often is the model correct. Therefore, I decide to use f1 score of "1" as the evaluation metric.

4.Decide and __customize__ evaluation metrics at the model building step. Although the model evaluation seems to be the next step of model building, please notice that we use gridsearch to choose the best hyperparameters, while we have to tell gridsearch the criteria of "best". Therefore, we have to think about the model evaluation metric at the model building step. Many analysts conduct gridsearch without setting evaluation metric and then choose the best model according to other metrics which they infer from "business context". This would reduce the productivity and reliability of the model.

### Result
The best models are __Gradient Boosting__ and __Gaussian Naive Bayes__, with f1 core 0.63 and 0.57 respectively. More specifically, the gradient boosting can predict 75% of churn customers and give 54% right alerts, while the gaussian naive bayes can predict 90% of churn customers and give 42% right alerts.


## [Customer Segmentation](https://github.com/mengyjia/Marketing-Analytics/blob/master/Customer%20Segmentation/Customer%20Segmentation.ipynb)

In this notebook, I’ll show how to use k-means clustering to segment customers into distinct groups based on purchasing habits. The dataset is from [instacart-market-basket-analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data) on kaggle. Since there are several tables, and our target is not about table manipulation, I download the combined data directly from [my kaggle kernel](https://www.kaggle.com/jmy666/customer-segments-with-pca/output).

### Data Description

From the kaggle:
The dataset is anonymized and contains a sample of over 3 million grocery orders from more than 200,000 Instacart users. For each user, we provide between 4 and 100 of their orders, with the sequence of products purchased in each order. We also provide the week and hour of day the order was placed, and a relative measure of time between orders. For more information, see the blog post accompanying its public release.

Basicly, each customer has a unique use_id, and might have several orders, which have unique order_id. For each order, there might be different products, belonging to aisles and departments. All products, aisles and departments have unique id and name. Most variables are self-explanatory.

'Reordered' indicates that the customer has a previous order that contains the product. Note that some orders will have no reordered items. You may predict an explicit 'None' value for orders with no reordered items. See the evaluation page for full details. 'order_dow' is the day of week.

### Brief

I use aisles as the criteria of clustering and do not consider the sequence as well as quantity of orders. Therefore, in the table for clustering, each row represents a customer, and each column represents an aisle. The value of each cell is the total number of the specific aisle purchased by that customer. To remove the influence of quantity, I then calculate the proportion of each aisle by dividing the total number of aisles purchased by that customer.

For clustering, I run the models both with and without PCA. The result shows that under low tolerance of variance, PCA can reach similar result and reduce the complexity of models significantly.

Also, I demonstrate how to choose best k (number of clusters) step by step.

### Business Interpretation of Clusters

I understand that only clustering does not create values, and we need to interpret our models. This notebook shows how to gain insights from clusters from different aspects. At first I list top 10 aisles for each cluster and find that they are similar. I realize that this is because some aisles are necessaries of life. Then I remove the influence of necessaries, and list top 10 aisles for each cluster again. The result is really interesting, and we can gain clear insights about different clusters.
