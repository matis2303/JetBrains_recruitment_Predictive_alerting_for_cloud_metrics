# Solution Overview: Predictive Alerting for Cloud Metrics

This repository contains the solution for predicting whether an incident will occur within the next `H` time steps based on the previous `W` steps of time-series metrics. 

### 1. Problem Formulation & Data Preprocessing
The problem is approached with sliding-window formulation. For a given `W` various different statistical features get calculated(mean, std, lags, differences). The target is constructed by looking `H` steps ahead to predict whether an incident will occur or not. To ensure model viability data is split chronologically in train and test sets.

### 2. Modeling Choices & Training Setup
I chose XGBoost for this task, because of its efficieny and ability to handle non-linear relationships. To properly evaluate the model and ensure that there is no data leakage i used `Time Series Split Cross Validation`. This approach ensures that the model is trained on past chronological data and evaluated on future unseen data. 

### 3. Evaluation Setup
In cloud metrics incidents classes are often unbalanced. To counter this i used AUCPR as my evaluation method as it's more informative about the model predictions on the rarer class than the other methods avaiable.

Additionaly I tested 2 different prediction thresholds for the evaluation [0.5,0.7]. A lower threshold increases Recall (less missed incidents, but more False Positives), while a higher threshold increases Precision(more missed incidents, but less False Positives)

### 4. Results Analysis
The model tested on the custom dataset with the Threshold of 0.7 gave the least amount of false positives, but skipped some real incidents. A Threshold of 0.5 caught about 50% more incidents, but also had about 10x more False Positives that 0.7 threshold. It is important to decide what is more important during Cloud Predictive Alerting. With more False Positives Engineers can often get used to them and skip important alerts. With a threshold of 0.7 about 80% of alerts will be about a real incident. With a threshold of 0.5 about 25% of alerts will be about a real incident.