# Online Shopper Purchasing Intention
Written by: Aaron Zhang, Andrew Arochukwu, Qiguang (William) Zhu

## Dataset
We are using the [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) dataset from UCI. The dataset was designed for predicting whether a customer (online shopping session) will end in a purchase. 
The dataset includes 12,330 online browsing sessions, which were collected over a period of one year. Each session represents a different user and has 17
features collected. The features include 10 numerical and 7 categorical, with features ranging from pages visited, time spent, and exit rates, to browser and region. The target feature is the “Revenue” column, which tells us whether or not the user spent money on the site. 

### Data summary
| name | role | type |
| --- | --- | --- |
| Administrative | Feature  | Integer |
| Administrative_Duration | Feature | Integer |
| Informational | Feature | Integer |
| Informational_Duration | Feature | Integer |
| ProductRelated | Feature | Integer |
| ProductRelated_Duration | Feature | Continuous |
| BounceRates | Feature | Continuous |
| ExitRates | Feature | Continuous |
| PageValues | Feature | Integer |
| SpecialDay | Feature | Integer |
| Month | Feature | Categorical |
| OperatingSystems | Feature | Integer |
| Browser | Feature | Integer |
| Region | Feature | Integer |
| TrafficType | Feature | Integer |
| VisitorType | Feature | Categorical |
| Weekend | Feature | Binary |
| Revenue | Target | Binary |

### Data Preparation

Our data preparation process involved three main steps:

#### 1. Loading the Data
We downloaded the Online Shoppers Purchasing Intention dataset from UCI containing 12,330 customer browsing sessions with 18 total features (17 inputs + 1 target). Each row represents one online shopping session from a unique customer.

#### 2. Handling Categorical Data
We had 7 features (including month, browser and region) which were not numerical. We converted these to one-hot encoded features, while dropping the first feature to avoid multicollinearity.

#### 3. Scaling and Log Transformation
Our numerical features (like the amout of time spent on a page or bounce rates) had very different scales. We use standard scale to ensure all features a similar range. We also used log-transformations to remove skewness and normalize the features (see **Figure 1 & 2**).

#### Final Data:
After these steps, we had 25 total features (the original 10 numerical + expanded categorical features from one-hot encoding) (see **Figure 3** for Heat-Map of final transformed data). We also noted a class-imbalance, less than 20% of the sessions resulted in a purchase, so we in turn let no purchases be the true class. 

## Methodology

We tested 5 statisitcal modelling approaches to predict whether a customer will make a purchase. We chose a mix of interprable and high predictive performance models. 

### 1. **Logistic Regression**
A interpretable method that outputs a probability between 0 and 1 for whether a purchase will occur. 
- **Cross-validation**: We used a 5-fold cross-validation method to produce the best performing regression model. We selected the models which produced the highest accuracy and auc.

### 2. **Neural Network**
This approach allowed us to potentially capture non-linear patterns in the data.
- **Structure**: We chose a simple deep learning model with two hidden layers (64 neurons each).
- **Training**: We trained it for 10 epochs with a batch size of 32. This could have been fine-tuned further, with more epochs and/or a more aggresive learning rate optimizer, however with a small dataset and model architecture this was sufficient. 

### 3. **Random Forest**
An ensemble method that builds many decision trees and combines their predictions. Each tree learns different patterns, and by voting across all trees, overfitting is reduced and this leads to higher accuracy when compared to only using on tree. With random forest we created two models, a baseline model and a tuned model.
- **Baseline Model**: First we trained a Random Forest classifier using default parameters, this was done to establish a performance baseline.
- **Tuned Model**: Using randomized search 5-fold cross-validation we optimized the following key parameters:
    - Number of trees in the forest
    - Maximum depth of each tree
    - Minimum samples required to split an internal node
    - Minimum samples required at each leaf node

### 4. **XGBoost**
This is an advanced gradient boosting implementation that sequentially builds trees, where new trees correct errors made by previous trees. A major difference between Random Forest and XGBoost is that Random Forest builds trees independently while XGBoost learns iteratively, which makes it effective for complex patterns. Similarly to the random forest implementation we built two models, a baseline and a tuned model.
- **Baseline Model**: We established the baseline performance by making use of default XGBoost parameters.
- **Tuned Model**: Using randomized search 5-fold cross-validation we optimized the following  parameters:
    - Number of boosting rounds
    - Maximum tree depth to control model complexity
    - Step size for weight updates
    - Fraction of samples used for each tree
    - Fraction of features used for each tree

### 5. **Naive Bayes Classifier** (naive_bayes_baseline.ipynb)
A probabilistic model that predicts the likelihood of purchase by calculating the probability of each feature value given the customer made or didn't make a purchase. Key steps include:
- **Data Processing**: features with simple scaling and log transform are both being implemented
- **Training**: Learning the distribution of features for each class (purchaser vs non-purchaser)
- **Parameter Tuning**: Testing different smoothing values (1.0, 2.0, 5.0, etc.) and variance thresholds to optimize performance
- **Cross-Validation**: Using 5-fold cross-validation to ensure robust results
- **ROC Analysis**: Computing Receiver Operating Characteristic curves to visualize the trade-off between correctly identifying purchasers and avoiding false alarms

### Evaluation Metrics
For each model, we calculated:
- **Accuracy**: Percentage of predictions that were correct
- **Precision**: Of predicted purchases, how many were actually purchases
- **Recall**: Of actual purchases, how many we correctly identified
- **AUC (Area Under the Curve)**: Measures how well the model separates purchasers from non-purchasers (0.5 = random, 1.0 = perfect)

## Results

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | AUC |
| --- | --- | --- | --- | --- |
| Logistic Regression (Best Acc) | 90.06% | 95.74% | 92.89% | 91.30% |
| Logistic Regression (Best AUC) | 89.82% | 96.32% | 92.05% | 92.06% |
| Neural Network | 89.17% | 92.90% | 93.67% | 91.98% |
| Random Forest (baseline) | 89.17% | 74.59% | 54.99% | 91.94% |
| Random Forest (tuned) | 89.46% | 77.45% | 51.82% | 92.7% |
| XGBoost (baseline) | 89.17% | 71.95% | 57.42% | 91.51% |
| Random Forest (tuned) | 89.94% | 75.39% | 58.88% | 92.94% |
| Naive Bayes (baseline) | ~81.43% | Moderate | Moderate | Good |
| Naive Bayes (optimized) | ~83.80% | Improved | Improved | Improved |

### Key Findings

1. **Best Performer**: The Logistic Regression with Scaled and Log-transformed data achieved the highest accuracy at 90.06%, making it the most reliable and interpretable model for this problem.

2. **Neural Network**: The deep learning approach was competitive, reaching 89.17% accuracy. It showed promise for capturing complex patterns in customer behavior, and potentially could overcome the robustness of Logistic-Regression with further hyper-parameter tuning.

3. **Naive Bayes**: Our pure-Python     implementation provided a lightweight baseline at approximately 81% accuracy. Through parameter tuning (adjusting smoothing and variance thresholds), we improved it to 82-84% accuracy. While this didn't match the best models, it showed how even simple probabilistic assumptions can capture meaningful patterns.
Tuning has been done by grid searching both auc and best accuracy. 
The Naive Bayes parameter tuning included ROC curve analysis showing that:
- Different smoothing parameters had varying effects on model calibration
- Variance thresholds influenced how the model handles outliers
- The optimized configuration achieved better balance between true positive rate and false positive rate
- This analysis helped us to pick the best parameter combination for the purpose. we observed that tuning the variance gives large improvement on the AUC of the model. 

![roc_parameter_impact](https://hackmd.io/_uploads/Sy9ztbbG-l.png)

- In terms of log transformation feature compared to non log-tranformed simple scaling features, the best auc improved by 3%. 

![log_transform_comparison](https://hackmd.io/_uploads/ByiuFWZMbe.png)
- the best auc acheived after log tranformation applied reached 87%, although the straight forward accuracy rate stays roughly the same(84.72% -> 84.8%).

![categorical_parameters](https://hackmd.io/_uploads/BJYy77WMZg.png)

![numerical_parameters2](https://hackmd.io/_uploads/Sy5hmQWfZe.png)


And then we have a internal probability distribution of Naive Bayes of important features. This directly relates to feature analysis later in the report. One of Naive Bays's advantage is its interpretibility of parameters. 


4. **Minimal Impact of Hyperparameter Tuning in XGBoost and Random Forest Models**: Contrary to expectations, hyperparameter tuning resulted in marginal improvements, let us consider the random forest and xgboost cases seperately;
- **Random Forest**: Tuning only resulted in a 0.29% improvement in accuracy (89.17% → 89.46%), however, there was a a 3.17% decrease in recall (54.99% → 51.82%). Which suggests the tuned model may have become slightly more biased toward the majority class. This is seen in the confusion matrix where there is an increase in the true negatives but a reduction in the true positives.
![rf_cm](https://hackmd.io/_uploads/By-cczbMWe.png)
![rf_cmt](https://hackmd.io/_uploads/Bylo5z-MZl.png)
- **XGBoost**: The tuned model performed slightly better with a 0.77% accuracy improvement (89.17% → 89.94%) and we see improvements across the board even in recall where the tuned random forest had a reduction. Unlike the random forest after tuning there is an increase in both true negative and true positive prediction which is can be seen in the confusion matrix.
![xg_cf](https://hackmd.io/_uploads/B1nj9MWGZg.png)
![xg_cft](https://hackmd.io/_uploads/Byno5zbMbl.png)


### ROC Curve Analysis

Now comparing all the models, here's what we have as a result.

![unified_roc_all_models](https://hackmd.io/_uploads/BJ_eL7WMWg.png)


We see that XGboost and Randomforrest and neural network, the three models that perform great in accuracy also leads in AUC performance. 




**Comparative Analysis of Feature Attribution in Purchase Intention Classifiers**


![unified_top5_feature_importance](https://hackmd.io/_uploads/H1UGL7bMZg.png)




**1. Universal Feature Dominance**

**Consensus Finding: The Predominance of `PageValues`**
Across all evaluated architectures—Linear (Logistic Regression), Probabilistic (Naive Bayes), and Tree Ensemble (Random Forest, XGBoost)—the feature **`PageValues`** exhibits a normalized importance score approaching unity (1.0).

* **Interpretation:** This consensus indicates that `PageValues` serves as the primary discriminant for the target variable $Y$ (Revenue). Regardless of the decision boundary topology (hyperplane, probability density, or orthogonal splits), the average transactional value of visited pages contains the highest mutual information with the purchasing event.

---

**2. Model-Specific Attribution Analysis**

**A. Logistic Regression: Linear Separability of Exit Behavior and Seasonality**

**Observation:** This model assigns uniquely high importance to **`ExitRates`** and specific temporal features (**`Month_Nov`**, **`Month_May`**, **`Month_Mar`**, **`Month_Feb`**).

* **Mathematical Reasoning:** Logistic Regression models the log-odds of the probability of purchase as a linear combination of input features:
    $$\ln\left(\frac{P(Y=1)}{P(Y=0)}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$
    Feature importance in this context is directly proportional to the magnitude of the coefficient $|\beta_j|$.
    * **Exit Rates:** The model detects a strong linear inverse relationship between `ExitRates` and conversion. Unlike tree methods, which can isolate high-value segments, logistic regression requires a global slope ($\beta_{exit}$) to punish probabilities for sessions with high exit rates across the entire dataset.
    * **Seasonality:** The high coefficients for specific months suggest that the baseline conversion probability shifts significantly during these periods. The model accommodates this by shifting the intercept ($\beta_0$) via the dummy variables for these months.

**B. Naive Bayes: Independence Assumption and Feature Amplification**

**Observation:** Naive Bayes disproportionately weights user activity metrics, specifically **`ProductRelated_Duration`**, **`ProductRelated`**, and **`Administrative`**, assigning them significantly higher importance than tree-based ensembles.

* **Mathematical Reasoning:** The Naive Bayes classifier operates on the assumption of conditional independence among features given the class $Y$:
    $$P(Y|X_1, \dots, X_n) \propto P(Y) \prod_{i=1}^{n} P(X_i | Y)$$
    In the log-domain, this becomes an additive sum of log-likelihoods:
    $$\log P(Y|X) \propto \log P(Y) + \sum_{i=1}^{n} \log P(X_i | Y)$$
    * **Redundancy Amplification:** Activity metrics such as *duration* on a page and the *count* of those pages are naturally highly correlated (multicollinear). While other models penalize or ignore redundant information, Naive Bayes treats each feature as an independent piece of evidence. Consequently, the signals from `ProductRelated` (count) and `ProductRelated_Duration` (time) are summed, effectively "double-counting" the user's engagement level and inflating the perceived importance of these activity metrics.

**C. XGBoost: Residual Learning and Seasonal Discovery**

**Observation:** XGBoost identifies seasonality (**`Month_Nov`**, **`Month_May`**, **`Month_Mar`**) as a critical predictor, utilizing these features as major splitting criteria significantly more than Random Forest.

* **Mathematical Reasoning:** XGBoost is a boosting algorithm that constructs an ensemble sequentially, where each new tree $f_t(x)$ attempts to correct the errors (residuals) of the prior ensemble $F_{t-1}(x)$:
    $$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$$
    * **Gradient Descent in Function Space:** While `PageValues` likely explains the variance for the majority of "obvious" shoppers, there remains a subset of variance (residuals) that cannot be explained by behavior alone. Seasonality often captures non-linear interaction effects (e.g., "users browsing in November are inherently more motivated"). By iteratively focusing on misclassified instances, XGBoost extracts these temporal nuances to reduce the loss function, effectively "discovering" seasonality as a necessary corrector for specific sub-populations.

**D. Random Forest: Variance Reduction and Signal Masking**

**Observation:** Random Forest presents a "middle-ground" profile. It acknowledges `ExitRates` and `Duration` but suppresses their importance relative to `PageValues`, showing shorter bars for secondary features compared to Logistic Regression or Naive Bayes.

* **Mathematical Reasoning:** Random Forest utilizes bagging (Bootstrap Aggregating) and random feature selection. The importance is typically calculated as the mean decrease in impurity (Gini or Entropy):
    $$\text{Importance}(X_j) = \frac{1}{N_{trees}} \sum_{t=1}^{N_{trees}} \Delta I(X_j, t)$$
    * **Greedy Splitting & Masking:** Decision trees employ greedy splitting. If `PageValues` is the strongest predictor, it will frequently be chosen as the primary split node. Once the data is partitioned by `PageValues`, the remaining *conditional* information gain provided by correlated features like `ExitRates` or `ProductRelated_Duration` is marginal. Unlike Naive Bayes (which adds signals together), the decision tree structure effectively "masks" secondary features that offer redundant information, leading to their diminished importance scores.

---

**3. Structural Disagreements**

**The Multicollinearity Divergence (ExitRates vs. PageValues)**
A fundamental disagreement exists regarding **`ExitRates`**.
* **Linear Perspective:** Logistic Regression views `ExitRates` as a top-tier feature because it requires a distinct vector component to push the decision boundary away from high-exit sessions.
* **Tree-Based Perspective:** XGBoost and Random Forest deem `ExitRates` less important. Mathematically, `PageValues` and `ExitRates` carry high mutual information (high-value pages retain users). In a tree structure, once a split is made on `PageValues`, the information gain from a subsequent split on `ExitRates` is negligible.

**The Temporal-Behavioral Split**
* **Temporal Focus:** Logistic Regression and XGBoost rely on the *context* of the session (Seasonality) to adjust their baselines (intercepts or residuals).
* **Behavioral Focus:** Random Forest and Naive Bayes rely almost exclusively on the *content* of the session (Duration/PageViews), largely ignoring the temporal context. This suggests that for these models, the user's immediate actions are sufficient to minimize entropy, rendering the "time of year" statistically superfluous.
## Contributions

- **Aaron Zhang**: Led data pre-processing, implemented neural network and logistic regression, planned and wrote report
- **Andrew Arochukwu**:Implemented XGBoost and random forest, and the tuned xgboost and random forest models, additionally inspected the best model by comparing the confusion matrix for the different models.
- **Qiguang (William) Zhu**: Implemented Naive Bayes algorithm, tuned parameters and inspected best model by comparing Area under ROC curve. 

## Supplement

