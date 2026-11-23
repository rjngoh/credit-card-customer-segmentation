# Credit Card Customer Segmentation & Promotional Strategy Optimisation Using Machine Learning

**An end-to-end descriptive data mining project using Python (Google Colab), K-Means Clustering, and Apriori Association Rule Mining.**

*University Project | Singapore University of Social Sciences | ANL303 Fundamentals of Data Mining | 2026*

## Project Overview

This project explores how a credit card provider can use descriptive data mining to identify suitable customer segments for targeted promotional campaigns.

The business objective was to increase credit card usage among zero-balance cardholders while reducing unnecessary marketing expenditure associated with targeting the entire customer base.

Using the **CRISP-DM framework**, two modelling techniques were selected to address the business problem:

- **K-Means Clustering:** Segment customers based on age and income to identify groups suitable for targeted promotions.
- **Apriori Association Rule Mining:** Identify relationships between past promotional campaign responses to help the company develop promotional strategies.

## Tools & Technologies

- **Programming:** Python (Google Colab)
- **Data Manipulation:** Pandas
- **Data Visualisation:** Matplotlib, Seaborn
- **Cluster Analysis:** K-Means Algorithm
- **Association Analysis:** Apriori Algorithm
- **Model Evaluation:** Average Silhouette Coefficient, Minimum Support Threshold, Minimum Confidence Threshold
- **Data Preprocessing:** Data Cleaning, Min-Max Normalisation, Boolean Encoding
- **Methodology:** CRISP-DM

## Methodology
The **Cross-Industry Standard Process for Data Mining (CRISP-DM)** framework involves six phases in a typical data mining project:
### Phase 1: Business Understanding

This phase involves identifying the business problem, which is that a credit card provider is losing out on potential revenue due to inactive customers. The business objective is to encourage spending via targeted promotional offers to increase the revenue. The data mining objectives are to first segment customers to identify target groups for promotions, followed by bundling the appropriate promotional activities together for the campaign, to increase customer engagement of the targeted segments, giving them the opportunity to make purchases using their credit cards thus, generating more revenue for the company.

### Phase 2: Data Understanding

This phase involves collecting data from the application process. Data exploration using descriptive statistics and visualisation will be conducted to get insights and detect data quality issues. For example, boxplots are used to identify outliers.
Exploratory data analysis identified several data quality issues:

- Invalid age records
- Missing income values
- Inconsistent categorical values
- Duplicate observations
- Statistical outliers in age and income

### Phase 3: Data Preparation

This phase involves selecting the appropriate data, and preparing them for Clustering, and Association Rule Mining. Additionally, data cleaning is performed to handle missing values, inconsistent categorical values, duplicate observations, and outliers.

K-Means Clustering: Since K-means relies on distance calculations, continuous variables are selected. As K-means is sensitive to scale, data normalisation will be performed.

Apriori Association Rule Mining: The Apriori algorithm on python requires input variables to be converted into Boolean – True or False. Once converted, the data will be suitable for use with the Apriori algorithm.

### Phase 4: Modelling

This phase involves building the K-Means model, and generating strong association rules to derive meaningful insights.

**Customer Segmentation (K-Means Clustering)**

Different values of K (number of clusters) will be tested, and the final model will be selected based on metrics such as the average silhouette coefficient, as well as the principle of parsimony.
Multiple cluster configurations (K=2 to K=6) were evaluated using average silhouette coefficients.

**Silhouette Analysis results:**

<img width="471" height="97" alt="Silhouette Analysis Results" src="https://github.com/user-attachments/assets/af17a689-8351-4b99-9396-07b983ab63fe" />

**The chosen 'best model' (K=2 & outliers retained):**

<img width="589" height="455" alt="Cluster Solution" src="https://github.com/user-attachments/assets/37633d99-1f14-43bb-b0d4-7a7a74e0ba93" />

Key findings:
- K=2 achieved the highest average silhouette coefficient of approximately 0.4545.
- The resulting clusters were categorised as ‘Younger, Lower Income’ and ‘Older, Higher Income’.
- Outlier sensitivity testing revealed negligible differences in silhouette performance between models retaining and excluding outliers.
- The model retaining valid outliers was selected to preserve potentially meaningful observations.

**Promotional Preference Analysis (Apriori Association Rule Mining)**

The minimum support and minimum confidence thresholds were evaluated to select appropriate values, where meaningful business insights can be derived from the generated strong rules (there should not be too many rules).

The following Bar chart for customer engagement frequency of promotional activities in past campaigns helps us decide on a minimum support threshold for the Apriori algorithm:

<img width="571" height="564" alt="Bar chart for customer engagement frequency of promotional activities in past campaigns" src="https://github.com/user-attachments/assets/6e3b8b5f-6c8b-4ee4-8016-7a50125aa58e" />

As promotional campaigns take a substantial investment from businesses, it is important to decide on effective marketing strategies that are likely to yield positive returns. If the company decides to invest in promotional campaigns to include promotional activities such as Travel Packages, Beauty and Wellness, and Entertainment, the return on investment (ROI) from the campaign may not be satisfactory due to potential low turnout rates for including these activities. Therefore, a **minimum Support value of 0.1** (10%) is selected to filter out infrequent promotional activities.

**Selected parameters:**

- Minimum Support: **10%**
- Minimum Confidence: **80%**
- Association Rules Generated: **3**

**The selected thresholds produced a manageable set of association rules for business interpretation:**

<img width="1688" height="134" alt="Strong Association Rules" src="https://github.com/user-attachments/assets/e8d7a157-7bfc-4058-ad56-f7a6f299a514" />

One notable association identified was:

**{Insurance Plans, Electronics} -> {Home Appliances} (rule support: 11%, confidence: 83.6%)**

### Phase 5: Evaluation

This phase involves assessing whether the Clustering solution, and the generated strong rules are meaningful and useful from a business perspective.

**K-Means Clustering:** The proportion of zero-balance cardholders will be computed for each cluster. Clusters with a larger proportion of zero-balance cardholders will be identified as the target(s) for promotions.

**Proportional Analysis Results in %:**

<img width="268" height="131" alt="Proportional Analysis Results" src="https://github.com/user-attachments/assets/85ab8fbb-5add-466f-8327-cbfe639c1554" />

The analysis reveals that 75.2% of zero-balance cardholders are from the younger/lower-income cluster which is significantly larger than the older/higher-income cluster where the remaining 24.8% of zero-balance cardholders are from. Therefore, the **'Younger, Lower Income'** cluster is the proposed target group for promotional campaigns.

**Apriori Association Rule Mining:** The generated strong rules will be examined to ensure that they are actionable, and can help the company to develop effective promotional strategies. – The analysis produced a notable association rule – **{Insurance Plans, Electronics} -> {Home Appliances}** – which indicates that approximately 83.6% of the observations involving both insurance plan and electronics promotional activities also involved home appliances promotional activity. The association provides a basis for exploring complementary promotional offerings.

### Phase 6: Deployment

This phase involves integrating the Clustering solution, and the generated strong rules into business processes.

K-Means Clustering: Future customers can be assigned to clusters, and those belonging to targeted segments can be flagged for personalized marketing efforts.

Apriori Association Rule Mining: Based on the strong rules generated, the company can develop effective promotional strategies, such as bundling the appropriate promotional activities together for the campaign, to increase customer engagement of the targeted segments.

Model performance for both should be monitored over time to maintain their benefits.

### Project Background

This project was completed as part of the ANL303 Fundamentals of Data Mining group-based assignment at the Singapore University of Social Sciences.

The analysis demonstrates the application of descriptive data mining techniques to a business problem.

### Repository Contents

The repository contains the Python analysis notebook, project documentation, and supporting visualisations.
