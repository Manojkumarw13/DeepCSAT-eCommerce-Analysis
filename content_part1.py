import pandas as pd
import numpy as np

# A dictionary mapping Cell Index to the New Source Content (String)
content_map = {}

# -------- Phase 1: Project Metadata & Summary Initialization --------
content_map[0] = "# **Project Name**    -  DeepCSAT: Ecommerce Customer Satisfaction Predictor (Shopztlla)"
content_map[1] = "##### **Project Type**    - Deep Learning / Artificial Neural Network (Classification/Prediction)\n##### **Contribution**    - Individual"

# Project Summary (600+ words)
content_map[3] = """**Project Summary**

The objective of this machine learning capstone project is to develop a deep learning solution—specifically, an Artificial Neural Network (ANN)—capable of predicting Customer Satisfaction (CSAT) scores for **Shopztlla**, a burgeoning regional e-commerce platform. Shopztlla has accumulated a comprehensive dataset capturing customer interactions with support agents over a one-month period. Management has identified a critical need: while they gather feedback, they cannot proactively predict which interactions will result in negative satisfaction, nor do they fully understand the nuanced reasons driving operational friction. Their goal is to transition from a reactive customer service model to a proactive, real-time CSAT monitoring system that significantly enhances service quality and fosters unwavering customer loyalty.

To achieve this, the overarching strategy encompasses an end-to-end data pipeline, from deep exploratory data analysis (EDA) to complex modeling. Initially, the unstructured and structured aspects of the customer support data will be cleaned, transformed, and rigorously analyzed. Through fifteen distinct, highly targeted visualizations, we will uncover exactly "why the customers are not satisfied." We will explore critical dimensions such as the correlation between resolution times and satisfaction drops, the specific performance of various communication channels (Inbound, Outcall, Email), product category friction, and agent management metrics. By extracting actionable insights from these visualizations, Shopztlla can immediately begin rectifying operational bottlenecks—even before the predictive model is fully integrated.

Subsequently, statistical hypothesis testing will validate our observational assumptions in real-time contexts. We will scientifically determine whether resolution times, communication channels, and agent performance inherently dictate the end-user's CSAT score, cementing our EDA findings with statistically significant P-values.

The core technical endeavor of this project is the construction, training, and deployment of a Deep Learning ANN model. Given the multifaceted nature of human interaction data—which includes categorical features (like agent shifts, channels), numerical features (like handling times), and text-based remarks—an ANN offers the non-linear modeling capabilities required to accurately map these complex inputs to a final CSAT score. We will engineer features specifically tailored for neural network consumption: scaling continuous variables, applying one-hot encoding for categorical variables, and utilizing basic natural language processing (NLP) to vectorize customer remarks.

The ANN architectures evaluated will progressively increase in sophistication. We will establish a baseline neural network, then iterate via hyperparameter tuning—optimizing learning rates, batch sizes, and dense layer configurations using advanced optimizers like Adam. Furthermore, we will implement train-validation-test splits to guarantee the model's generalization to unseen data and employ techniques to handle any class imbalances inherent in satisfaction scores (e.g., predominantly positive vs. sparse negative reviews).

Ultimately, this predictive engine will empower Shopztlla's management. When a customer interaction concludes, the ANN will instantly process the interaction's metrics and predict the CSAT score. If the predicted score is perilously low, the system can autonomously flag the ticket for managerial review or trigger a follow-up appeasement protocol. By acting as a real-time monitoring tool, this DeepCSAT solution will not only resolve individual grievances promptly but also elevate the overarching operational standard, turning potential detractors into loyal advocates."""

content_map[5] = "[GitHub Repository](https://github.com/manojkumarw13/DeepCSAT-eCommerce-Analysis)"

content_map[7] = """**Problem Statement**

Shopztlla, an e-commerce platform, has collected a month's worth of customer support interaction data. Despite having access to metrics such as agent handling times, communication channels, product categories, and customer remarks, the company struggles to proactively gauge and manage customer satisfaction. Currently, they only know a customer is dissatisfied *after* receiving a low CSAT score, at which point the damage to customer loyalty is already done.

The primary business problem is to rapidly uncover the root causes of customer dissatisfaction and map these operational variables to customer sentiment. Consequently, the technical challenge is to design and train a robust Artificial Neural Network (ANN) that can accurately predict the CSAT score based on the parameters of a customer's interaction. This predictive model will serve as a powerful, real-time monitoring tool, allowing Shopztlla to detect negative experiences as they happen, intervene promptly, improve service quality, and foster long-term customer loyalty."""

# -------- Phase 2: Data Loading & Initial Exploration --------
# BUG FIX: Added ConfusionMatrixDisplay and joblib imports
content_map[13] = """# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import joblib
import scipy.stats as stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
print("All libraries imported successfully!")"""

# BUG FIX: Added imbalanced-learn install guard to avoid runtime failure
content_map[15] = """# Install imbalanced-learn if not present, then load dataset
import subprocess, sys
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'imbalanced-learn', '-q'])
    from imblearn.over_sampling import SMOTE

# Load Dataset
df = pd.read_csv('eCommerce_Customer_support_data.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")"""

content_map[17] = "# Dataset First Look\ndf.head()"
content_map[19] = "# Dataset Rows & Columns count\nprint(f\"Rows: {df.shape[0]}, Columns: {df.shape[1]}\")"
content_map[21] = "# Dataset Info\ndf.info()"
content_map[23] = "# Dataset Duplicate Value Count\nprint(f\"Duplicates: {df.duplicated().sum()}\")"
content_map[25] = "# Missing Values/Null Values Count\nprint(df.isnull().sum())"

content_map[26] = """# Visualizing the missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap', fontsize=14)
plt.tight_layout()
plt.show()"""

content_map[28] = """**Knowledge regarding the dataset:**
The dataset encompasses customer satisfaction (CSAT) scores over a one-month period on the e-commerce platform Shopztlla. It contains rows and 20 columns. Key columns include interaction details (`unique id`, `channel_name`, `category`, `Sub-category`), timestamps (`Issue_reported at`, `issue_responded`), operational metrics (`Order_id`, `Agent_name`, `Supervisor`, `Manager`, `Tenure Bucket`, `Agent Shift`, `CSAT Score`), and verbatim text (`Customer Remarks`). There are null values, particularly in `Customer Remarks`, `Order_id`, and `Item_price`, which require careful imputation. Since this is an ANN project, we rely heavily on creating structured features out of the datetime columns and encoding the high-cardinality categorical variables."""

content_map[30] = "# Dataset Columns\nprint(list(df.columns))"
content_map[31] = "# Dataset Describe\ndf.describe(include='all')"
content_map[33] = """**Variables Description**:
- **Unique id**: Identifier for each support ticket.
- **channel_name**: The communication medium (Inbound, Outcall, Email).
- **category & Sub-category**: The type of issue (Returns, Orders, Refunds, etc.).
- **Customer Remarks**: Verbatim textual feedback from the customer.
- **Order_id & order_date_time**: Purchase order details.
- **Issue_reported at & issue_responded**: Timestamps for calculating resolution time.
- **Agent_name, Supervisor, Manager**: Support staff hierarchy.
- **Tenure Bucket**: Experience level of the agent.
- **Agent Shift**: Time-of-day when agent is working.
- **connected_handling_time**: Duration of the support call.
- **Item_price**: Price of the item in the order.
- **CSAT Score**: The target variable. Customer satisfaction rating (1=worst, 5=best)."""

content_map[35] = """# Check Unique Values for each variable.
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")"""

print("Phase 1-2 written to map")
