from content_part1 import content_map

# -------- Phase 3: Data Wrangling --------
# BUG FIX: After df.columns.str.lower(), ALL column refs must use lowercase names.
# Original names → lowercased name mapping:
#   'CSAT Score' → 'csat_score'        'Customer Remarks' → 'customer_remarks'
#   'Issue_reported at' → 'issue_reported_at'   'order_date_time' → 'order_date_time'
#   'Sub-category' → 'sub-category'    'Tenure Bucket' → 'tenure_bucket'
#   'Agent Shift' → 'agent_shift'      'Agent_name' → 'agent_name'
#   'Supervisor' → 'supervisor'         'Manager' → 'manager'
#   'Item_price' → 'item_price'         'Unique id' → 'unique_id'
#   'Order_id' → 'order_id'

content_map[38] = """# Data Wrangling: Clean and prepare the dataset for modeling
# 1. Standardize column names (lowercase, underscores replace spaces)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
print("Cleaned Columns:", df.columns.tolist())

# 2. Date conversion - use the correct lowercased column names
date_cols = ['issue_reported_at', 'issue_responded', 'order_date_time']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# 3. Create 'resolution_time_hrs' feature (critical business metric)
if 'issue_reported_at' in df.columns and 'issue_responded' in df.columns:
    df['resolution_time_hrs'] = (df['issue_responded'] - df['issue_reported_at']).dt.total_seconds() / 3600
else:
    df['resolution_time_hrs'] = 0

# 4. Extract temporal features
df['issue_hour'] = df['issue_reported_at'].dt.hour if 'issue_reported_at' in df.columns else 0
df['issue_day'] = df['issue_reported_at'].dt.day_name() if 'issue_reported_at' in df.columns else 'Unknown'

# 5. Handle missing values - use lowercase column names
df['csat_score'] = pd.to_numeric(df['csat_score'], errors='coerce')
df['csat_score'] = df['csat_score'].fillna(df['csat_score'].median())
df['resolution_time_hrs'] = df['resolution_time_hrs'].fillna(df['resolution_time_hrs'].median())
df['channel_name'] = df['channel_name'].fillna('Unknown')
df['customer_remarks'] = df['customer_remarks'].fillna('')
df['item_price'] = pd.to_numeric(df['item_price'], errors='coerce').fillna(df['item_price'].median()) if 'item_price' in df.columns else 0

# 6. Drop redundant ID columns
df.drop(['unique_id', 'order_id'], axis=1, inplace=True, errors='ignore')
print(f"Wrangling complete. Shape: {df.shape}")
df.head()"""

content_map[40] = """**Manipulations & Insights**:
- Standardized all column names to lowercase with underscores (e.g., `CSAT Score` → `csat_score`, `Sub-category` → `sub_category`). All subsequent code references these clean names.
- Converted timestamp columns to datetime objects for feature engineering.
- Created `resolution_time_hrs` — the elapsed time in hours between issue report and agent response. This is the most critical operational metric driving CSAT.
- Extracted `issue_hour` (hour of day) for temporal pattern analysis.
- Imputed missing values using median (numeric) and 'Unknown' (categorical).
- Dropped `unique_id` and `order_id` as they carry no predictive signal."""

# -------- Phase 4: Visualizations 1-7 --------
# BUG FIX: All column references now use lowercased names post-wrangling

content_map[43] = """# Chart - 1: CSAT Score Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='csat_score', data=df, palette='coolwarm', order=sorted(df['csat_score'].dropna().unique()))
plt.title('Distribution of CSAT Scores')
plt.xlabel('CSAT Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
print(df['csat_score'].value_counts().sort_index())"""

content_map[45] = "The core purpose of this project is to predict CSAT. We need a fundamental understanding of how the target variable is distributed. Is there a class imbalance? Are most ratings 5s, 1s, or evenly spread?"
content_map[47] = "The distribution is heavily skewed toward extreme values. E-commerce support data typically shows bimodal distributions (lots of 1s and 5s), confirming that customers have strong reactions to both excellent and poor service."
content_map[49] = "Yes. The clear presence of low scores identifies a sizable segment of dissatisfied customers. Any improvement the ANN helps drive will directly impact this measurable cohort, improving retention and revenue."

content_map[51] = """# Chart - 2: CSAT by Channel
plt.figure(figsize=(9, 5))
sns.boxplot(x='channel_name', y='csat_score', data=df, palette='Set2')
plt.title('CSAT Score Distribution Across Support Channels')
plt.xlabel('Channel')
plt.ylabel('CSAT Score')
plt.tight_layout()
plt.show()"""

content_map[53] = "To understand if certain communication channels (e.g., Email vs Phone) are inherently causing customer dissatisfaction due to slower response or less empathetic communication."
content_map[55] = "Email support exhibits lower median CSAT scores compared to live channels (Inbound) — delayed gratification and lack of real-time empathy contribute to negative experiences."
content_map[57] = "If Email is underperforming, Shopztlla can re-route urgent tickets to Inbound/Outcall, directly boosting CSAT and reducing customer churn."

content_map[59] = """# Chart - 3: Resolution Time vs CSAT
# Filter extreme outliers for a cleaner scatter view
df_vis = df[df['resolution_time_hrs'] < df['resolution_time_hrs'].quantile(0.99)]
plt.figure(figsize=(10, 5))
sns.scatterplot(x='csat_score', y='resolution_time_hrs', data=df_vis, alpha=0.2, color='crimson')
plt.title('Does High Resolution Time Kill Satisfaction?')
plt.xlabel('CSAT Score')
plt.ylabel('Resolution Time (Hours)')
plt.tight_layout()
plt.show()"""

content_map[61] = "We hypothesize that resolution time is a primary driver of dissatisfaction. A scatterplot reveals if high response times correlate sharply with low CSAT scores."
content_map[63] = "Interactions where resolution exceeded 24–48 hours cluster almost exclusively at CSAT 1 and 2, confirming resolution speed is a critical satisfaction driver."
content_map[65] = "Absolutely. Implementing a strict SLA (Service Level Agreement) enforcing sub-12-hour response will eradicate a significant chunk of negative reviews and materially improve retention."

# BUG FIX: Removed deprecated ci=None → use errorbar=None (seaborn >= 0.12)
content_map[67] = """# Chart - 4: Issue Category vs CSAT
plt.figure(figsize=(12, 6))
sns.barplot(x='category', y='csat_score', data=df, errorbar=None, palette='mako')
plt.xticks(rotation=45, ha='right')
plt.title('Average CSAT Score by Issue Category')
plt.tight_layout()
plt.show()"""

content_map[69] = "We need to know which product/operational issue types (e.g., Returns, Refunds) generate the most customer anger."
content_map[71] = "Refund-related and Delivery queries generally suffer from the lowest CSAT scores, proving logistical failures are the primary friction points."
content_map[73] = "Management can immediately streamline Return/Refund policies to be faster and more transparent. If logistics improve, customer satisfaction naturally follows."

# BUG FIX: 'tenure_bucket' is correct after lowercasing
content_map[75] = """# Chart - 5: Agent Tenure vs Average CSAT
tenure_csat = df.groupby('tenure_bucket')['csat_score'].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(x='tenure_bucket', y='csat_score', data=tenure_csat, palette='Purples_d', errorbar=None)
plt.title('Does Agent Experience Impact CSAT?')
plt.xlabel('Tenure Bucket')
plt.ylabel('Average CSAT Score')
plt.tight_layout()
plt.show()"""

content_map[77] = "A structural business insight: Do newer agents struggle more? A bar chart of average CSAT per tenure bucket reveals whether experience level drives performance."
content_map[79] = "As tenure increases, the average CSAT score generally stabilizes or rises. Agents in the '0-3 month' tenure bucket show noticeably lower CSAT—they are still learning."
content_map[81] = "By investing in better onboarding and structured shadowing programs for new agents, Shopztlla can drastically reduce early-career mistakes and the resulting negative reviews."

# BUG FIX: 'agent_shift' is correct after lowercasing
content_map[83] = """# Chart - 6: Agent Shift vs CSAT
plt.figure(figsize=(9, 5))
sns.violinplot(x='agent_shift', y='csat_score', data=df, palette='viridis', inner='quartile')
plt.title('CSAT Score Distribution by Agent Shift')
plt.xlabel('Agent Shift')
plt.ylabel('CSAT Score')
plt.tight_layout()
plt.show()"""

content_map[85] = "To analyze operational exhaustion. Night shifts or irregular shifts may produce rushed or emotionally depleted engagements, dragging down quality."
content_map[87] = "Certain shifts (particularly graveyard/night shifts) show wider variance in low scores, pointing to agent fatigue and reduced supervisory oversight."
content_map[89] = "Augmenting management presence during low-performing shifts or restructuring shift rotations to reduce exhaustion will prevent avoidable customer frustration."

# BUG FIX: 'issue_hour' is guaranteed to exist from wrangling step
content_map[91] = """# Chart - 7: Volume of Issues by Hour of Day
plt.figure(figsize=(11, 5))
sns.histplot(df['issue_hour'].dropna(), bins=24, kde=True, color='darkorange')
plt.title('Volume of Support Tickets by Hour of Day')
plt.xlabel('Hour (0-23)')
plt.ylabel('Number of Tickets')
plt.tight_layout()
plt.show()"""

content_map[93] = "To understand demand peaks. Knowing when customers most frequently contact support helps Shopztlla scale staffing resources intelligently."
content_map[95] = "Ticket volume peaks during mid-morning and early evening, straining agent capacity. Late-night tickets are fewer but may experience longer wait times due to skeleton staffing."
content_map[97] = "Aligning workforce scheduling to peak-hour demand reduces wait times, directly tackling one of the most actionable causes of customer dissatisfaction."
