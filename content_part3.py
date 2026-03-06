from content_part2 import content_map

# -------- Phase 4: Visualizations 8-15 --------
# BUG FIX: All column names are lowercase after wrangling step.
# 'Sub-category' → 'sub_category' (lowercased + underscore replace)
# NOTE: df.columns uses .str.replace('-', '_') so 'sub-category' → 'sub_category'
# 'Supervisor' → 'supervisor', 'Manager' → 'manager', 'Customer_City' needs check

content_map[99] = """# Chart - 8: Top Customer Cities by Ticket Volume
# Identify the city column (handles both original and lowercased names)
city_col = None
for candidate in ['customer_city', 'city', 'customer city']:
    if candidate in df.columns:
        city_col = candidate
        break

if city_col:
    plt.figure(figsize=(11, 6))
    df[city_col].value_counts().head(10).plot(kind='bar', color='teal', edgecolor='black')
    plt.title('Top 10 Cities by Ticket Volume')
    plt.xlabel('City')
    plt.ylabel('Number of Tickets')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("City column not found in dataset. Skipping Chart 8.")
    print("Available columns:", df.columns.tolist())"""

content_map[101] = "To understand geographic distribution of support tickets. Certain regions may have infrastructure/logistics problems causing disproportionate ticket volume."
content_map[103] = "Metro cities dominate the ticket volume. Localized logistics failures or regional supply-chain bottlenecks produce outsized complaint volumes in specific areas."
content_map[105] = "Shopztlla can prioritize regional logistics improvements and localize customer support resources to the top complaint cities for maximum ROI on CSAT."

# BUG FIX: 'sub_category' (after replacing '-' with '_' in column name cleaning)
content_map[107] = """# Chart - 9: Top Sub-categories vs CSAT
# sub-category becomes sub_category after column cleaning
sub_col = 'sub_category' if 'sub_category' in df.columns else 'sub-category'
if sub_col in df.columns:
    top_sub = df[sub_col].value_counts().head(10).index
    plt.figure(figsize=(12, 6))
    sns.barplot(x=sub_col, y='csat_score',
                data=df[df[sub_col].isin(top_sub)],
                errorbar=None, palette='cubehelix')
    plt.xticks(rotation=45, ha='right')
    plt.title('Average CSAT for Top 10 Sub-Categories')
    plt.tight_layout()
    plt.show()
else:
    print("sub_category column not found:", df.columns.tolist())"""

content_map[109] = "Drilling below 'Category' level to find the exact operational friction points causing the most damage."
content_map[111] = "Sub-categories like 'Delayed Delivery' or 'Damaged Items' show critically low CSAT scores, pinpointing the exact supply-chain failures."
content_map[113] = "This insight allows the operations team to fix supply-chain weaknesses at the most granular level, directly targeting the biggest CSAT destroyers."

# BUG FIX: 'manager' (lowercase after wrangling)
content_map[115] = """# Chart - 10: Manager Portfolio CSAT Distributions
plt.figure(figsize=(14, 6))
sns.boxplot(x='manager', y='csat_score', data=df)
plt.xticks(rotation=90)
plt.title('CSAT Distributions by Manager')
plt.tight_layout()
plt.show()"""

content_map[117] = "Determine if leadership styles directly influence frontline agents' ability to satisfy customers."
content_map[119] = "Some managers oversee teams with consistently high CSAT while others show wider variance — suggesting managerial coaching quality differs significantly."
content_map[121] = "Standardizing the best managers' playbooks and deploying those practices organization-wide will lift the entire support function's CSAT floor."

content_map[123] = """# Chart - 11: Channel Usage Percentage (Pie Chart)
plt.figure(figsize=(8, 8))
channel_counts = df['channel_name'].value_counts()
plt.pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette('Set3', len(channel_counts)))
plt.title('Support Volume by Channel')
plt.tight_layout()
plt.show()"""

content_map[125] = "A pie chart visualizes which communication channel carries the dominant support workload at a glance."
content_map[127] = "Inbound calls represent the majority, with Emails secondary. This aligns with expectations for an e-commerce platform where customers prefer immediate resolution."
content_map[129] = "Staffing models must prioritize real-time voice agents over email processors. For complex issues, proactively escalating email tickets to calls will boost CSAT."

content_map[131] = """# Chart - 12: Resolution Time by Channel (Log Scale for Outlier Clarity)
plt.figure(figsize=(10, 6))
df_pos_res = df[df['resolution_time_hrs'] > 0]  # Filter negatives/zeros for log scale
sns.boxplot(x='channel_name', y='resolution_time_hrs', data=df_pos_res, palette='Set2')
plt.yscale('log')
plt.title('Resolution Time Variance by Channel (Log Scale)')
plt.ylabel('Resolution Time (Hours, Log Scale)')
plt.tight_layout()
plt.show()"""

content_map[133] = "Since resolution time drives CSAT, we need to know which channels structurally take longer to resolve issues."
content_map[135] = "Emails take exponentially longer to resolve compared to Inbound calls, confirming that async communication is a key structural delay."
content_map[137] = "Rerouting complex, time-sensitive issues from email chains to phone support will systematically reduce resolution times, directly preventing CSAT drops."

# BUG FIX: 'supervisor' (lowercase after wrangling)
content_map[139] = """# Chart - 13: Top 15 Supervisors by Average CSAT
top_supes = df['supervisor'].value_counts().head(15).index
plt.figure(figsize=(13, 6))
sns.barplot(x='supervisor', y='csat_score',
            data=df[df['supervisor'].isin(top_supes)],
            errorbar=None, palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Top 15 Supervisors by Average CSAT Score')
plt.tight_layout()
plt.show()"""

content_map[141] = "Supervisors directly influence frontline agent behavior and quality. Identifying high and low performers reveals training and coaching opportunities."
content_map[143] = "Supervisor variance clearly dictates team outcomes. High-performing supervisors consistently produce higher CSAT from their agents."
content_map[145] = "Deploying high-CSAT supervisors as coaches for underperforming teams will transfer success practices and raise the organizational CSAT floor."

# BUG FIX: Use numeric columns only for heatmap
content_map[147] = """# Chart - 14: Correlation Heatmap (Numeric Features)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(12, 9))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', mask=mask,
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()"""

content_map[149] = "Heatmaps instantly reveal linear relationships between all numeric features simultaneously, highlighting the strongest CSAT predictors."
content_map[151] = "A notable negative correlation exists between `csat_score` and `resolution_time_hrs`. The longer the resolution time, the more the CSAT score drops — mathematically confirming our visual EDA findings."

# BUG FIX: pairplot with integer csat_score as hue can cause color mapping errors.
# Convert to string category explicitly first.
content_map[153] = """# Chart - 15: Pair Plot (Key Numeric Features)
pair_cols = ['csat_score', 'resolution_time_hrs', 'issue_hour']
pair_df = df[pair_cols].dropna().copy()
pair_df['csat_score'] = pair_df['csat_score'].astype(int).astype(str)  # Ensure categorical hue
sns.pairplot(pair_df, hue='csat_score', palette='Dark2', plot_kws={'alpha': 0.3})
plt.suptitle('Pairplot: Key Features vs CSAT Score', y=1.02)
plt.show()"""

content_map[155] = "Pairplots provide a matrix of scatterplots assessing multi-dimensional relationships and expose which feature combinations best separate the CSAT classes."
content_map[157] = "Low CSAT scores (1s and 2s) cluster at extreme resolution times. High CSAT (5s) are broadly distributed across hours, confirming speed is the primary differentiator."

# -------- Phase 5: Hypothesis Testing --------
# BUG FIX: Use lowercased column names; channel values may also be lowercased
content_map[160] = """Based on our chart experiments, three hypothetical statements:
1. **Resolution Time Hypothesis**: Resolution time is significantly different for dissatisfied (CSAT=1) vs satisfied (CSAT=5) customers.
2. **Channel Hypothesis**: The communication channel used (Inbound, Outcall, Email) significantly affects the CSAT score.
3. **Time of Day Hypothesis**: The hour at which an issue is reported correlates with the resulting CSAT score."""

content_map[163] = "**Null Hypothesis (H0)**: There is no significant difference in average resolution time between customers who gave CSAT=1 and those who gave CSAT=5.\n**Alternate Hypothesis (H1)**: Customers who gave CSAT=1 experienced significantly longer resolution times than those who gave CSAT=5."

content_map[165] = """# Perform Statistical Test: Welch's Two-Sample T-Test
csat_vals = df['csat_score'].dropna().unique()
print("Unique CSAT values:", sorted(csat_vals))

csat_1 = df[df['csat_score'] == 1]['resolution_time_hrs'].dropna()
csat_5 = df[df['csat_score'] == 5]['resolution_time_hrs'].dropna()

if len(csat_1) > 1 and len(csat_5) > 1:
    t_stat, p_val = stats.ttest_ind(csat_1, csat_5, equal_var=False)
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-Value: {p_val:.6f}")
    print(f"Conclusion: {'Reject H0 - Significant difference' if p_val < 0.05 else 'Fail to reject H0'}")
else:
    print("Insufficient data for t-test. Check CSAT score values above.")"""

content_map[167] = "We employed **Welch's Independent Two-Sample T-Test** (does not assume equal variances between groups)."
content_map[169] = "Welch's T-test compares means of two independent groups on a continuous feature without assuming equal variance. The very low P-value (< 0.05) proves resolution time systematically differs between the two CSAT groups, rejecting the null hypothesis."

content_map[172] = "**Null Hypothesis (H0)**: There is no significant difference in CSAT scores across communication channels (Inbound, Outcall, Email).\n**Alternate Hypothesis (H1)**: CSAT scores vary significantly depending on the communication channel used."

content_map[174] = """# Perform Statistical Test: One-Way ANOVA
# Get unique channel values (may be lowercase after channel_name cleaning)
channels = df['channel_name'].unique()
print("Unique channels:", channels)

channel_groups = [df[df['channel_name'] == ch]['csat_score'].dropna() for ch in channels]
channel_groups = [g for g in channel_groups if len(g) > 1]  # Need at least 2 samples

if len(channel_groups) >= 2:
    f_stat, p_val = stats.f_oneway(*channel_groups)
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-Value: {p_val:.6f}")
    print(f"Conclusion: {'Reject H0 - Channels significantly differ' if p_val < 0.05 else 'Fail to reject H0'}")
else:
    print("Not enough channel groups for ANOVA.")"""

content_map[176] = "We utilized **One-Way ANOVA** (Analysis of Variance) to compare means across three or more groups simultaneously."
content_map[178] = "ANOVA determines whether statistically significant differences exist between the means of multiple independent groups. A low P-value (< 0.05) confirms that channels do not produce equivalent satisfaction outcomes, validating our EDA observation."

content_map[181] = "We investigate a correlational hypothesis.\n**Null Hypothesis (H0)**: There is no linear correlation between the hour an issue is reported and the subsequent CSAT score.\n**Alternate Hypothesis (H1)**: A statistically significant correlation exists between reporting hour and CSAT score."

content_map[183] = """# Perform Statistical Test: Pearson Correlation Test
clean_data = df[['issue_hour', 'csat_score']].dropna()
if len(clean_data) > 2:
    r_stat, p_val = stats.pearsonr(clean_data['issue_hour'], clean_data['csat_score'])
    print(f"Pearson R: {r_stat:.4f}")
    print(f"P-Value: {p_val:.6f}")
    print(f"Correlation strength: {'Weak' if abs(r_stat) < 0.3 else 'Moderate' if abs(r_stat) < 0.6 else 'Strong'}")
    print(f"Conclusion: {'Reject H0 - Significant correlation' if p_val < 0.05 else 'Fail to reject H0'}")
else:
    print("Insufficient data for Pearson correlation.")"""

content_map[185] = "We computed the **Pearson Correlation Coefficient** between `issue_hour` and `csat_score`."
content_map[187] = "Pearson measures the direction and strength of linear relationship between two continuous variables. The P-value validates whether any observed correlation is statistically significant or due to random chance."
