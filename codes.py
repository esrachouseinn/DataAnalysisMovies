import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set the plotting style
plt.style.use('ggplot')
from matplotlib.pyplot import figure

# Set default figure size for plots
plt.rcParams['figure.figsize'] = (12, 8)

# Suppress chained assignment warning
pd.options.mode.chained_assignment = None

# File path - update this to your file location
file_path = r'C:\Users\esra\Desktop\DataAnalysisProject2\movies.csv'

# Load the data
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    # Display the first few rows of the dataframe
    print(df.head())
except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")

# We need to see if we have any missing data
# Let's loop through the data and see if there is anything missing
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing * 100)))

print(df.dtypes)

# Are there any Outliers?
df.boxplot(column=['gross'])

df.drop_duplicates()

# Order our Data a little bit to see
df.sort_values(by=['gross'], inplace=False, ascending=False)

# Regression plots
sns.regplot(x="gross", y="budget", data=df)
sns.regplot(x="score", y="gross", data=df)

# Correlation Matrix between all numeric columns
# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=[np.number])

# Pearson correlation
correlation_matrix = numeric_df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation matrix for Numeric Features")
plt.xlabel("Movie features")
plt.ylabel("Movie features")
plt.show()

# Using factorize - this assigns a random numeric value for each unique categorical value
df.apply(lambda x: x.factorize()[0]).corr(method='pearson')
correlation_matrix = df.apply(lambda x: x.factorize()[0]).corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation matrix for Movies")
plt.xlabel("Movie features")
plt.ylabel("Movie features")
plt.show()

# Correlation pairs and strong correlations
correlation_mat = df.apply(lambda x: x.factorize()[0]).corr()
corr_pairs = correlation_mat.unstack()
sorted_pairs = corr_pairs.sort_values(kind="quicksort")
print(sorted_pairs)

# Strong correlations (> 0.5)
strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]
print(strong_pairs)

# Top 15 companies by gross revenue
CompanyGrossSum = df.groupby('company')[["gross"]].sum()
CompanyGrossSumSorted = CompanyGrossSum.sort_values('gross', ascending=False)[:15]
CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64')
print(CompanyGrossSumSorted)

# Adding a 'Year' column from 'released'
df['Year'] = df['released'].astype(str).str[:4]

# Grouping by company and year
CompanyGrossSum = df.groupby(['company', 'year'])[["gross"]].sum()
CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross', 'company', 'year'], ascending=False)[:15]
CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64')
print(CompanyGrossSumSorted)

# Scatter plot for Budget vs Gross Earnings
plt.scatter(x=df['budget'], y=df['gross'], alpha=0.5)
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()

# Numeric conversion for categorical columns
df_numerized = df.copy()

for col_name in df_numerized.columns:
    if df_numerized[col_name].dtype == 'object':
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

# Pearson correlation for numerized data
correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation matrix for Movies")
plt.xlabel("Movie features")
plt.ylabel("Movie features")
plt.show()

# Swarm plot and strip plot
sns.swarmplot(x="rating", y="gross", data=df.head(100))
plt.show()

sns.stripplot(x="rating", y="gross", data=df)
plt.show()

