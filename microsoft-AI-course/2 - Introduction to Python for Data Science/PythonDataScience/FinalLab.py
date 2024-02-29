##### Final Lab Code Repository:
#####

DATAFRAME:
    
Index(['rank', 'major_code', 'major', 'major_category', 'total', 'sample_size',
       'men', 'women', 'sharewomen', 'employed', 'full_time', 'part_time',
       'full_time_year_round', 'unemployed', 'unemployment_rate', 'median',
       'p25th', 'p75th', 'college_jobs', 'non_college_jobs', 'low_wage_jobs',
       'sharemen', 'gender_diff'], dtype='object')
      
      

###
### PART ONE: Importing and Summarizing Data
###

### Read and explore your data

# Import pandas 
import pandas as pd

# Use pandas to read in recent_grads_url
recent_grads = pd.read_csv(recent_grads_url)

# Print the shape
print(recent_grads.shape)



### Exploring Your Data

# Print .dtypes
print(recent_grads.dtypes)

# Output summary statistics
print(recent_grads.describe())

# Exclude data of type object
print(recent_grads.describe(exclude=['object']))


### Replacing Missing Values

# Names of the columns we're searching for missing values 
columns = ['median', 'p25th', 'p75th']

# Take a look at the dtypes
print(recent_grads[columns].dtypes)

# Find how missing values are represented
print(recent_grads["median"].unique())

# Replace missing values with NaN
for column in columns:
    recent_grads.loc[recent_grads[column] == 'UN', column] = np.nan


### Select a Column

# Select sharewomen column
sw_col = recent_grads['sharewomen']

# Output first five rows
print(sw_col.head(5))


### Column Maximum Value

# Import numpy
import numpy as np

# Use max to output maximum values
max_sw = np.max(sw_col)

# Print column max
print(max_sw)


### Selecting a Row

# Output the row containing the maximum percentage of women
print(recent_grads.loc[recent_grads['sharewomen'] >= max_sw])


### Converting a DataFrame to Numpy Array

# Convert to numpy array
recent_grads_np = recent_grads[['unemployed', 'low_wage_jobs']].values

# Print the type of recent_grads_np
print(type(recent_grads_np))



### Correlation Coefficient

# Calculate correlation matrix
print(np.corrcoef(recent_grads_np[:,0], recent_grads_np[:,1]))





###
### PART TWO: Manipulating Data
###


### Creating Columns I

# Add sharemen column
recent_grads['sharemen'] = recent_grads['men'] / recent_grads['total']
# print(recent_grads['sharemen'])


### Select Row with Highest Value

# Find the maximum percentage value of men
max_men = np.max(recent_grads['sharemen'])
 
# Output the row with the highest percentage of men
print(recent_grads.loc[recent_grads['sharemen'] >= max_men])


### Creating columns II

# Add gender_diff column
recent_grads['gender_diff'] = recent_grads['sharewomen'] - recent_grads['sharemen']


### Updating columns

# Make all gender difference values positive
recent_grads['gender_diff'] = recent_grads['gender_diff'].abs()

# Find the 5 rows with lowest gender rate difference
# print(recent_grads['gender_diff'].nsmallest(5))
print(recent_grads.nsmallest(5, 'gender_diff'))


### Filtering rows

# Rows where gender rate difference is greater than .30 
diff_30 = recent_grads['gender_diff'] > .30

# Rows with more men
more_men = recent_grads['men'] > recent_grads['women']

# Combine more_men and diff_30
more_men_and_diff_30 = np.logical_and(more_men, diff_30)

# Find rows with more men and and gender rate difference greater than .30
fewer_women = recent_grads[more_men_and_diff_30.values]



### Grouping with Counts

# Group by major category and count
# print(recent_grads.____(['____']).major_category.____)
# print(recent_grads.groupby('major_category')['sharewomen'].count())
print(recent_grads.groupby(['major_category']).major_category.count())



### Grouping with Counts, Part 2

# Group departments that have less women by category and count
print(fewer_women.groupby(['major_category']).major_category.count())


### Grouping One Column with Means

# Report average gender difference by major category
print(recent_grads.groupby(['major_category']).gender_diff.mean())


### Grouping Two Columns with Means

# Find average number of low wage jobs and unemployment rate of each major category
dept_stats = ____.____(['____'])['____', '____'].____
print(dept_stats)



# Find average number of low wage jobs and unemployment rate of each major category
# print(fewer_women.groupby(['major_category']).major_category.count())
# print(recent_grads.groupby(['major_category']).gender_diff.mean())
# dept_stats = ____.____(['____'])['____', '____'].____

dept_stats = recent_grads.groupby(['major_category'])['low_wage_jobs', 'unemployment_rate'].mean()
print(dept_stats)




###
### PART THREE Visualizing Data
###


### Plotting Scatterplots

# Import matplotlib
import matplotlib.pyplot as plt

# Create scatter plot
plt.scatter(unemployment_rate.values, low_wage_jobs.values)

# Label x axis
plt.xlabel("Unemployment rate")

# Label y axis
plt.ylabel("Low pay jobs")

# Display the graph 
plt.show()



### Modifying Plot Colors

# Colors
# https://matplotlib.org/users/colors.html
# Markers
# https://matplotlib.org/api/markers_api.html

# Plot the red and triangle shaped scatter plot  
plt.scatter(unemployment_rate.values, low_wage_jobs.values, marker ="^", color='r')

# Display the visualization
plt.show()



### Plotting Histograms

# Create histogram of life_exp data
plt.hist(life_exp)

# Display histogram
plt.show()



### Plotting with pandas

# Import matplotlib and pandas
import matplotlib.pyplot as plt
import pandas as pd

# Create scatter plot
dept_stats.plot(kind='scatter', x='unemployment_rate', y='low_wage_jobs')
plt.show()


### Plotting One Bar Graphs

# Import matplotlib and pandas
import matplotlib.pyplot as plt
import pandas as pd

# Create histogram
recent_grads.sharewomen.plot(kind='hist')
plt.show()



### Plotting Two Bar Graphs

# DataFrame of college and non-college job sums
df1 = recent_grads.groupby(['major_category'])['college_jobs', 'non_college_jobs'].sum()

# Plot bar chart
df1.plot(kind='bar')

# Show graph
plt.show()



### Dropping Missing Values

# Print the size of the DataFrame
print(recent_grads.size)

# Drop all rows with a missing value
recent_grads = recent_grads.dropna()

# Print the size of the DataFrame
print(recent_grads.size)



### Plotting Quantiles of Salary, Part 1

# Convert to numeric and divide by 1000
# recent_grads['median'] = ____
# recent_grads['p25th'] = ____
# recent_grads['p75th'] = ____

recent_grads['median'] = pd.to_numeric(recent_grads['median'])/1000
recent_grads['p25th'] = pd.to_numeric(recent_grads['p25th'])/1000
recent_grads['p75th'] = pd.to_numeric(recent_grads['p75th'])/1000

# Select averages by major category
columns = ['median', 'p25th', 'p75th']
sal_quantiles = recent_grads.groupby(['major_category'])[columns].mean()



### Plotting Quantiles of Salary, Part 2

# Plot the data
sal_quantiles.plot()

# Set xticks
plt.xticks(
    np.arange(len(sal_quantiles.index)),
    sal_quantiles.index, 
    rotation='vertical')

# Show the plot
plt.show()

# Plot with subplots
sal_quantiles.plot(subplots=True)
plt.show()





###
### THE END.
###
















