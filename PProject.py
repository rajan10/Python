"""
    Python Project: by RAJAN GAUCHAN
"""
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr 
import os

os.getcwd()
os.chdir("c:/ds/python/project")
os.getcwd()

""" 
    Q1(a) Read the energy data from the file Energy Indicators.xls, which is a 
    list of indicators of [energy supply and renewable electricity production] 
    from the [United Nations](http://unstats.un.org/unsd/environment/excel_
    file_tables/2013/Energy%20Indicators.xls) for the year 2013, and should 
    be put into a Data Frame with the variable name of energy. Keep in mind 
    that this is an Excel file, and not a comma separated values file. Also, 
    make sure to exclude the footer and header information from the data file. 
    The first two columns are unnecessary, so you should get rid of them, and 
    you should change the column labels so that the columns are:
 
    ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable]
"""

# Data Import
energy = pd.read_excel('EnergyIndicators.xls',
                       sheetname= 'Energy', 
                       skiprows=17,
                       header=0,
                       skip_footer=38,
                       na_values=['NA'],
                       comment='...')

# Dropping first and second column from the dataset
energy.drop(energy.columns[[0,1]],axis=1,inplace=True)

# Renaming the column
energy.columns = ['Country','Energy Supply','Energy Supply per Capita',
                  '% Renewable']

energy.head()

# Saving the DF to excel file
energy.to_excel('energy.xls')


##############################################################################
""" 
    Q1(b) Rename the following list of countries.
    "Republic of Korea" to "South Korea",
    "United States of America" to "United States",
    "United Kingdom of Great Britain and Northern Ireland" to "United Kingdom",
    "China, Hong Kong Special Administrative Region" to "Hong Kong"
    There are also several countries with parenthesis in their name. 
    Be sure to remove these, e.g. `'Bolivia (Plurinational State of)'` 
    should be `'Bolivia'`.
"""
energy = pd.read_excel('energy.xls')

# Renaming countries names
old_names = ['Republic of Korea','United States of America20',
             'United Kingdom of Great Britain and Northern Ireland19',
             "China, Hong Kong Special Administrative Region3"]
new_names = ['South Korea','United States','United Kingdom','Hong Kong']

# Function to change the name of selected countries
def change_name(country):
    if country in old_names:
        return (new_names[old_names.index(country)])
    else:
        return country

# Function to remove parentheses
def remove_parentheses(country):
    if '(' in country:
        return (country[0:country.index('(')]).strip()
    else:
        return country
            
# Updating country name
energy['Country'] = energy['Country'].apply(change_name)

# Removing name in parentheses
energy['Country'] = energy['Country'].apply(remove_parentheses)
energy.to_excel("energy.xls")


##############################################################################
""" 
    Q1(c) Next, load the GDP data from the file world_bank.csv, which is a csv 
    containing countries' GDP from 1960 to 2015 from [World Bank] 
    (http://data.worldbank.org/indicator/NY.GDP.MKTP.CD). Call this Data Frame GDP. 
    Make sure to skip the header, and rename the following list of countries:
    "Korea, Rep." to "South Korea", 
    "Iran, Islamic Rep." to "Iran",
    "Hong Kong SAR, China" to "Hong Kong"```
"""

# Import Data: Reading 'world_bank.csv' file
GDP = pd.read_csv('world_bank.csv',
                  skiprows=4,
                  header=0,
                  na_values=['NA'])
GDP.rename(columns={'Country Name': 'Country'},inplace=True)

# Cleaning Data: Renaming the country names
old_names = ["Korea, Rep.","Iran, Islamic Rep.","Hong Kong SAR, China"]
new_names = ['South Korea','Iran','Hong Kong']

GDP['Country'] = GDP['Country'].apply(change_name)
GDP.to_excel('gdp.xls')


##############################################################################
""" 
    Q1(d) Finally, load the [Sciamgo Journal and Country Rank data for Energy 
    Engineering and Power Technology] (http://www.scimagojr.com/countryrank.php?
    category=2102) from the file scimagojr-3.xlsx, which ranks countries based 
    on their journal contributions in the aforementioned area. Call this Data Frame 
    ScimEn.
"""
                                 
# Import Data: Reading 'Scimagojr-3.xlsx' file
ScimEn = pd.read_excel('Scimagojr-3.xlsx')
ScimEn.head()
ScimEn.to_excel('scimen.xls')

#list(ScimEn.columns.values)

##############################################################################
""" 
    Q1(e) Join the three datasets: GDP, Energy, and ScimEn into a new dataset 
    (using the intersection of country names). Use only the last 10 years 
    (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' 
    (Rank 1 through 15). The index of this Data Frame should be the name of the 
    country, and the columns should be ['Rank', 'Documents', 'Citable documents', 
    'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy 
    Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', 
    '2009', '2010', '2011', '2012', '2013', '2014', '2015']. You should finally 
    get a Data Frame with 20 columns and 15 entries.
"""
# Import datasets
energy = pd.read_excel('energy.xls')
GDP = pd.read_excel('gdp.xls')
ScimEn = pd.read_excel('scimen.xls')
    
# Merging datasets 'energy, gdp, scimen' into new_ds
merged_df= pd.merge(pd.merge(energy,GDP, on='Country',how='outer'),
                    ScimEn,on='Country',how='outer')

# Dropping columns for last 10 years (2006-2015)   
merged_df.set_index('Country',inplace=True)
df = merged_df[['Rank', 'Documents', 'Citable documents', 
                'Citations', 'Self-citations', 'Citations per document', 
                'H index', 'Energy Supply', 'Energy Supply per Capita', 
                '% Renewable', '2006', '2007', '2008', '2009', '2010', 
                '2011', '2012', '2013', '2014', '2015']]

# Subsetting dataset according to Rank
df = (df.loc[df['Rank'].isin([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])])
df = df.sort_values('Rank')

df['Rank'].head(15)
df.to_excel('new_ds.xls')


##############################################################################
""" 
    Q2: What are the top 15 countries for average GDP over the last 10 years? 
    [NB: This function should return a Series named ‘avgGDP’ with 15 countries 
    and their average GDP sorted in descending order.]
"""
                                 
df = pd.read_excel('new_ds.xls')
list(df.columns.values)

def avgfunc(df):
    df['avgGDP'] = df[['2006','2007','2008','2009','2010','2011','2012',
      '2013','2014','2015']].mean(axis=1).sort_values(ascending=False)   
    return df['avgGDP']

print(avgfunc(df))
df[['Country','Rank','avgGDP']].head()

df.to_excel('new_ds.xls')

###################################################################################
"""
    Q3: By how much had the GDP changed over the 10 year span for the country with 
        the 6th largest average GDP? [NB: This function should return a single number.]
"""

df = pd.read_excel('new_ds.xls',na_values=['NA'])
list(df.columns.values)

def calc_growth():
    df['rank_gdp'] = df['avgGDP'].rank(ascending=0)
    df[['avgGDP','rank_gdp']].head(15)
    largest_6th = df[df['rank_gdp']==6]
    
    gdp_2015 = largest_6th['2015']
    gdp_2006 = largest_6th['2006']
    g = ((gdp_2015/gdp_2006)**10 - 1) * 100
    return g

print('GDP growth rate:'+ str(round(calc_growth(),2)))


###################################################################################
"""
    Q4: What is the mean energy supply per capita? 
    [NB: This function should return a single number.]
"""

def mean_espc():
    return df['Energy Supply per Capita'].mean()

print("Mean Energy Supply per Capita: {}".format(round(mean_espc(),2)))


###################################################################################
"""
    Q5: Which country has the maximum % Renewable and what is the percentage? 
    [NB: This function should return a tuple with the name of the country 
    and the percentage.]
"""
def max_renewable(df):
    max = df.iloc[:,10].max()
    ct = df.sort_values(by='% Renewable',ascending=False).iloc[0]
    return (ct.Country,max)
    
print(max_renewable(df))


###################################################################################
"""
    Q6: Create a new column that is the ratio of Self-Citations to Total Citations. 
    What is the maximum value for this new column, and which country has the 
    highest ratio? 
    [NB: This function should return a tuple with the name of the country and 
    the ratio.]
"""

def ratiofunc():
    df['ratio'] = df['Self-citations']/df['Citations']
    max_ratio = df['ratio'].max()
    #df[['Country','ratio']]
    ct = df.sort_values(by='ratio',ascending=False).iloc[0]
    return (ct.Country,max_ratio)

print(ratiofunc())


###################################################################################
"""
    Q7: Create a column that estimates the population using Energy Supply and 
    Energy Supply per capita. What is the third most populous country 
    according to this estimate? [NB: This function should return a single 
    string value.]
"""

def get_country():
    # Calculating population 
    df['Population'] = df['Energy Supply']/df['Energy Supply per Capita']
    # Creating rank column based on population of the country
    df['Pop_rank'] = df['Population'].rank(ascending=0) 
    country = df['Country'][df['Pop_rank']==3]
    ct = df.sort_values(by='Population',ascending=False).iloc[2]
    return ct.Country

print('Third most populous country :'+get_country())

df.to_excel('new_ds.xls') # to save new columns in dataset


###################################################################################
"""
    Q8: Create a column that estimates the number of citable documents per person. 
    What is the correlation between the number of citable documents per 
    capita and the energy supply per capita? Use the “.corr()” method, 
    (Pearson's correlation). 
    [NB: This function should return a single number.] Plot to visualize 
    the relationship between Energy Supply per Capita vs. Citable docs per 
    Capita.
"""

df = pd.read_excel('new_ds.xls',na_values=['NA'])
list(df.columns.values)
df.dropna(inplace=True)

# Calculating correlation
def get_corr():
    df['Citable doc per Capita'] = df['Citable documents']/(df['Energy Supply']/df['Energy Supply per Capita'])    
    # Calculating Pearson Correlation
    corr = pearsonr(df['Citable doc per Capita'],df['Energy Supply per Capita'])
    return corr[0]

print('Pearson Correlation Coefficient: '+str(round(get_corr(),2)))

# Visualization
def visualize():
    energy = df['Energy Supply per Capita']
    citable = df['Citable doc per Capita']
    plt.scatter(energy,citable)
    plt.xlabel('Citable Doc Per Capita', fontsize=16)
    plt.ylabel('Energy Supply Per Capita', fontsize=16)
    plt.title('Scatter plot\n Energy Supply Per Capita vs. Citable Doc Per Capita ',
              fontsize=20)
    plt.show()
    
visualize()
    
    
###################################################################################
"""
    Q9: Create a new column with a 1 if the country's % Renewable value is at or 
    above the median for all countries in the top 15, and a 0 if the country's 
    % Renewable value is below the median. 
    [NB: This function should return a series named “HighRenew” whose index is 
    the country name sorted in ascending order of rank.]
"""
df = pd.read_excel('new_ds.xls',na_values=['NA'])
df.dropna(inplace=True)
list(df.columns.values)

def funcQ9():
    med = df['% Renewable'].median()
    df['HighRenew'] = df['% Renewable'].apply(lambda x: 1 if x>=med else 0)
    #df['HighRenew'] = np.where(df['% Renewable']>=med, 1, 0)   
    return pd.Series(df['HighRenew'])
    
s = funcQ9()


###################################################################################
"""
    Q10: Use the following dictionary to group the Countries by Continent, then 
    create a dataframe that displays the sample size (the number of countries 
    in each continent bin), and the sum, mean, and std deviation for the 
    estimated population of each continent.
    ContinentDict  = {'China':'Asia', 
                       'United States':'North America', 
                       'Japan':'Asia', 
                       'United Kingdom':'Europe', 
                       'Russian Federation':'Europe', 
                       'Canada':'North America', 
                       'Germany':'Europe', 
                       'India':'Asia',
                       'France':'Europe', 
                       'South Korea':'Asia', 
                       'Italy':'Europe', 
                       'Spain':'Europe', 
                       'Iran':'Asia',
                       'Australia':'Australia', 
                       'Brazil':'South America'}
   [NB: This function should return a DataFrame with index named Continent 
   ['Asia', 'Australia', 'Europe', 'North America', 'South America'] and with 
   columns ['size', 'sum', 'mean', 'std'].]
"""
# Creating dictionary
ContinentDict = {'China':'Asia','United States':'North America', 'Japan':'Asia', 
                   'United Kingdom':'Europe','Russian Federation':'Europe', 
                   'Canada':'North America','Germany':'Europe','India':'Asia',
                   'France':'Europe','South Korea':'Asia', 'Italy':'Europe', 
                   'Spain':'Europe', 'Iran':'Asia','Australia':'Australia', 
                   'Brazil':'South America'}

# function to counting countries in continents
def get_continent(x):
   if x in ['China','Japan','India','South Korea','Iran']:
       return 'Asia'
   elif x in ['United States','Canada']:
       return 'North America'
   elif x in ['United Kingdom','Russian Federation','Germany','France','Italy','Spain']:
       return 'Europe'
   elif x in ['Australia']:
       return 'Australia'
   elif x in ['Brazil']:
       return 'South America'

# function to return dataframe with the statistics
def func10():
    df['Continent'] = df['Country'].apply(get_continent) 
    
    continent = ['Asia','Australia','Europe','North America','South America']
    size = df['Population'].groupby(df['Continent']).count()
    sum = df['Population'].groupby(df['Continent']).sum()
    mean = df['Population'].groupby(df['Continent']).mean()
    std = df['Population'].groupby(df['Continent']).std()
    group = list(zip(continent,size,sum,mean,std))
    
    new_df = pd.DataFrame(data=group,columns =['Continent','Size','Sum','Mean','Std'])
    new_df.set_index('Continent',inplace=True)

    return new_df    

final_df = func10()
