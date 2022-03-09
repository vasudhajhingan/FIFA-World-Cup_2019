#!/usr/bin/env python
# coding: utf-8

# In[219]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('fifa.csv', encoding='UTF-8-SIG')


# In[217]:


import chardet 

with open('fifa.csv','rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
    
print(result)


# In[190]:


df.head(10)


# In[8]:


df.shape


# In[40]:


df.columns


# In[39]:


df.info()


# # Descriptive Statistics

# In[41]:


df.describe().T


# In[55]:


df["Age"].value_counts().head()
#21 age is the most common followed by 26 age


# In[56]:


df["Age"].value_counts().tail()
#41 and 40 age is the least common 
#That is after the age of 38, football players usually retire from the sport


# In[61]:


plt.boxplot(df["Age"])
#Minimum age is around 16 
#Maximum age is around 38 with 6 outliers 
#Mean is the age of 25
#First quartile is 20 
#Second quartile is 25


# # Some visualizations

# In[23]:


plt.figure(figsize = (10,7))
sns.countplot(df["Age"])
plt.title("Distribution of age")
#Most players are between the age of 21 - 26


# In[26]:


plt.figure(figsize = (12,9))
sns.countplot(df["Overall"])
plt.title("Distribution of overall rating")
#Most players are rated 66 


# In[159]:


#Finding the mean of overall rating 
np.mean(df["Overall"])


# In[158]:


#Finding the standard deviation of overall rating
np.std(df["Overall"])


# In[53]:


x = np.random.normal(df["Height_cm"])
plt.figure(figsize = (12,8))
plt.title("Distribution of height")
plt.hist(x)
plt.show() 
#Most common height is 175-185 cms


# In[165]:


#Finding the mean height 
np.mean(df["Height_cm"])


# In[166]:


#Finding the standard deviation of height
np.std(df["Height_cm"])


# In[54]:


x = np.random.normal(df["Weight_kg"])
plt.figure(figsize = (12,8))
plt.title("Distribution of weight")
plt.hist(x)
plt.show() 
#Most common weight is between 68-80 kg


# In[103]:


plt.figure(figsize = (15,5))
print("\nTop 10 nationalities with player numbers: \n\n", df["Nationality"].value_counts().head(10).plot(kind = 'bar'))
#England has the most amount of football players


# # Grouping Data

# # Data Cleaning 

# In[191]:


df.drop(columns = ["Unnamed: 0","Photo","Flag","Club Logo"],  axis = 1, inplace = True)


# In[192]:


df.groupby(["Age","Potential"]).mean()


# In[193]:


df.columns


# In[194]:


df.isnull().sum()


# In[195]:


#Since 48 is occuring over and over again, I'm checking whether it's for the same ID or not 
missing_height = df[df['Height'].isnull()].index.tolist()
missing_weight = df[df['Weight'].isnull()].index.tolist()
if missing_height == missing_weight:
    print('For same IDs ')
else:
    print('For different IDs')


# In[196]:


#Now, since it is for the same IDs, we can drop these rows 
df.drop(df.index[missing_height],inplace =True)


# In[197]:


df.shape


# In[198]:


#We initially had 18159 rows and dropped 48 columns and got 18159 rows.
18207-18159


# In[48]:


df.isnull().sum()


# In[199]:


#Dropping columns whose missing values are quite high 
df.drop(['Loaned From','Release Clause','Joined'],axis=1,inplace=True)


# In[200]:


df.columns


# In[50]:


df.shape
#Number of rows do not change
#85 columns - 3 dropped columns = 82 columns


# In[73]:


#finding duplicates 
df.duplicated()
#Found no duplicates


# # ANOVA 

# In[121]:


from scipy import stats
stats.f_oneway(df["Age"],df["Potential"])
#F-testscore is really high! 
#P-value is 0 
#So both columns are strong correlated.


# # Exploratory Data Analysis with visualizations

# In[201]:


#Changing height from string to float values
#1 inch = 2.54 cms
#Taking the first element of string and after connverting it to float multiplying with 12.0 and adding it to the third element of the string.
#Multiplying this whole thing with 2.54 cms
Height_cm = []

for i in list(df['Height'].values):
    try:
        Height_cm.append((float(str(i)[0])*12.0 + float(str(i)[2:]))*2.54)
    except ValueError:
        Height_cm.append(np.nan)
        
df['Height_cm'] = Height_cm


# In[202]:


#Dropping missing values from height 
df.dropna(inplace = True)


# In[203]:


#checking
print(df['Height_cm'].head())


# In[204]:


#Changing weight from string to float values and showing these values
df["Weight_kg"] = df["Weight"].str[:3].astype(float)/2.20462
df[["Name","Height_cm","Weight_kg"]].head()


# In[112]:


sns.jointplot(x=df["Height"],y=df["Weight"])
#Results show that as height increases, weight does as well.


# In[51]:


sns.jointplot(x=df["Age"],y=df["Potential"])
#Results show that a player's potential falls as he grows older


# In[149]:


sns.jointplot(x=df["Wage"],y=df["Value"])


# In[54]:


#Finding correlations 
df.corr()


# In[109]:


#Visualizaing correlations on a heatmap
plt.figure(figsize = (20,15))
sns.heatmap(df.corr())
plt.title("HeatMap of FIFA dataset")
#This also supports data grouping analysis done earlier


# In[108]:


#Showing the scatter plot for Standing Tackle and sliding tackle which are highly correlated
import matplotlib.pyplot as plt
x = df["StandingTackle"]
y = df["SlidingTackle"]
plt.scatter(x, y, color="r")
plt.title("Standing and sliding Tackle are highly correlated")
plt.show()


# In[137]:


df["Weight_kg"].plot(kind = 'hist')
#Most players are 70-75 kg 


# In[142]:


df["Height_cm"].plot(kind = 'hist')
#Most players are 175-185 cms


# In[210]:


def str2float(euros):
    if euros[-1]=='M':
        return float (euros[1:-1])*1000000
    elif euros[-1]=='K':
        return float(euros[1:-1])*1000
    else: 
        return float(euros[1:])

    #Converting string value and wage to float values
df["Value"] = df["Value"].apply(lambda x: str2float(x) )
df["Wage"]=df["Wage"].apply(lambda x: str2float(x))


# In[211]:


#Printing wage and value column to confirm the conversion
df[["Name","Wage","Value"]].head()


# In[144]:


# Sorting the top 10 players based on their wage
df.sort_values(by="Wage", ascending = False).head(10)
#We see that L. Messi has the highest wage of 565000 euros followed by L. Suarez at 455000 euros


# In[152]:


# Sorting the top 10 players based on their Value
df.sort_values(by="Value", ascending = False).head(10)
#Neymar Jr has the highest value of 118500000.0 followed by L.Messi at 110500000.0


# In[212]:


y = df[["Age","Nationality","Height_cm","Weight_kg","Value","Wage","Potential","Overall"]]

sns.pairplot(y)
plt.show()
#As potential increases, wage increases 
#As overall rating increases, value increases

