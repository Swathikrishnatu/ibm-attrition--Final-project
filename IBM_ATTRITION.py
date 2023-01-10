#!/usr/bin/env python
# coding: utf-8

# # IBM HR Attrition

# Attrition: When an employee leaves the company due to resignation or retirement, then it is called Attrition. Employees leave the company for personal and professional reasons like retirement, lower growth potential, lower work satisfaction, lower pay rate, bad work environment, etc. Attrition is part and parcel of any business. Attrition is a cause of concern when it crosses a limit.
# 
# The attrition rate, also known as churn rate, can be defined as the rate at which employees leave an organization from a specific group over a particular period of time.
# 
# The dataset for the analysis is taken from Kaggle. To get insights about what factors contribute to employee attrition, we use Python and libraries like pandas, matplotlib, and seaborn. In this blog, we mostly talk about absolute and percentage values.

# In[1]:


#importing packages
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#calling data
HR_data=pd.read_csv('HR dataset-team 7(1).csv')
HR_data.head()


# In[3]:


#basic informations
HR_data.info()


# In[4]:


#data types
HR_data.dtypes


# The data types are float and object.

# In[5]:


#shape
HR_data.shape


# There are 23436 rows and 37 columns

# In[6]:


#features & target
HR_data.columns


# In all we have 36 features consisting of both the categorical as well as the numerical features. The target variable is the 'Attrition' of the employee which can be either a Voluntary resignation or Currently working.

# In[7]:


#numerical columns
num_cols=[features for features in HR_data.columns if HR_data[features].dtypes !='O']
num_cols


# In[8]:


#categorical columns
cat_cols=HR_data.select_dtypes(include='object')
cat_cols.columns


# There are 25 numerical columns and 12 categorical columns.

# In[9]:


#checking unique values
HR_data.nunique()


# Dropping two columns.

# In[10]:


HR_data=HR_data.drop(['EmployeeNumber'],axis=1)


# In[11]:


HR_data=HR_data.drop(['Application ID'],axis=1)


# In[12]:


HR_data.head()


# In[13]:


#descriptive statistics
HR_data.describe().T


# In[14]:


#null values
HR_data.isna().sum()


# In[15]:


#null value percentage
Nullvalue_percentage=(HR_data.isna().sum()/len(HR_data))*100
Nullvalue_percentage


# In[16]:


Total_nullvalue_percentage=Nullvalue_percentage.sum()
Total_nullvalue_percentage


# The total percentage of null values present in the dataset is 1.5%.

# In[17]:


#histogram for numerical features
HR_data.hist(figsize=(18, 14))


# # Inference from histogram
# 1)The age graph is almost normally distibuted.The minimum & maximum age of workforce is 18 & 60.Most of the employees are in between the age group 30-40.
# 2)The daily rate is between 100 and 1500.
# 3)Majority of the employees living space distance from company is less than 10km.
# 4)Most of the employees education level is 3.
# 5)The higest rating for the environment satisfaction is 3 & 4.
# 6)The hourly rate is between 20-100.
# 7)The higest rating for job involvement is 3.
# 8)Majority of the employees joblevel is 1 & 2.
# 9)3 & 4 has the higest rating for job satisfaction.
# 10)Monthly salary is between 1000-20000.
# 11)Major number of employees worked for more than one company before joining IBM.
# 12)Many of them have salary hike percentage less than 15%.
# 13)3 is the highest performance rating of employees.
# 14)Most of the employees relationship satisfaction rating is 3 & 4.
# 15)The satandared working hours of employees is 80hr.
# 16)Only few employees have taken stock option plan.
# 17)Most of the employees working years is 20 and below.
# 18)Many employees got 2 time training for last year.
# 19)3 is the higest rating for work life balance.
# 20)Many of the employees are working for IBM for less than 13 years.
# 21)Emloyees are working in the current role for less than 10 years.
# 22)Majority of them got promotions & working with the current manager.

# In[18]:


#outlier visualization 
HR_data.plot(kind='box',subplots=True,layout=(5,5),figsize=(15,15),title='Outlier Visualization')
plt.show()


# Some columns in the dataset contain outliers.It will be handelled during data pre-processing.

# # 1) Exploratory Data Visualization

# # 1.1) Univariate Visualization

# # a)Target Column

# In[19]:


#count plot of target.
sns.countplot(x='Attrition', data=HR_data)


# In[20]:


#pie chart
plt.rcParams['figure.figsize'] =5,5
labels = HR_data['Attrition'].value_counts().index.tolist()
sizes = HR_data['Attrition'].value_counts().tolist()
explode = (0, 0.1)
colors = ['yellowgreen', 'lightcoral']
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',startangle=90, textprops={'fontsize': 14})
plt.show()


# # Inference
# The figure showing is the figure of employee attrition. In this dataset, 3709 employees left the company while 19714 stay(ie only 15.8% of the employees left, rest are still working in IBM).The data is very imbalanced.

# # *I) Visualization for Numerical Features

# # b)Age

# In[21]:


#distplot of Age column
plt.figure(figsize=[10,8])
sns.distplot(HR_data['Age'],hist=True,kde=True,color='k',bins=10)


# # Inference
# The graph is normally distributed.
# The minimum age of workforce is 18.
# The maximum age of workforce is 60.
# Majority of the employees lie between the age range 30-40.

# # * distplot of all numerical features

# In[22]:


fig,ax = plt.subplots(6,4, figsize=(12,12))                
sns.distplot(HR_data['DailyRate'], ax = ax[0,0]) 
sns.distplot(HR_data['DistanceFromHome'], ax = ax[0,1]) 
sns.distplot(HR_data['Education'], ax = ax[0,2]) 
sns.distplot(HR_data['EmployeeCount'], ax = ax[0,3]) 
sns.distplot(HR_data['EnvironmentSatisfaction'], ax = ax[1,0]) 
sns.distplot(HR_data['HourlyRate'], ax = ax[1,1]) 
sns.distplot(HR_data['JobInvolvement'], ax = ax[1,2]) 
sns.distplot(HR_data['JobLevel'], ax = ax[1,3]) 
sns.distplot(HR_data['JobSatisfaction'], ax = ax[2,0]) 
sns.distplot(HR_data['MonthlyIncome'], ax = ax[2,1])
sns.distplot(HR_data['MonthlyRate'], ax = ax[2,2])
sns.distplot(HR_data['NumCompaniesWorked'], ax = ax[2,3])
sns.distplot(HR_data['PercentSalaryHike'], ax = ax[3,0])
sns.distplot(HR_data['PerformanceRating'], ax = ax[3,1])
sns.distplot(HR_data['RelationshipSatisfaction'], ax = ax[3,2])
sns.distplot(HR_data['StandardHours'], ax = ax[3,3])
sns.distplot(HR_data['StockOptionLevel'], ax = ax[4,0])
sns.distplot(HR_data['TotalWorkingYears'], ax = ax[4,1])
sns.distplot(HR_data['TrainingTimesLastYear'], ax = ax[4,2])
sns.distplot(HR_data['WorkLifeBalance'], ax = ax[4,3])
sns.distplot(HR_data['YearsAtCompany'], ax = ax[5,0])
sns.distplot(HR_data['YearsInCurrentRole'], ax = ax[5,1])
sns.distplot(HR_data['YearsSinceLastPromotion'], ax = ax[5,2])
sns.distplot(HR_data['YearsWithCurrManager'], ax = ax[5,3])
plt.tight_layout()
plt.show()


# # Inference
# Using distplot we find out the distribution of all numerical features.Here we can see that EmployeeCount and StandardHours does not have much influence.

# # b) Distance from home

# In[23]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['DistanceFromHome'])


# In[24]:


print('Distance count')
print(HR_data.DistanceFromHome.value_counts())
print('Distance in percentage')
print(HR_data.DistanceFromHome.value_counts()*100/len(HR_data))


# # Inference
# From this graph we can see that 28% employees live near to the company,ie within 1 or 2km.

# # c)Education

# In[25]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['Education'])


# In[26]:


print('Education count')
print(HR_data.Education.value_counts())
print('Education in percentage')
print(HR_data.Education.value_counts()*100/len(HR_data))


# # Inference
# This feature have five levels.They are:
# 1:'Below College',2:'College',3:'Bachelor',4:'Master',5:'Doctor'.
# Around 39% of employees have Bachelors level of education.

# # d) Environment satisfaction

# In[27]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['EnvironmentSatisfaction'])


# In[28]:


print('Environment satisfaction count')
print(HR_data.EnvironmentSatisfaction.value_counts())
print('Environment satisfaction in percentage')
print(HR_data.EnvironmentSatisfaction.value_counts()*100/len(HR_data))


# # Inference
# It is the satisfaction of the employee with the working environment.This has four levels:
#     1:'Low',2:'Medium',3:'High',4:'Very High'.
# 60% employees vote for level 3 & 4 ie they are satisfied with the working environment.Rest of them vote for level 1 & 2.                    

# # e)Hourly rate

# In[29]:


plt.figure(figsize=[10,10])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['HourlyRate'])


# Hourly rate is between 20 to 100.

# # f)Job involvement

# In[30]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['JobInvolvement'])


# In[31]:


print('Involvement count')
print(HR_data.JobInvolvement.value_counts())
print('Involvement in percentage')
print(HR_data.JobInvolvement.value_counts()*100/len(HR_data))


# # Inference
# It contains four level.They are:
#      1:'Low',2:'Medium',3:'High',4:'Very High'.
# 59% are voting for level 3,ie they are highly involved in their job.                        

# # g)Job level

# In[32]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['JobLevel'])


# In[33]:


print('Job level count')
print(HR_data.JobLevel.value_counts())
print('Job level in percentage')
print(HR_data.JobLevel.value_counts()*100/len(HR_data))


# # Inference
# It's the position of employees in the company.72% of employees are at 1st and 2nd level.

# # h) Job satisfaction

# In[34]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['JobSatisfaction'])


# In[35]:


print('Job satisfaction count')
print(HR_data.JobSatisfaction.value_counts())
print('Job satisfaction in percentage')
print(HR_data.JobSatisfaction.value_counts()*100/len(HR_data))


# # Inference
# It is the satisfaction level of employees in their job.There are four level:
#     1:'Low',2:'Medium',3:'High',4:'Very High'.
# 61% of employees are satisfied in their job, ie they are rating 3 & 4.                    

# # i)Monthly income

# In[36]:


plt.figure(figsize=[10,10])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['MonthlyIncome'])


# # j)Number of companies worked

# In[37]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['NumCompaniesWorked'])


# In[38]:


print('No of companies worked count')
print(HR_data.NumCompaniesWorked.value_counts())
print('No of companies worked in percentage')
print(HR_data.NumCompaniesWorked.value_counts()*100/len(HR_data))


# # Inference
# The number of companies in which the emloyees worked before they joined IBM.Different employees worked for nine different companies.35% of employees worked for only 1 company before joining IBM.

# # k)Percentage salary hike

# In[39]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['PercentSalaryHike'])


# In[40]:


print('Salary hike count')
print(HR_data.PercentSalaryHike.value_counts())
print('Salary hike in percentage')
print(HR_data.PercentSalaryHike.value_counts()*100/len(HR_data))


# # Inference
# It's the percentage of salary hike.The many employees got upto 14% of salary hike every year.

# # l)Performance rating

# In[41]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['PerformanceRating'])


# In[42]:


print('Performance rating count')
print(HR_data.PerformanceRating.value_counts())
print('Performance rating in percentage')
print(HR_data.PerformanceRating.value_counts()*100/len(HR_data))


# # Inference
# It's the performance of the employee in the company.
# 1:'Low',2:'Good',3:'Excellent',4:'Outstanding'.
# All the employees have performance rating 3 and 4,ie all are performing their maximum.                

# # m)Relationship satisfaction

# In[43]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['RelationshipSatisfaction'])


# In[44]:


print('Relationship satisfaction count')
print(HR_data.RelationshipSatisfaction.value_counts())
print('Relationship satisfaction in percentage')
print(HR_data.RelationshipSatisfaction.value_counts()*100/len(HR_data))


# # Inference
# Contain four levels.
# 1:'Low',2:'Medium',3:'High',4:'Very High'.
# 60% of them are rating 3 & 4.                

# # n)Stock option level

# In[45]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['StockOptionLevel'])


# In[46]:


print('Stock option level count')
print(HR_data.StockOptionLevel.value_counts())
print('Stock option level in percentage')
print(HR_data.StockOptionLevel.value_counts()*100/len(HR_data))


# # Inference
# Stock option plan is an employee benefit plan isssued by the company to encourage employee ownership in the company.But 43% of the employees doesnt take that plan.

# # o)Total working years

# In[47]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['TotalWorkingYears'])


# In[48]:


print('Total working years count')
print(HR_data.TotalWorkingYears.value_counts())
print('Total working years in percentage')
print(HR_data.TotalWorkingYears.value_counts()*100/len(HR_data))


# # Inference
# 13% of the employees have work experience of 10 years.

# # p)Training times last year

# In[49]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['TrainingTimesLastYear'])


# In[50]:


print('Training times last year count')
print(HR_data.TrainingTimesLastYear.value_counts())
print('Training times last year in percentage')
print(HR_data.TrainingTimesLastYear.value_counts()*100/len(HR_data))


# # Inference
# How much time does the employee got training for last year.
# 70% of them got training for 2 to 3 times.

# # q)Work life balance

# In[51]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['WorkLifeBalance'])


# In[52]:


print('Work life balance count')
print(HR_data.WorkLifeBalance.value_counts())
print('Work life balance in percentage')
print(HR_data.WorkLifeBalance.value_counts()*100/len(HR_data))


# # Inference 
# It's the rating given by employees that how their work and personal life is balanced.
# It has four level;1:'Low',2:'Good',3:'Better',4:'Best'.
# 60% of them are rating as 3.               

# # r)Years at company

# In[53]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['YearsAtCompany'])


# In[54]:


print('Years at company count')
print(HR_data.YearsAtCompany.value_counts())
print('Years at company in percentage')
print(HR_data.YearsAtCompany.value_counts()*100/len(HR_data))


# # Inference
# How many years in which the employee is working for IBM.
# 13% of the employees are working for 5 years.

# # s)Years in current role

# In[55]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['YearsInCurrentRole'])


# In[56]:


print('Years in current in role count')
print(HR_data.YearsInCurrentRole.value_counts())
print('Years in current role in percentage')
print(HR_data.YearsInCurrentRole.value_counts()*100/len(HR_data))


# # Inference
# 25% of employees are working in the same role  for 2 years.

# # s)Years since last promotion

# In[57]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['YearsSinceLastPromotion'])


# In[58]:


print('Years since last promotion count')
print(HR_data.YearsSinceLastPromotion.value_counts())
print('Years since last promotion in percentage')
print(HR_data.YearsSinceLastPromotion.value_counts()*100/len(HR_data))


# # Inference
# It's the years passed since their last promotion.
# The higest count is for 0 years.

# # t)Years with current manager

# In[59]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['YearsWithCurrManager'])


# In[60]:


print('Years with current manager count')
print(HR_data.YearsWithCurrManager.value_counts())
print('Years with current manager in percentage')
print(HR_data.YearsWithCurrManager.value_counts()*100/len(HR_data))


# # Inference
# 23% of the employees are working under the current manager for 2 years.

# # *II)Visualization for Categorical Features

# # a)Business travel

# In[61]:


sns.countplot(x='BusinessTravel', data=HR_data)


# In[62]:


print('Business travel count')
print(HR_data.BusinessTravel.value_counts())
print('Business travel in percentage')
print(HR_data.BusinessTravel.value_counts()*100/len(HR_data))


# # Inference
# 70% of the employees belong to travel rarely group. This indicates that most of them did not have a job which asked them for frequent travelling.

# # b)Department

# In[63]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['Department'])


# In[64]:


print('Department count')
print(HR_data.Department.value_counts())
print('Department in percentage')
print(HR_data.Department.value_counts()*100/len(HR_data))


# # Inference
# 65% of employees work under Research & Development department.

# # c)Education field

# In[65]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['EducationField'])


# In[66]:


print('Education field count')
print(HR_data.EducationField.value_counts())
print('Education field in percentage')
print(HR_data.EducationField.value_counts()*100/len(HR_data))


# # Inference
# 72% of the employees education field are Life science and Medical.

# # d)Gender

# In[67]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['Gender'])


# In[68]:


print('Gender count')
print(HR_data.Gender.value_counts())
print('Gender in percentage')
print(HR_data.Gender.value_counts()*100/len(HR_data))


# # Inference
# Majority(59%) of the employees working in IBM are males.

# # e)Job role

# In[69]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['JobRole'])


# In[70]:


print('Job role count')
print(HR_data.JobRole.value_counts())
print('Job role in percentage')
print(HR_data.JobRole.value_counts()*100/len(HR_data))


# # Inference
# Around 22% of the employees are Sales executive, 20% areResearch scientist and 18% areLaboratory technician.

# # f)Marital status

# In[71]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['MaritalStatus'])


# In[72]:


print('Marital status count')
print(HR_data.MaritalStatus.value_counts())
print('Marital status in percentage')
print(HR_data.MaritalStatus.value_counts()*100/len(HR_data))


# # Inference
# 45% of the employees are married.

# # g)Over 18

# In[73]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['Over18'])


# # Inference 
# All the employees working in the company are above the age of 18.

# # h)Over time

# In[74]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['OverTime'])


# In[75]:


print('Over time count')
print(HR_data.OverTime.value_counts())
print('Over time in percentage')
print(HR_data.OverTime.value_counts()*100/len(HR_data))


# # Inference
# Only around 28% of the employees are working over time.

# # i)Employee source

# In[76]:


plt.figure(figsize=[5,5])
plt.xticks(rotation='vertical')
sns.countplot(HR_data['Employee Source'])


# # Inference
# The major source of employees are from company website.

# # >*Cor-relation between features

# In[77]:


f, ax = plt.subplots(figsize=(20, 20))
corr = HR_data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True)


# In[78]:


#cor-relation in percentage values
corr=HR_data.corr()
import  seaborn as sns 
plt.figure(figsize=[20,15])
sns.heatmap(corr,annot=True,cmap='YlGnBu',fmt='.0%')


# # Inference
# From the correlation table we see that monthly income is highly correlated with job level as expected as senior employees will definately earn more. However, daily rate, hourly rate and monthly rate are barely correlated with anything. We will be using monthly income in later analysis as a measurement of salary and get rid of other income related variables.Employee count and Standard hours also does not have any effect.
# 
# SOME OTHER INFERENCES FROM THE ABOVE HEATMAP:
# 1)Job level and total working years are highly correlated which is expected as senior employees must have worked for a larger span of time
# 2)Monthly Income and total working years are highly correlated which is expected as the employee with more work experince will earn more salary.
# 3)Years in current role and years at company are highly correlated.
# 4)Years with current manager and years at company are highly correlated.
# 5)Self relation ie of a feature to itself is equal to 1 as expected.

# # 1.2)Bivariate Visulaization

# # >I)Plotting the Features against the 'Target' variable.

# # a)Age vs Attririon

# In[79]:


sns.factorplot(data=HR_data,y='Age',x='Attrition',aspect=1,kind='bar')


# # Inference
# This graph shows that younger age group are leaving the company(ie below 35) and that the people with higher age have lesser tendency to leave the company which makes sense as they may have settled in the organisation.

# # b)Bussines travel vs Attrition

# In[80]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='BusinessTravel')


# In[81]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.BusinessTravel],margins=True,normalize='index')


# # Inference
# The employees who travel frequently have higher percentage(24%) of leaving the company. 

# # c)Department vs Attrition

# In[82]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='Department')


# In[83]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.Department],margins=True,normalize='index')


# # Inference
# The higest percentage of voluntary resignation happens in sales department(20%).They may be leaving the company due to reasons like greater workfoce,lower salary,etc.
# The currently working employee percent is higher in research & development department(86%).

# # d)Distance from home vs Attrition

# In[84]:


plt.figure(figsize=[10,10])
sns.countplot(x='DistanceFromHome',hue='Attrition',data=HR_data)
plt.show()


# In[85]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.DistanceFromHome],margins=True,normalize='index')


# # Inference
# The employees with greater distance are leaving company.

# # e)Education vs Attrition

# In[86]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='Education')


# In[87]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.Education],margins=True,normalize='index')


# # Inference
# From this graph we can observe that the employees with education level 1 have the higher percentage(18%) of voluntary resignation(ie they are leaving the company)

# # f)Education field vs Attrition

# In[88]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='EducationField')


# In[89]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.EducationField],margins=True,normalize='index')


# # Inference
# The employees having technical degree have higher ratio(22%) of voluntary resignation.The employee with life science as the education field have lesser chance of leaving the company.

# # g)Environment satisfaction vs Attrition

# In[90]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='EnvironmentSatisfaction')


# In[91]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.EnvironmentSatisfaction],margins=True,normalize='index')


# # Inference
# Again we can notice that the relatively high percent of 'current employee' in employees with higher grade of environment satisfacftion.This means that they are satisfied with their working environment.

# # h)Gender vs Attrition

# In[92]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='Gender')


# In[93]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.Gender],margins=True,normalize='index')


# # Inference
# About 85 % of females want to stay in the company while only 15 % want to leave. All in all 84 % of employees want to be in the company with only being 16% wanting to leave the company.

# # i)Job involvement vs Attrition

# In[94]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='JobInvolvement')


# In[95]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.JobInvolvement],margins=True,normalize='index')


# # Inference
# Again we can notice that the relatively high percent of 'current employee' in employees with higher grade of job involvement.This means that they are doing  their maximum.

# # j)Job level vs Attrition

# In[96]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='JobLevel')


# In[97]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.JobLevel],margins=True,normalize='index')


# # Inference
# There is a higher rate of voluntary resignation in employees having job level 1(20%).

# # k)Job role vs Attrition

# In[98]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='JobRole')


# In[99]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.JobRole],margins=True,normalize='index')


# # Inference
# The lower ratio of voluntary resignation happens in manager role.The higher is in sales representatives,this may due to workforce,low salary,etc.

# # l)Job satisfaction vs Attrition

# In[100]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='JobSatisfaction')


# In[101]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.JobSatisfaction],margins=True,normalize='index')


# # Inference
#  Note that for higher values of job satisfaction( ie more a person is satisfied with his job) lesser percent of voluntary resignation which is quite obvious as highly contented workers will obvioulsy not like to leave the company.

# # m)Marital status vs Attrition

# In[102]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='MaritalStatus')


# In[103]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.MaritalStatus],margins=True,normalize='index')


# # Inference
# Single peopel are more likely to quit compared to married and divorced people.

# # n)Over time vs Attrition

# In[104]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='OverTime')


# In[105]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.OverTime],margins=True,normalize='index')


# # Inference
# The over time working employees are more likely to quit.

# # o)Performance rating vs Attrition

# In[106]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='PerformanceRating')


# In[107]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.PerformanceRating],margins=True,normalize='index')


# # Inference
# Attrition ratio for both performance level is equal .

# # p)Relationship satisfaction vs Attrition

# In[108]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='RelationshipSatisfaction')


# In[109]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.RelationshipSatisfaction],margins=True,normalize='index')


# # Inference
# The lower ratio of voluntary resignation is with level 2.

# # q)Stock option level vs Attrition

# In[110]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='StockOptionLevel')


# In[111]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.StockOptionLevel],margins=True,normalize='index')


# # Inference
# The level 0 has higher ratio of voluntary resignation.

# # r)Training times last year vs Attrition

# In[112]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='TrainingTimesLastYear')


# In[113]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.TrainingTimesLastYear],margins=True,normalize='index')


# # Inference
# The employees who does not get any training during last year are most likely to resing.

# # s)Work life balance vs Attrition

# In[114]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='WorkLifeBalance')


# In[115]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.WorkLifeBalance],margins=True,normalize='index')


# # Inference
# The lower rating employees are likely to quit the company.They have bad level of work-life balance.

# # t)Number of companies worked vs Attrition

# In[116]:


plt.figure(figsize=[10,10])
sns.countplot(x='NumCompaniesWorked',hue='Attrition',data=HR_data)
plt.show()


# In[117]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.NumCompaniesWorked],margins=True,normalize='index')


# # Inference
# The employees who have worked for many companies have higher chance of leaving the company.

# # u)Years with current manager vs Attrition

# In[118]:


plt.figure(figsize=[10,10])
sns.countplot(x='YearsWithCurrManager',hue='Attrition',data=HR_data)
plt.show()


# In[119]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.YearsWithCurrManager],margins=True,normalize='index')


# # Inference
# The greater ratio of attrition happens in the employee who had worked 14 years with the current manager.

# # v)Years since last promotion vs Attrition

# In[120]:


plt.figure(figsize=[10,10])
sns.countplot(x='YearsSinceLastPromotion',hue='Attrition',data=HR_data)
plt.show()


# In[121]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.YearsSinceLastPromotion],margins=True,normalize='index')


# 25% of attrition happens in the employees who doesnt got promotion for last 10 years.

# # w)Years in current role vs Attrition

# In[122]:


plt.figure(figsize=[10,10])
sns.countplot(x='YearsInCurrentRole',hue='Attrition',data=HR_data)
plt.show()


# In[123]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.YearsInCurrentRole],margins=True,normalize='index')


# # x)Years at company vs Attrition

# In[124]:


plt.figure(figsize=[10,10])
sns.countplot(x='YearsAtCompany',hue='Attrition',data=HR_data)
plt.show()


# In[125]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.YearsAtCompany],margins=True,normalize='index')


# # y)Total working years vs Attrition

# In[126]:


plt.figure(figsize=[10,10])
sns.countplot(x='TotalWorkingYears',hue='Attrition',data=HR_data)
plt.show()


# In[127]:


pd.crosstab(columns=[HR_data.Attrition],index=[HR_data.TotalWorkingYears],margins=True,normalize='index')


# # z)Employee Source vs Attrition

# In[128]:


sns.factorplot(data=HR_data,kind='count',x='Attrition',col='Employee Source')


# # Inference
# The higest ratio of voluntary resignation is by employees who have  entered by referal(20%).

# # >II)Age vs Joblevel

# In[129]:


sns.factorplot(x = 'Age', y='JobLevel', kind = 'bar', data=HR_data, aspect = 3)


# In[130]:


pd.crosstab(columns=[HR_data.JobLevel],index=[HR_data.Age],margins=True,normalize='index')


# # Inference
# Younger age group employees (ie between 18 to 28)are in job level 1.

# # >III)Job level vs Monthly income

# In[131]:


sns.scatterplot(data=HR_data,x='JobLevel',y='MonthlyIncome')
plt.title("Job level vs Monthly income")  


# In[132]:


HR_data.groupby(['JobLevel'])['MonthlyIncome'].mean().to_frame()


# # Inference
# The employees with job level 5 have highest salary which make a sense that job level 5 is the highest position in the company.

# # >IV)Job level vs Total working years

# In[133]:


plt.figure(figsize=(10,5))
sns.regplot(data=HR_data,x='JobLevel',y='TotalWorkingYears')
plt.title("Job level vs Total working years")
plt.show()


# In[134]:


HR_data.groupby(['JobLevel'])['TotalWorkingYears'].mean().to_frame()


# # Inference
# The employees with greater working years are in job level 5.

# # >V)Age vs Monthly income

# In[135]:


plt.figure(figsize = (12,4))
sns.regplot(x= 'Age', y = 'MonthlyIncome' , data =HR_data,color='lightseagreen')
plt.show()


# In[136]:


HR_data.groupby(['Age'])['MonthlyIncome'].mean().to_frame()


# # Inference
# The monthly salary is low for younger age employees.

# # >VI)Performance rating vs Percentage salary hike

# In[137]:


plt.figure(figsize = (10,5))
sns.regplot(x= 'PerformanceRating', y = 'PercentSalaryHike' , data = HR_data)
plt.show()


# In[138]:


HR_data.groupby(['PerformanceRating'])['PercentSalaryHike'].mean().to_frame()


# # Inference
# There is a linear relationship between performance rating and percentage salary hike.
# The employees with performance rating 4 has 18% salary hike.

# # >VII)Age vs Education

# In[139]:


plt.figure(figsize = (10,5))
sns.regplot(x= 'Age',y = 'Education',data = HR_data)
plt.show()


# In[140]:


HR_data.groupby(['Education'])['Age'].mean().to_frame()


# # >VIII)Job level vs Years at company

# In[141]:


plt.figure(figsize=(10,5))
sns.regplot(data=HR_data,x='JobLevel',y='YearsAtCompany')
plt.title("Job level vs Years at company")
plt.show()


# In[142]:


HR_data.groupby(['JobLevel'])['YearsAtCompany'].mean().to_frame()


# # Inference
# The employees who are working in the company for more years are at job level 5. 

# # >IX)Total working years vs Monthly income

# In[143]:


sns.scatterplot(data=HR_data,x='TotalWorkingYears',y='MonthlyIncome')
plt.title("Total working years vs Monthly income") 


# # Inference
# The employees who are working for more years have higher salary.

# # >X)Job level vs Years since last promotion

# In[144]:


sns.scatterplot(data=HR_data,x='JobLevel',y='YearsSinceLastPromotion')
plt.title("Job level vs Years since last promotion") 


# In[145]:


HR_data.groupby(['JobLevel'])['YearsSinceLastPromotion'].mean().to_frame()


# # Inference
# The employees who are in job level 4 & 5 have higher number of promotion since last year.

# # >XI)Years since last promotion vs Monthly income

# In[146]:


sns.scatterplot(data=HR_data,x='YearsSinceLastPromotion',y='MonthlyIncome')
plt.title("Years since last promotion vs Monthly income") 


# In[147]:


HR_data.groupby(['YearsSinceLastPromotion'])['MonthlyIncome'].mean().to_frame()


# # Inference
# The employees getting more promotion have high salary.

# # 1.3)Multivariate Visualization

# # a)Gender & Monthly income vs Attrition

# In[148]:


plt.figure(figsize=(10,5))
sns.boxplot(x="Gender", y="MonthlyIncome", data=HR_data,hue='Attrition',palette='GnBu')


# In[149]:


HR_data.groupby(['Gender','Attrition'])['MonthlyIncome'].median().to_frame()


# In[150]:


HR_data.groupby(['Gender','Attrition'])['MonthlyIncome'].mean().to_frame()


# # Inference
# Monthly income rate of male and female for attrition is almost same .

# # b)Gender & Age vs Attrition

# In[151]:


plt.figure(figsize=(10,5))
sns.boxplot(x="Gender", y="Age", data=HR_data,hue='Attrition',palette='GnBu')


# In[152]:


HR_data.groupby(['Gender','Attrition'])['Age'].mean().to_frame()


# In[153]:


HR_data.groupby(['Gender','Attrition'])['Age'].median().to_frame()


# # Inference
# The attrition age for both the genders are equal.

# From these we can conclude that gender does not have much effect in employee attrition.

# # c)Department & Distance from home vs Attrition

# In[154]:


plt.figure(figsize=(10,5))
sns.barplot( x="Department", y='DistanceFromHome',data=HR_data,hue='Attrition',palette='GnBu')


# In[155]:


HR_data.groupby(['Department','Attrition'])['DistanceFromHome'].mean().to_frame()


# In[156]:


HR_data.groupby(['Department','Attrition'])['DistanceFromHome'].median().to_frame()


# # Inference
# The employees whoes distance from home is large is leaving the company.

# # d)Age & Monthly income vs Attrition

# In[157]:


plt.figure(figsize = (16,6))
sns.jointplot(x='Age',y='MonthlyIncome',data=HR_data,hue='Attrition')
plt.show()


# In[158]:


HR_data.groupby(['Attrition'])['Age'].mean().to_frame()


# In[159]:


HR_data.groupby(['Attrition'])['MonthlyIncome'].mean().to_frame()


# # Inference
# Younger age employees with lower salary is leaving the company.

# # e)Business travel & Monthly income vs Attrition

# In[160]:


plt.figure(figsize=(10,5))
sns.boxplot(x='BusinessTravel',y='MonthlyIncome',data=HR_data,hue='Attrition',palette='GnBu')


# In[161]:


HR_data.groupby(['BusinessTravel','Attrition'])['MonthlyIncome'].mean().to_frame()


# In[162]:


HR_data.groupby(['BusinessTravel','Attrition'])['MonthlyIncome'].median().to_frame()


# # Inference
# The employees who travel frequently with lower salary have chance of leaving the company.

# # f)Job stisfaction & Monthly income vs Attrition

# In[163]:


plt.subplots(figsize=(10,5))
sns.boxplot(x='JobSatisfaction',y='MonthlyIncome',data=HR_data,hue='Attrition',palette='GnBu')
plt.show()


# In[164]:


HR_data.groupby(['JobSatisfaction','Attrition'])['MonthlyIncome'].mean().to_frame()


# # Inference
# The employees with low job satisfaction & low payment have chance of leaving the company.

# # g)Years since last promotion & Monthly income vs Job level

# In[165]:


plt.figure(figsize = (16,6))
sns.jointplot(x='YearsSinceLastPromotion',y='MonthlyIncome',data=HR_data,hue='JobLevel')
plt.show()


# In[166]:


HR_data.groupby(['JobLevel'])['MonthlyIncome'].mean().to_frame()


# In[167]:


HR_data.groupby(['JobLevel'])['YearsSinceLastPromotion'].mean().to_frame()


# # Inference
# The employees with high job level & higher number of promotions have greater salary.

# # h)Education field & Performance rating vs Attrition

# In[168]:


plt.figure(figsize=(10,5))
sns.violinplot( x="EducationField", y='PerformanceRating',data=HR_data,hue='Attrition',palette='GnBu')


# In[169]:


HR_data.groupby(['EducationField','Attrition'])['PerformanceRating'].mean().to_frame()


# In[170]:


HR_data.groupby(['EducationField','Attrition'])['PerformanceRating'].median().to_frame()


# # Inference 
# The performance rating for the currently working & resigned employees in each education field is almost same.

# # 2)Data Pre-Processing

# The steps of preprocessing include checking for null value and filling it.We have checked it & a total of 1.5% of null values are present in our dataset.Now for filling the null values we need to check, if there is any outliers present in numerical columns.If there is outlier we need to fill the missing values by using median if not fill it with mean.For the categorical columns & discrete columns we use mode to fill missing values.

# # 2.1)Droping not useful column

# We notice that 'EmployeeCount', 'Over18' and 'StandardHours' have only one unique values. This features aren't useful for us, So we are going to drop those columns.

# In[171]:


HR_data.drop(['EmployeeCount','Over18','StandardHours'], axis="columns", inplace=True)


# In[172]:


HR_data.shape


# # 2.2)Feature Engineering

# Adding some columns to our data.

# In[173]:


HR_data['TotalSatisfaction_mean'] = (HR_data['RelationshipSatisfaction']  + HR_data['EnvironmentSatisfaction'] +HR_data['JobSatisfaction'] + HR_data['JobInvolvement'] + HR_data['WorkLifeBalance'])/5
HR_data['Stability'] = HR_data['YearsInCurrentRole'] / HR_data['YearsAtCompany']
HR_data['Income_YearsComp'] = HR_data['MonthlyIncome'] /HR_data['YearsAtCompany']
HR_data['Fidelity'] = (HR_data['NumCompaniesWorked']) / HR_data['TotalWorkingYears']


# In[174]:


HR_data.shape


# # 2.3)Feature Reduction

# From the correlation heat map we observed that the columns 'DailyRate','HourlyRate'& 'MonthlyRate' are barely correlated with anything.So we can drop these columns.And from the multivariate graph we can notice that 'Gender' doesnot have much influence in employeee attrition,so we can drop that column.

# In[175]:


HR_data.drop(['DailyRate','HourlyRate','MonthlyRate','Gender','Employee Source'], axis="columns", inplace=True)


# In[ ]:





# In[176]:


HR_data.shape


# In[177]:


HR_data.columns


# In[178]:


#outlier visualization 
HR_data.plot(kind='box',subplots=True,layout=(5,5),figsize=(15,15),title='Outlier Visualization')
plt.show()


# # 2.4)Null value Handling

# We have checked for the outliers and the continues columns:'Monthly income', 'NumCompaniesWorked', 'TotalWorkingYears','YearsAtCompany', 'Income_YearsComp' are having outliers, so we are filling null vaues in these columns using median.

# In[ ]:





# In[179]:


for i in ['MonthlyIncome','NumCompaniesWorked','TotalWorkingYears','YearsAtCompany','Fidelity', 'Income_YearsComp']:
 HR_data[i]=HR_data[i].fillna(HR_data[i].median())


# The other numerical columns which doesnot contain outliers are:'Age', 'DistanceFromHome' ,'Stability','PercentSalaryHike' can be filled using mean.

# In[180]:


for i in ['Age','DistanceFromHome','Stability','PercentSalaryHike']:
 HR_data[i]=HR_data[i].fillna(HR_data[i].mean()) 


# The categorical columns:'Attrition', 'BusinessTravel', 'Department', 'EducationField', 'JobRole','MaritalStatus', 'OverTime','EnvironmentSatisfaction', 'Employee Source' & the discrete columns:'PerformanceRating','StockOptionLevel','Education','JobLevel','WorkLifeBalance','RelationshipSatisfaction','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','TotalSatisfaction_mean','YearsWithCurrManager','YearsSinceLastPromotion','YearsInCurrentRole','TrainngTimesLastYear','NumCompaniesWorked' can filled by mode.

# In[181]:


for i in ['BusinessTravel','Department','EnvironmentSatisfaction','TotalSatisfaction_mean','EducationField','JobRole','MaritalStatus','OverTime','Attrition','PerformanceRating','StockOptionLevel','Education','JobLevel','WorkLifeBalance','RelationshipSatisfaction','JobInvolvement','JobSatisfaction','YearsWithCurrManager','YearsSinceLastPromotion','YearsInCurrentRole','TrainingTimesLastYear']:
 HR_data[i]=HR_data[i].fillna(HR_data[i].mode()[0])


# In[182]:


HR_data.isnull().sum()


# Hence the null values are filled.

# # 2.5)Outlier Handling

# In[183]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


# In[184]:


lr,ur=remove_outlier(HR_data["MonthlyIncome"])
HR_data["MonthlyIncome"]=np.where(HR_data["MonthlyIncome"]>ur,ur,HR_data["MonthlyIncome"])
HR_data["MonthlyIncome"]=np.where(HR_data["MonthlyIncome"]<lr,lr,HR_data["MonthlyIncome"])


# In[185]:


plt.boxplot(HR_data['MonthlyIncome'])
plt.title('Box plot of Monthly income')


# In[186]:


lr,ur=remove_outlier(HR_data["TotalWorkingYears"])
HR_data["TotalWorkingYears"]=np.where(HR_data["TotalWorkingYears"]>ur,ur,HR_data["TotalWorkingYears"])
HR_data["TotalWorkingYears"]=np.where(HR_data["TotalWorkingYears"]<lr,lr,HR_data["TotalWorkingYears"])


# In[187]:


plt.boxplot(HR_data['TotalWorkingYears'])
plt.title('Box plot of Total working years')


# In[188]:


lr,ur=remove_outlier(HR_data["YearsAtCompany"])
HR_data["YearsAtCompany"]=np.where(HR_data["YearsAtCompany"]>ur,ur,HR_data["YearsAtCompany"])
HR_data["YearsAtCompany"]=np.where(HR_data["YearsAtCompany"]<lr,lr,HR_data["YearsAtCompany"])


# In[189]:


plt.boxplot(HR_data['YearsAtCompany'])
plt.title('Box plot of years at company')


# In[190]:


lr,ur=remove_outlier(HR_data["Income_YearsComp"])
HR_data["Income_YearsComp"]=np.where(HR_data["Income_YearsComp"]>ur,ur,HR_data["Income_YearsComp"])
HR_data["Income_YearsComp"]=np.where(HR_data["Income_YearsComp"]<lr,lr,HR_data["Income_YearsComp"])


# In[191]:


plt.boxplot(HR_data['Income_YearsComp'])
plt.title('Box plot of income_yearscompany')


# In[192]:


lr,ur=remove_outlier(HR_data["Fidelity"])
HR_data["Fidelity"]=np.where(HR_data["Fidelity"]>ur,ur,HR_data["Fidelity"])
HR_data["Fidelity"]=np.where(HR_data["Fidelity"]<lr,lr,HR_data["Fidelity"])


# In[193]:


plt.boxplot(HR_data['Fidelity'])
plt.title('Box plot of Fidelity')


# Hence the outliers are removed.

# Before feeding our data into a ML model we first need to prepare the data. This includes encoding all the categorical features, as the model expects the features to be in numerical form. Also for better performance we will do the feature scaling ie bringing all the features onto the same scale by using the StandardScaler provided in the scikit library.

# In[194]:


#Data frame
HR_data=pd.DataFrame(HR_data)


# In[195]:


#Dependent column
Target_col=pd.DataFrame(HR_data,columns=['Attrition'])


# In[196]:


Target_col


# In[197]:


#Categorical columns
Cat_cols=pd.DataFrame(HR_data,columns=["BusinessTravel","Department","EducationField","JobRole","MaritalStatus","OverTime"])
Cat_cols


# In[198]:


Cat_cols.columns


# In[199]:


# Continues numeric columns
Continues_numeric_col=pd.DataFrame(HR_data,columns=['Age','DistanceFromHome','MonthlyIncome','PercentSalaryHike','TotalWorkingYears','YearsAtCompany','Stability','Fidelity','Income_YearsComp'])
Continues_numeric_col


# In[200]:


#Discrete numeric columns
Discrete_numeric_col=pd.DataFrame(HR_data,columns=['Education','EnvironmentSatisfaction','JobInvolvement','JobLevel','JobSatisfaction','NumCompaniesWorked','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TrainingTimesLastYear','WorkLifeBalance','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','TotalSatisfaction_mean'])
Discrete_numeric_col


# In[201]:


Discrete_numeric_col.columns


# # 2.6)Encoding

# We use Ordinal encoder to encode categorical columns & Label encoder to encode target column.

# In[202]:


from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
o=OrdinalEncoder()
l=LabelEncoder()


# In[203]:


#Ordinal encoding
for i in Cat_cols.columns:
    if Cat_cols[i].dtypes=='O':
     Cat_cols[i]=o.fit_transform(Cat_cols[i].values.reshape(-1,1))


# In[204]:


Cat_cols.head()


# In[205]:


#Label encoding
Target_col['Attrition']=l.fit_transform(Target_col['Attrition'])


# In[206]:


Target_col.head()


# In[207]:


#Model Creation
x=pd.concat([Continues_numeric_col,Discrete_numeric_col,Cat_cols],axis=1)
y=Target_col


# In[208]:


x.head(2)


# In[209]:


y.head(2)


# In[210]:


#imbalanced
y.Attrition.value_counts()


# # 2.7)Balancing Data

# The data can be balanced by Over-sampling method or by Under-sampling method.
# Oversampling focuses on increasing minority class samples.
# We can also duplicate the examples to increase the minority class samples. Although it balances the data, it does not provide additional information to the classification model.
# Therefore synthesizing new examples using an appropriate technique is necessary. Here we use SMOTE to balance our data.
# SMOTE stands for Synthetic Minority Oversampling Technique.
# SMOTE selects the nearest examples in the feature space, then draws a line between them, and at a point along the line, it creates a new sample.

# In[211]:


from collections import Counter


# In[212]:



# In[213]:


from imblearn.over_sampling import SMOTE

smote_object = SMOTE()
print('Imbalance data:',Counter(y))
x_smote , y_smote = smote_object.fit_resample(x,y)
print('balance data:',Counter(y_smote))


# In[214]:


y_smote


# In[215]:


y_smote.value_counts()


# Hence we balance our data.

# In[216]:


x_smote


# In[217]:


x_smote.columns


# # 2.8)Standardization

# In[218]:


Continues_numeric_col=pd.DataFrame(x_smote,columns=['Age','DistanceFromHome','MonthlyIncome','PercentSalaryHike','TotalWorkingYears','YearsAtCompany','Stability','Fidelity','Income_YearsComp'])
Continues_numeric_col


# In[219]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled=scaler.fit_transform(Continues_numeric_col)
scaled=pd.DataFrame(scaled,columns=Continues_numeric_col.columns)

scaled.describe()


# Using StandardScaler we standardize continues numeric columns.

# In[220]:


Discrete_numeric_col=pd.DataFrame(x_smote,columns=['Education','EnvironmentSatisfaction','JobInvolvement','JobLevel','JobSatisfaction','NumCompaniesWorked','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TrainingTimesLastYear','WorkLifeBalance','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','TotalSatisfaction_mean'])
Discrete_numeric_col


# In[221]:


Cat_cols=pd.DataFrame(x_smote,columns=["BusinessTravel","Department","EducationField","JobRole","MaritalStatus","OverTime"])
Cat_cols


# In[222]:


x_smote=pd.concat([scaled,Discrete_numeric_col,Cat_cols],axis=1)


# In[223]:


x_smote.head()


# Here we joined the standardized continues numeric column(scaled),discrete numeric column and categorical columns.

# In[224]:


y_smote.head()


# # 3)Modelling

# In[225]:


#Splitting the data into train & test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x_smote,y_smote,random_state=42,test_size=0.2)


# In[226]:


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[227]:


print(y_train.value_counts(),'\n', y_test.value_counts())


# # 3.1)Logistic Regression

# In[228]:


from sklearn.linear_model import LogisticRegression
logit_model =LogisticRegression()
logit_model=LogisticRegression().fit(x_train,y_train)
y_pred_lr =logit_model.predict(x_test)


# In[229]:


from sklearn.metrics import classification_report,confusion_matrix
print("Test Accuracy of Logistic Regression classifier: {}%".format(round(logit_model.score(x_test, y_test)*100, 2)))


# In[230]:


print("Logistic Regression Classifier report: \n\n", classification_report(y_test, y_pred_lr))


# In[231]:


cm = confusion_matrix(y_test, y_pred_lr)
x_axis_labels = ["0", "1"]
y_axis_labels = ["0","1"]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Logistic Regression Classifier')
plt.show()


# # 3.2)K Nearest Neighbourhood

# In[232]:


from sklearn.neighbors import KNeighborsClassifier


# In[233]:


from sklearn.metrics import confusion_matrix,accuracy_score
acc_values = []
neighbors =np.arange(3,15)
for k in neighbors:
    knn =KNeighborsClassifier(n_neighbors=k,metric='minkowski')
    knn.fit(x_train,y_train)
    y_pred =knn.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    acc_values.append(acc)


# In[234]:


acc_values


# In[235]:


#checking which k value is having more accuracy value 
plt.plot(neighbors,acc_values,'o-')
plt.xlabel('k value')
plt.ylabel('accuracy')


# In[236]:


knn = KNeighborsClassifier(leaf_size=1,n_neighbors=3,metric="minkowski",p=1)
knn.fit(x_train,y_train)
y_pred_knn =knn.predict(x_test)
print("Accuracy is :",round(accuracy_score(y_test,y_pred)*100,2))


# Classification report of KNN Classifier

# In[237]:


y_pred_knn = knn.predict(x_test)
print("KNN Classifier report: \n\n", classification_report(y_test, y_pred_knn))


# In[238]:


cm = confusion_matrix(y_test, y_pred_knn)
x_axis_labels = ["0","1"]
y_axis_labels = ["0", "1"]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for KNN Classifier')
plt.show()


# # 3.3)Suport Vector Machine

# # a)Linear SVM

# In[239]:


from sklearn.svm import SVC


# In[240]:


svm_linear = SVC(kernel='linear')
svm_linear.fit(x_train,y_train)


# In[241]:


print("Test Accuracy of Linear svm Classifier: {}%".format(round(svm_linear.score(x_test, y_test)*100, 2)))


# In[242]:


y_pred_ls =svm_linear.predict(x_test)
print("Linear svm  Classifier report: \n\n", classification_report(y_test, y_pred_ls))


# In[ ]:





# In[243]:


cm = confusion_matrix(y_test, y_pred_ls)
x_axis_labels = ["0", "1"]
y_axis_labels = ["0","1"]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Linear svm  Classifier')
plt.show()


# # b)Polynomial SVM

# In[244]:


from sklearn.svm import SVC
svm_poly = SVC(kernel='poly',degree=3)
svm_poly.fit(x_train,y_train)


# In[245]:


print("Test Accuracy of Polynomial SVM Classifier: {}%".format(round(svm_poly.score(x_test, y_test)*100, 2)))


# In[246]:


y_pred_ps =svm_poly.predict(x_test)
print("Polynomial SVM Classifier report: \n\n", classification_report(y_test, y_pred_ps))


# In[247]:


cm = confusion_matrix(y_test, y_pred_ps)
x_axis_labels = ["0", "1"]
y_axis_labels = ["0", "1"]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Polynomial SVM Classifier')
plt.show()


# # c)Radial SVM

# In[248]:


from sklearn.svm import SVC


# In[249]:


svm_radial = SVC(kernel='rbf')
svm_radial.fit(x_train,y_train)


# In[250]:


print("Test Accuracy of Radial svm Classifier: {}%".format(round(svm_radial.score(x_test, y_test)*100, 2)))


# In[251]:


y_pred_rs =svm_radial.predict(x_test)
print("Radial SVM Classifier report: \n\n", classification_report(y_test, y_pred_rs))


# In[252]:


cm = confusion_matrix(y_test, y_pred_rs)
x_axis_labels = ["0", "1"]
y_axis_labels = ["0", "1"]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Radial SVM Classifier')
plt.show()


# # 3.4)Decision Tree Classifier

# In[253]:


from sklearn.tree import DecisionTreeClassifier
dt_model =  DecisionTreeClassifier()
dt_model.fit(x_train,y_train)


# In[254]:


print("Test Accuracy of Decision tree Classifier: {}%".format(round(dt_model.score(x_test, y_test)*100, 2)))


# In[255]:


y_pred_dt =dt_model.predict(x_test)
print(" Decision Tree Classifier report: \n\n", classification_report(y_test, y_pred_dt))


# In[256]:


cm = confusion_matrix(y_test, y_pred_dt)
x_axis_labels = ["0", "1"]
y_axis_labels = ["0", "1"]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Decision Tree Classifier')
plt.show()


# In[257]:


pd.DataFrame(index = x_smote.columns, data = dt_model.feature_importances_, 
             columns=['Feature Importance']).sort_values('Feature Importance', ascending = False)


# # 3.5)Random Forest Classifier

# In[258]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)


# In[259]:


print("Test Accuracy of Random Forest Classifier: {}%".format(round(rfc.score(x_test, y_test)*100, 2)))


# In[260]:


y_pred_rf = rfc.predict(x_test)
print("Random Forest Classifier report: \n\n", classification_report(y_test, y_pred_rf))


# In[261]:


cm = confusion_matrix(y_test, y_pred_rf)
x_axis_labels = ["0", "1"]
y_axis_labels = ["0", "1"]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Random Forest Classifier')
plt.show()


# In[262]:


pd.DataFrame(index = x_smote.columns, data = rfc.feature_importances_, 
             columns=['Feature Importance']).sort_values('Feature Importance', ascending = False)


# # 3.6)Gradient Boosting Classifier

# In[263]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)


# In[264]:


print("Test Accuracy of Gradient Boosting Classifier: {}%".format(round(gbc.score(x_test, y_test)*100, 2)))


# In[265]:


y_pred_gbc = gbc.predict(x_test)
print("Gradient Boosting Classifier report: \n\n", classification_report(y_test, y_pred_gbc))


# In[266]:


cm = confusion_matrix(y_test, y_pred_gbc)
x_axis_labels = ["0", "1"]
y_axis_labels = ["0", "1"]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Gradient Boosting Classifier')
plt.show()


# # 3.7)Extra Gradient Boosting Classifier

# In[267]:


import xgboost as xgb


# In[268]:


from xgboost import XGBClassifier


# In[269]:


xgbc = XGBClassifier()
xgbc.fit(x_train, y_train)


# In[270]:


print("Test Accuracy of Extra gradiant boosting Classifier: {}%".format(round(xgbc.score(x_test, y_test)*100, 2)))


# In[271]:


y_pred_eg = xgbc.predict(x_test)
print("Extra gradiant boosting Classifier report: \n\n", classification_report(y_test, y_pred_eg))


# In[272]:


cm = confusion_matrix(y_test, y_pred_eg)
x_axis_labels = ["0", "1","2"]
y_axis_labels = ["0", "1","2"]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Extra gradiant boosting Classifier')
plt.show()


# # 3.8)Naive Bayes Classifier

# In[273]:


from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)

print("Test Accuracy of Naive Bayes Classification: {}%".format(round(nb.score(x_test, y_test)*100, 2)))


# In[274]:


y_pred_nb = nb.predict(x_test)
print("Naive Bayes Classifier report: \n\n", classification_report(y_test, y_pred_nb))


# In[275]:


cm = confusion_matrix(y_test, y_pred_eg)
x_axis_labels = ["0", "1"]
y_axis_labels = ["0","1",]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Naive Bayes Classifier')
plt.show()


# # Random forest regressor

# In[276]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[277]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 500, stop = 1000, num = 3)]
# Number of features to consider at every split
max_features = ['sqrt']
max_depth = []
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1]
# Method of selecting samples for training each tree
#bootstrap = [True, False]
bootstrap = [True]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# In[290]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train,y_train)


# In[291]:


rf_random.best_params_


# In[292]:


rf = RandomForestClassifier(random_state = 42,n_estimators= 750,max_features='sqrt')


# In[293]:


model = rf.fit(x_train,y_train)
predictions = model.predict(x_test)
print("Test Accuracy of RandomForest regressor : {}%".format(round(rf.score(x_test, y_test)*100, 2)))


# In[294]:



print("Random Forest regressor report: \n\n", classification_report(y_test,predictions))


# In[295]:


cm = confusion_matrix(y_test, predictions)
x_axis_labels = ["0", "1"]
y_axis_labels = ["0","1",]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Random forest regressor')
plt.show()


# In[296]:


pd.DataFrame(index = x_smote.columns, data = rf.feature_importances_, 
             columns=['Feature Importance']).sort_values('Feature Importance', ascending = False)


# In[297]:


models = pd.DataFrame({
    'Model' : ["K Nearest Neighbourhood","Linear SVM","Polynomial SVM","Radial SVM","Decision Tree Classifier", 'Random Forest Classifier', 
               "Gradient Boosting","Extra Gradient Boostig Classifier","Naive Bayes Classifier","Logistic Regression classifier"],

    'Score' : [accuracy_score(y_test,y_pred)*100,svm_linear.score(x_test, y_test)*100,svm_poly.score(x_test, y_test)*100,svm_radial.score(x_test, y_test)*100,dt_model.score(x_test, y_test)*100,rfc.score(x_test,y_test)*100, 
               gbc.score(x_test,y_test)*100,xgbc.score(x_test,y_test)*100,nb.score(x_test, y_test)*100,logit_model.score(x_test, y_test)*100]
    })

models.sort_values(by = 'Score', ascending = False)


# In[298]:


import plotly.express as px
models = models.sort_values(by=['Score'])
px.bar(data_frame = models, x = 'Score', y = 'Model', orientation='h', color = 'Score', template = 'plotly_dark', title = 'Models Comparison')


# In[299]:


import pickle


# In[300]:


pickle.dump(rfc, open('model.pkl', 'wb'))#Save the model


# In[301]:


pickled_model = pickle.load(open('model.pkl', 'rb'))#load the model
pickled_model.predict(x_test)


# In[ ]:




