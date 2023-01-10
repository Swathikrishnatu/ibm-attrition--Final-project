from flask import Flask, render_template, request
#import requests
import pickle
#import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':

        #1Age
        Age=float(request.form['Age'])
        
        #2BusinessTravel
        BusinessTravel=int(request.form['BusinessTravel'])
        
        #3Department
        Department=int(request.form['Department'])
        
        #4distanceFromHome
        DistanceFromHome=float(request.form['DistanceFromHome'])
        #5Education
        Education=float(request.form['Education'])
        
        #6EducationField
        EducationField=int(request.form['EducationField'])
                
        
                
        #7EnvironmentSatisfaction
        EnvironmentSatisfaction=float(request.form['EnvironmentSatisfaction'])

        #8JobInvolvement
        JobInvolvement=float(request.form['JobInvolvement'])

        #9JobLevel
        JobLevel=float(request.form['JobLevel'])
        
           
        

        #10JobRole
        JobRole=int(request.form['JobRole'])

        #11JobSatisfaction
        JobSatisfaction=float(request.form['JobSatisfaction'])

        #12MaritalStatus
        MaritalStatus=int(request.form['MaritalStatus'])

       
        #14MonthlyIncome
        MonthlyIncome=float(request.form['MonthlyIncome'])

        #15NumCompaniesWorked
        NumCompaniesWorked=float(request.form['NumCompaniesWorked'])

        

        #16OverTime
        OverTime=int(request.form['OverTime'])

        #17PercentSalaryHike
        PercentSalaryHike=float(request.form['PercentSalaryHike'])
       
        #18PerformanceRating
        PerformanceRating=float(request.form['PerformanceRating'])
        
        #19RelationshipSatisfaction
        RelationshipSatisfaction=float(request.form['RelationshipSatisfaction'])

       #20StockOptionLevel
        StockOptionLevel=float(request.form['StockOptionLevel'])

        #21TotalWorkingYears
        TotalWorkingYears=float(request.form['TotalWorkingYears'])

        #22TrainingTimesLastYear
        TrainingTimesLastYear=float(request.form['TrainingTimesLastYear'])

        #23YearsSinceLastPromotion
        WorkLifeBalance=float(request.form['WorkLifeBalance'])

        #24YearsAtCompany
        YearsAtCompany=float(request.form['YearsAtCompany'])

        #25YearsInCurrentRole
        YearsInCurrentRole=float(request.form['YearsInCurrentRole'])

        #26YearsSinceLastPromotion
        YearsSinceLastPromotion=float(request.form['YearsSinceLastPromotion'])

        #27YearsWithCurrManager
        YearsWithCurrManager=float(request.form['YearsWithCurrManager'])

        

        #28Stability
        Stability=float(request.form['Stability'])

        #29Fidelity
        Fidelity=float(request.form['Fidelity'])


        #30Income_YearsComp
        Income_YearsComp=float(request.form['Income_YearsComp'])
        
       
        
        #31Total_Satisfaction
        TotalSatisfaction_mean=float(request.form['Total_Satisfaction'])

        prediction=model.predict([[Age,
                                    Department,
                                    DistanceFromHome,
                                    Education,
                                    EducationField,
                                    EnvironmentSatisfaction,
                                    JobInvolvement,
                                    JobLevel,
                                    JobRole,
                                    BusinessTravel,
                                    JobSatisfaction,
                                    MaritalStatus,
                                    MonthlyIncome,
                                    NumCompaniesWorked,
                                    OverTime,
                                    PercentSalaryHike,
                                    PerformanceRating,
                                    RelationshipSatisfaction,
                                    StockOptionLevel,
                                    TotalWorkingYears,
                                    TrainingTimesLastYear,
                                    WorkLifeBalance,
                                    YearsAtCompany,
                                    YearsInCurrentRole,
                                    YearsSinceLastPromotion,
                                    YearsWithCurrManager,
                                     Stability,
                                     Fidelity,
                                    Income_YearsComp,
                                    TotalSatisfaction_mean
                                   ]])

        #output=prediction
        if(prediction==0):
            return render_template('result.html',prediction_text="This employee likely to leave Company")
        elif(prediction==1):
            return render_template('result.html',prediction_text="This employee will not leave the Company")
        else:
            return render_template('result.html',prediction_text=" data not present")

if __name__=="__main__":
  app.run()

   