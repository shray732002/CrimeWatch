# CrimeWatch

## Problem Statement :
Controlling crime rates has constantly challenged governments and law enforcement agencies, as these acts are committed in an unprecedented manner which requires laborious efforts and skills.To provide a better insight into such acts, we need preventive methods such as predictive policing using the CCTNS dataset containing information on the crimes committed over a period.This enables the Assam police to better detect and prevent future crimes from occurring and, to maintain peace and order among the public.

## Dataset :
The dataset is taken from the official Boston Crime Department. You can download the dataset from [here](https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system)

## Approach :
Applying machine learing tasks like Data Exploration, Data Cleaning, Feature Engineering, Model Building and model testing to build a solution with the goal of improving public safety and crime prevention efforts.

- **Data Exploration :** Exploring the dataset using pandas, numpy, matplotlib, plotly and seaborn.
- **Exploratory Data Analysis :** Plotted different graphs to get more insights about dependent and independent variables/features.
- **Feature Engineering :** Numerical features scaled down and Categorical features encoded.
- **Model Building :** In this step, first dataset Splitting is done. After that model is trained on different Machine Learning Algorithms such as:
    1) Xgboost
    2) SARIMAX (Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors)
    3) Random Forest
    4) KNN
    5) LightGBM
    
- **Model Selection :** Tested all the models to check the Root mean squared error(RMSE) and Accuracy.
- **Pickle File** : Selected model as per best RMSE score & Accuracy and created pickle file using pickle library.
- **Webpage & Deployment :** Created a web application that takes all the necessary inputs from the user & shows the output.

&nbsp;  

## Installation 

To run our code:

```git clone https://github.com/mayank-kr/CrimeWatch.git```

```cd CrimeWatch```

Ensure that flask is downloaded and running!

```pip install -r requirements.txt```

```python main.py```


## Web Inerface :
![alt text](static/SS1.jpeg)


![alt text](static/SS2.jpeg)


## Libraries used :
    1) Pandas
    2) Numpy
    3) Matplotlib, Seaborn, Plotly
    4) Scikit-Learn
    5) Flask
    6) Folium
    7) Statsmodels

## Technical Aspects :
    1) Python 
    2) Front-end : HTML, CSS
    3) Back-end : Flask
    4) Deployment Replit and Vercel

## Team Members :
    1) Aaryan Gupta
    2) Mayank Kumar
    3) Shray Kumar Singh
    4) K Sai Vamshi Nayak

## Authors and Acknowledgement

#### **Aaryan Gupta**  
[LinkedIn :necktie:](https://www.linkedin.com/in/aaryan-gupta-a881661b8/)  
[Github :floppy_disk:](https://github.com/Aaryan0424)    

#### **Mayank Kumar**  
[LinkedIn :necktie:](https://www.linkedin.com/in/mayank-kumar2002/)     
[Github :floppy_disk:](https://github.com/mayank-kr) 

#### **Shray Kumar Singh**  
[LinkedIn :necktie:](https://www.linkedin.com/in/shray-singh-49a965241/)   
[Github :floppy_disk:](https://github.com/shray732002)

#### **K Sai Vamshi Nayak**  
[LinkedIn :necktie:](https://www.linkedin.com/in/vamshi5421/)    
[Github :floppy_disk:](https://github.com/vamshi5421)

&nbsp;  

## Demo Link

A demo video can be found [here](https://youtu.be/g0OXXvPbjBM)

## License
[MIT](https://choosealicense.com/licenses/mit/)
