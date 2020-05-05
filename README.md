# Predicting hospital bed shortage due to COVID-19 

This project builds the <a href="https://covid19-hospital.herokuapp.com/ ">COVID-19 Hospital app</a>, which predicts the number of COVID-19 patients that will require hospital and ICU beds in the US.

## How does the model work?

The app uses COVID-19 data from the <a href="https://github.com/CSSEGISandData/COVID-19">Johns Hopkins CSSE repository</a> to fit a SIR model. It also uses state-level demographic information and hospital capacity to predict the number of COVID-19 patients that will require hospital and ICU beds. 

For details regarding the model, please read this <a href="https://medium.com/@irishryoon/predicting-hospital-bed-shortage-in-the-us-due-to-covid-19-2d860ecdaba2">blog on Medium.</a>

## Where are the data?
COVID-19 cases reported in the Johns Hopkins CSSE repository have been stored in the "data" directory as pickle files.   

State-level demographic and population information have been scraped from various websites and stored in the "data" directory as well. 

*NOTE: The app uses the Johns Hopkins data that was available on March 26, 2020*

## Where can I see the code?
To play around with building the model and making predictions, please see the Google Colab notebook in "COVID19.ipynb".  

To view the code for the <a href="https://covid19-hospital.herokuapp.com/ ">COVID-19 Hospital web app</a>, please see "app.py" 
