#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:57:13 2020

@author: irisyoon
irisyoon.org
irishryoon@gmail.com
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from collections import OrderedDict
import datetime
import matplotlib.pyplot as plt 
from matplotlib import style
plt.style.use('ggplot')

# using this package https://symfit.readthedocs.io/en/latest/fitting_types.html#ode-fitting
from symfit import Parameter, variables, Fit, D, ODEModel


# =============================== FUNCTIONS =============================== 

def get_state_df(data, state, state_name):
  # create dataframe for selected state
  """
  --- input ---
  data: (dic) 
  state: (str) state abbreviation
  state_name: (str) state name given by abbreviation[state]

  --- output ---
  state_df: (DataFrame)
  """

  # NOTE: in "data"
  #   On some days, the "Province/State" is reported as "County, State"
  #   On some days, the Province/State" is reported as "State"

  keys = list(data.keys())
  keys.sort()
  last = keys[-1]
  columns = data[last].columns

  state_data = pd.DataFrame(columns = columns)  

  # get state data from different days
  for date in keys:
    df = data[date]
    day_data = df[(df['Country/Region'] == 'US') & 
              ((df['Province/State'] ==  state_name) | (df['Province/State'].str.contains(state))
              )]
    state_data = state_data.append(day_data, ignore_index = True)

  # sort data, from oldest (top) to newest (bottom) 
  state_data = state_data.sort_values(by = ["report date"])

  # Group by: make sure that all reported under "County, State" are aggregated to one row with "State"
  state_df = pd.DataFrame(columns = state_data.columns)

  state_df["Confirmed"] = state_data.groupby(["report date"])["Confirmed"].sum().values
  state_df["Deaths"] = state_data.groupby(["report date"])["Deaths"].sum().values
  state_df["Recovered"] = state_data.groupby(["report date"])["Recovered"].sum().values
  state_df["Province/State"] = state_name
  state_df["Country/Region"] = "US"
  state_df["report date"] = state_data.groupby(["report date"])["Confirmed"].sum().keys()
  
  # Simulate the number of recovered people on each day
  # adjust "Confirmed" to keep track of actively ill people on given day. 
  # "Infected" is the number  of people that are ill on any given day. 
  n_days = len(state_df["Confirmed"])
  # remove number of deaths from confirmed cases 
  infected = [state_df["Confirmed"][0] - state_df["Deaths"][0]]
  recovered_sim = [0]

  for i in range(1, n_days):
    R_day = int(sum(infected[:i]) * 0.02)
    recovered_sim.append(R_day)
    infected.append(state_df.at[i,"Confirmed"] - R_day)

  state_df["Infected"] = infected
  state_df["death_or_recovered"] = recovered_sim

  return state_df


def check_data_sufficiency(state_df):
  # Check if the selected state has enough data for the model. 
  # Check if the state has 5 consecutive days satisfying the following:
  #      (a) The number of confirmed cases are all above 5
  #      (b) The number of confirmed cases are increasing every day
  """
  --- input ---
  state_df: (DataFrame) output of function "get_state_df"

  --- output ---
  sufficiency = (bool) True if there is enough data
  first_day = (str if sufficiency == True, None if sufficiency == False) 
              (str) The earliest day of the possible 5 consecutive days
  last_day = (str if sufficiency == True, None if sufficiency == False) 
              (str) The last available day of data 
  """

  infected = state_df["Infected"].values
  dates = state_df["report date"].values

  first_day_found = False
  for idx, I in enumerate(infected):
    if I >= 5:
      # check if the number of infected people increases for 5 consecutive days 
      next_days = infected[idx:idx+6]

      # check if the numbers are increasing:
      if np.all(np.diff(next_days )) > 0:
        first_day_found = True
        break
      
  if first_day_found == False:
    print("Insufficient data for state ", state)

  sufficient = first_day_found
  first_day = dates[idx]
  last_day = dates[-1]

  return sufficient, first_day, last_day 


def get_susceptible(state):
    state_population = US_dem[US_dem["GEO"] == state_name]["age999"].values[0]

    #st.sidebar.markdown("Select percentage of population that will eventually contract COVID-19")
    S_percentage = st.sidebar.slider('Select percentage of population that will eventually contract COVID-19. (As a reference, about 0.12% of the population tested positive in Hubei, China)',
                             min_value = 0.0,
                             max_value = 5.0,
                             step = 0.01)
    n_susceptible = S_percentage * state_population * 0.01                         
    st.write( "*", S_percentage, '% of population are susceptible.  \n * Population of ', state_name, " is ", int(state_population), '  \n * Number of susceptible population in ', state_name, " is ", int(n_susceptible) )

    return n_susceptible 


@st.cache
def fit_model(state, state_df, first_day, last_day, n_susceptible, n_days):
   # Take state data to fit the SIR model
   # Use the fitted model to make predictions for the next "n_days"

   """
   --- input ---
   state: (str) selected state. 
   state_df: (dataframe) of state data. output from function "get_state_df"
   first_day: (str) first day of data to use for model fitting. output of function "check_data_sufficiency"
   last_day: (str) last day of data to use for model fitting. output of function "check_data_sufficiency"
            NOTE: only data from "first_day" to "last_day" will be used in the parameter tuning process
   n_susceptible: (int) number of susceptible population 
   n_days: (int) number of days to make predictions. 
                  day 0 coincides with the "first_day" 
   --- output ---
   prediction: (dict) of prediction data
   """
    
   ### get data in correct format
   data_duration = (datetime.datetime.strptime(last_day, "%m-%d-%Y") - datetime.datetime.strptime(first_day, "%m-%d-%Y")).days + 1 
   data_time = np.array(range(0,data_duration)).astype(int) # time 0 corresponds to the first day appearing in dataframe
   first_day_idx = state_df[state_df["report date"] == first_day].index[0]
   last_day_idx = state_df[state_df["report date"] == last_day].index[0]

   data_I= np.array(state_df.loc[first_day_idx: last_day_idx, "Infected"])
   data_R = np.array(state_df.loc[first_day_idx:last_day_idx, "death_or_recovered"])
   data_S = [n_susceptible - x - data_R[idx] for idx, x in enumerate(data_I)]

   ### define variables and parameter for ODE model
   S, I, R, t = variables('S, I, R, t')

   ### find sensible initial parameter values
   dIdt = np.diff(data_I)
   dRdt = np.diff(data_R)
   gamma_0 = 0.02
   #gamma_0 = np.mean([x/infected_adjusted [idx] for idx, x in enumerate(dRdt)])
   beta_0 = np.mean([(x + gamma_0 * data_I[idx] ) * n_susceptible / (data_S[idx] * data_I[idx]) for idx, x in enumerate(dIdt)])

   beta = Parameter('beta', beta_0)
   #gamma = Parameter('gamma', gamma_0)
   gamma = Parameter("gamma", gamma_0)

   ### define ODE equations 
   model_dict = {
       D(S, t): - beta * S * I / n_susceptible,
       D(I, t): beta * S* I / n_susceptible - gamma * I,
       D(R, t): gamma * I
   }

   ### initial values
   I0 = data_I[0]
   S0 = n_susceptible - I0
   R0 = data_R[0]

   ### define the model
   model = ODEModel(model_dict, initial = { t : 0, S: S0, I: I0, R: R0})

   ### fit model parameters
   #fit = Fit(model, t = data_time, I = data_infected, S = None, R = data_recovered)
   fit = Fit(model, t = data_time, I = data_I, S = None, R = data_R )
   fit_result = fit.execute()

   ### make sure the parameters make sense
   # if the parameters don't make sense (<=0), just use the naive guess above
   params = fit_result.params
   if params["beta"] <= 0 or params["gamma"] <=0 :
     params = OrderedDict()
     params["beta"] = beta_0
     params["gamma"] = gamma_0
   if params["gamma"] >= 0.1:
     params["beta"] = beta_0
     params["gamma"] = gamma_0

   ### get predictions
   tvec = np.linspace(0, n_days-1, n_days)
   outcome = model(t=tvec, **params)
   I_pred = outcome.I
   S_pred = outcome.S
   R_pred = outcome.R 

   ### save predictions
   prediction = {}
   prediction["state"] = state
   prediction["first_day"] = first_day
   prediction["last_day"] = last_day
   prediction["num_days"] = n_days
   prediction["susceptible"] = n_susceptible
   prediction["infected_pred"] = I_pred
   prediction["susceptible_pred"] = S_pred
   prediction["recovered_pred"] = R_pred
   prediction["parameters"] = params

   return prediction

def predict_by_age(age_statistic, age_prediction, column_prefix):

  # Given a prediction of number of infected people by age group
  # and statistic on hospital admission and ICU admission by age group,
  # predict the number of hospital admission and ICU admission 

  """
  --- input ---
  age_statistic: (array) of statistic on either
              (1) hospital admission among infected people by age group, or
              (2) ICU admission among infected people by age group
  age_prediction: (DataFrame) prediction of number of infected people by age group
                  Must have the columns ["0-19","20-44","45-54","55-64","65-74","75-84","85+"]
  
  column_prefix: (str) prefix to column

  --- output ---
  df      :(DataFrame) of prediction by age group and prediction total
  """
  
  age_cols = ["0-19","20-44","45-54","55-64","65-74","75-84","85+"] 
  n_rows = age_prediction.shape[0]

  # Make prediction in each age group
  m = np.repeat(age_statistic, [n_rows], axis = 0)
  pred = np.multiply(age_prediction.values, m) 
  
  columns = [column_prefix + x for x in age_cols]
  df = pd.DataFrame(pred, columns = columns)
  
  # Make total prediction
  df[column_prefix + "_total"] = df.sum(axis = 1)

  return df

def predict_hospitalization_ICU(region_dem, P):

  """
  --- input ---
  region_dem: (DataFrame) of demographic information of particular state or county.
          Must have columns ["per0-19","per20-44", "per45-54", "per55-64","per65-74", "per75-84", "per85plus"]
          that indicate the percentage of particular age group  
  P: (dict) prediction outcome of function "train_model".
            Must have the key "infected_pred" and "num_days"



  --- output ---
  P_age: (dict) prediction of number of hospital and ICU admissions 

  """


  ### Statistic for hospital admission by age group ###
  # 0. data from CDC: https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm?s_cid=mm6912e2_w#T1_down
  age_cols = ["0-19","20-44","45-54","55-64","65-74","75-84","85+"] 

  hospitalization_max = np.array([0.025, 0.208, 0.283, 0.301, 0.435, 0.587, 0.703]).reshape(1,-1)
  hospitalization_min = np.array([0.016, 0.143, 0.212, 0.205, 0.286, 0.305, 0.313]).reshape(1,-1)
  ICU_max = np.array([0.   , 0.042, 0.104, 0.112, 0.188, 0.31 , 0.29 ]).reshape(1,-1)
  ICU_min = np.array([0.   , 0.02 , 0.054, 0.047, 0.081, 0.105, 0.063]).reshape(1,-1)

  first_dt = datetime.datetime.strptime(P["first_day"], "%m-%d-%Y")

  # 1. create dataframe P_age 
  P_age = pd.DataFrame(columns = ["date", "total"] + age_cols)
  P_age["total"] = P["infected_pred"]
  n_rows = P["num_days"]
  P_age["date"] = [(first_dt + datetime.timedelta(days = x)).strftime("%m-%d-%Y") for x in range(0,n_rows)]

  # 2. Predict the number of infected people in each age group
  for idx in P_age.index:
    total = P_age.at[idx,"total"]
    P_age.loc[idx, age_cols] = (total * region_dem[["per0-19","per20-44", "per45-54", "per55-64","per65-74", "per75-84", "per85plus"]]).values[0]

  # 3. Predict number of hospital admission using "hospitalization_min"
  hos_min_df = predict_by_age(hospitalization_min, P_age[age_cols], "hos_min")
  P_age = pd.concat([P_age, hos_min_df], axis = 1)

  # 4. Predict number of hospital amidssion using "hospitalization_max"
  hos_max_df = predict_by_age(hospitalization_max, P_age[age_cols], "hos_max")
  P_age = pd.concat([P_age, hos_max_df], axis = 1)

  # 5. Predict number of ICU amidssion using "ICU_min"
  ICU_min_df = predict_by_age(ICU_min, P_age[age_cols], "ICU_min")
  P_age = pd.concat([P_age, ICU_min_df], axis = 1)

  # 6. Predict number of ICU amidssion using "ICU_max"
  ICU_max_df = predict_by_age(ICU_max, P_age[age_cols], "ICU_max")
  P_age = pd.concat([P_age, ICU_max_df], axis = 1)

  return P_age

def hospital_info(hos_prediction, n_beds, region):
# predict when the hospital will run out of beds and how many extra beds hospitals need
  """
  --- input ---
  hos_prediciton: (DataFrame) output of function "predict_hospitalization_ICU"
  n_beds: (int) number of available beds
  region: (str) region name. ex) "WA" 
  
  --- output ---
  None
  """
  last_prediction_day = hos_prediction["date"].values[-1]

  # Conservative estimate 

  over = hos_prediction[hos_prediction["hos_min_total"] > n_beds]

  min_peak_idx = (hos_prediction["hos_min_total"]).idxmax()
  min_peak = hos_prediction.at[min_peak_idx, "date"]

  if over.empty == False:
    min_day = over["date"].values[0]
    last_day = over["date"].values[-1]

    if last_day < last_prediction_day:
      st.write("* According to a conservative estimate, hospitals in ", region, " will need more beds from ", 
          min_day, " to ", last_day)
    else:
      st.write("* According to a conservative estimate, hospitals in ", region, " will need more beds from ", 
          min_day, " to ", last_day," and beyond.")
  
    min_beds = int(hos_prediction.at[min_peak_idx, "hos_min_total"] - n_beds)
    st.write("* According to a conservative estimate, hospitals in ", region, " will need " + str(min_beds) + 
      " extra beds at its peak on " + min_peak)

  if over.empty == True:
    min_beds = int(hos_prediction.at[min_peak_idx, "hos_min_total"])
    st.write("* According to a conservative estimate, hospitals in ", region, " will need " + str(min_beds) +
          " beds dedicated to COVID-19 patients at its peak on " + min_peak)

  # Liberal estimate
  over = hos_prediction[hos_prediction["hos_max_total"] > n_beds]
  max_peak_idx = (hos_prediction["hos_max_total"]).idxmax()
  max_peak = hos_prediction.at[max_peak_idx, "date"]
  if over.empty == False:
    max_day = over["date"].values[0]
    last_day = over["date"].values[-1]
    
    if last_day < last_prediction_day:
      st.write("* According to a liberal estimate, hospitals in ", region, " will need more beds from ", 
          max_day, " to ", last_day)
    else:
      st.write("* According to a liberal estimate, hospitals in ", region, " will need more beds from ", 
          max_day, " to ", last_day, " and beyond.")


    max_beds = int(hos_prediction.at[max_peak_idx, "hos_max_total"] - n_beds)
    st.write("* According to a liberal estimate, hospitals in ", region, " will need " + str(max_beds) + 
        " extra beds at its peak on " + min_peak)

  if over.empty == True:
    max_beds = int(hos_prediction.at[max_peak_idx, "hos_max_total"])
    st.write("* According to a conservative estimate, hospitals in ", region, " will need " + str(max_beds) +
          " beds dedicated to COVID-19 patients at its peak on " + max_peak)

def plot_pred(ax, P, data_df, plot_title):
  # plot the prediction 
  """
  --- input ---
  ax: plot object
  P: (dict) output from function "train_model"
  data_df: (DataFrame) of COVID-data. 
          Must have columns "Infected", "Deaths", "Recovered", "Recovered_sim", "report date"
          There must be one row for each "report date"  
  plot_title: (str) title of string  

  --- output ---
  plot 
  """

  # load info from prediction 
  first_day = P["first_day"]
  last_day = P["last_day"]
  first_dt = datetime.datetime.strptime(P["first_day"], "%m-%d-%Y")
  last_dt = datetime.datetime.strptime(P["last_day"], "%m-%d-%Y")
  n_days = P["num_days"] 
  tvec = np.linspace(0, n_days-1, n_days)
  data_duration = (last_dt - first_dt).days + 1
  
  # plot predictions
  ax.plot(tvec, P["infected_pred"], label='prediction: infected', c = '#F95858' ) 
  ax.plot(tvec, P["recovered_pred"], label='prediction: dead or recovered ', c = '#6288E2') 

  # plot data
  data_df = data_df[(data_df["report date"] >= first_day) & (data_df["report date"] <= last_day)]
  ax.scatter(range(0, data_duration), 
             data_df["Infected"].values, 
             marker = 'o', 
             label = 'data: infected', 
             c = '#920808')

  ax.scatter(range(0, data_duration), 
             data_df["death_or_recovered"].values, 
             marker = 'o', 
             label = 'data: death or recovered', 
             c = '#244798')

  # set labels
  time_label = [(first_dt + datetime.timedelta(days = i)).date().strftime('%m-%d') for i in range(0,n_days,20)]
  ax.set_xticks(list(range(0,n_days,20))) 
  ax.set_xticklabels(time_label)
  ax.set_xlabel("date")
  ax.set_ylabel("number of people")
  ax.set_title(plot_title)

  return ax

def plot_hospital_capacity(ax, P, data_df, pred_infected, hospital_capacity, plot_title, scatter = True):

  """
  --- input --- 
  ax: plot object
  P: (dict) output of function "train_model"
  data_df: (DataFrame) of COVID-data. 
          Must have columns "Infected" and "report date"
          There must be one row for each "report date"  
  infected: (series) data of infected people 
  pred_infected: dataframe with columns ["total", "total_hos_min", "total_hos_max", "total_ICU_min", "total_ICU_max"]
  hospital_capacity: (int) number of hospital beds available.  
  title: (str) 
  """
  first_day = P["first_day"]
  n_days = pred_infected.shape[0]

  t = list(range(0, n_days))

  #ax.plot(t, pred_infected["total"], label = 'prediction: infected', c = '#F95858')
  ax.plot(t, pred_infected["hos_max_total"], label = 'liberal prediction: hospital beds', c = '#293EC9')
  ax.plot(t, pred_infected["hos_min_total"], label = 'conservative prediction: hospital beds', c = '#1FCED4')
  
  ax.plot(t, pred_infected["ICU_max_total"], label = 'liberal prediction: ICU beds', c= "#119202")
  ax.plot(t, pred_infected["ICU_min_total"], label ='conservative prediction: ICU beds', c = "#EBC400")

  ax.hlines(hospital_capacity, t[0], t[-1], label = "hospital capacity")


  # plot data
  first_day = P["first_day"]
  last_day = P["last_day"]
  first_dt = datetime.datetime.strptime(P["first_day"], "%m-%d-%Y")
  last_dt = datetime.datetime.strptime(P["last_day"], "%m-%d-%Y")
  data_duration = (last_dt - first_dt).days + 1 

  data_df = data_df[(data_df["report date"] >= first_day) & (data_df["report date"] <= last_day)]
  if scatter == True:
    ax.scatter(range(0, data_duration), data_df["Infected"].values, marker = 'o', 
              label = 'data:infected', c = '#920808')
  
  # 
  time_label = [(first_dt + datetime.timedelta(days = i)).date().strftime('%m-%d') for i in range(0,n_days,20)]
  ax.set_xticks(list(range(0,n_days,20))) 
  ax.set_xticklabels(time_label )
  ax.set_xlabel("date")
  ax.set_ylabel("number of beds")
  #ax.legend(loc = 'upper right')
  ax.set_title(plot_title)

  return ax


# =============================== LOAD DATA =============================== 


# Get US demographic data by state ----------------------------------------

# data from 
#https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-detail.html
# file: "Annual Estimates of the Resident Population by Single Year of Age and Sex: April 1, 2010 to July 1, 2018"  

@st.cache
def load_COVID_data():
    data = pickle.load( open( "data/COVID_data_03_23.pkl", "rb" ) )
    return data

@st.cache
def load_demographic():
    US_dem = pd.read_csv("data/US_demographics.csv")
    return US_dem

@st.cache
def load_hospitals():
    US_hospitals = pd.read_csv("data/US_hospitals.csv")
    return US_hospitals

# State name abbreviation  ----------------------------------------------
abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Palau': 'PW',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}

def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
# ================================APP======================================= 
_max_width_()
st.title("When will COVID-19 take over US hospitals?")
st.sidebar.markdown("### User input")
#------------------------- description ------------------------------- 
st.header("What does this app do?" )
st.markdown("This app predicts the following in each state :  \n * number of people that will be infected with COVID-19   \n * dates during which hospitals will experience bed shortage")
st.markdown("The app uses COVID-19 data from the Johns Hopkins CSSE repository to fit a SIR model. The app uses state demographics to predict the number of COVID-19 patients that will require hospital and ICU beds. Finally, the app uses state-wide hospital capacity information to predict the dates of bed shortage.")

# ------------------------------ select state ------------------------------
st.header("Predict number of infected people")
st.markdown("### State information")
# Select state

state_name = st.sidebar.selectbox("Select state", 
    ["Select state", "Alabama","Alaska","Arizona","Arkansas","California","Colorado",
  "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
  "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
  "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
  "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
  "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
  "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
  "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"])



# once a state is selected, return state name and its population
US_dem = load_demographic()

if state_name != "Select state":
    # ------------------------------ get data ---------------------------------- 
    data = load_COVID_data()
    US_dem = load_demographic()
    US_hospitals = load_hospitals()  

    state = abbrev[state_name]
    # get state info
    state_df = get_state_df(data, state, state_name)
    
    # Print most recent status
    date = state_df.tail(1)["report date"].values[0]
    confirmed = state_df.tail(1)["Confirmed"].values[0]
    death = state_df.tail(1)["Deaths"].values[0]

    st.write(state_name + " data as of ", date, ":  \n * Confirmed cases: ", int(confirmed), "  \n * Number of death: ", int(death))
    
    # check if state has enough data
    sufficient, first_day, last_day =  check_data_sufficiency(state_df)
    
    if sufficient == False:
        st.write("Selected state has insufficient data.")
    
#st.markdown("### Select susceptible population")    
#st.write( " * Select percentage of population that will eventually contract COVID-19.  \n * As a reference, about 0.12% of the population tested positive in Hubei, China. ")

st.markdown("### Susceptible population")
if state_name != "Select state":
    n_susceptible = get_susceptible(state)

st.markdown("### Prediction ")
if state_name != "Select state":
    if n_susceptible != 0:

        P = fit_model(state, state_df, first_day, last_day, n_susceptible, 100)
        

        # plot
        fig, ax = plt.subplots(ncols = 2, figsize = (15,5))
        fig.subplots_adjust(bottom = 0.2)
        plot_pred(ax[0], P, state_df, "Infection prediction in " + str(state) + " for 100 days")
    
        # zoom in on first 30 days
        plot_pred(ax[1], P, state_df, "Infection prediction in " + str(state) + " for immediate future")
        ax[1].set_xlim(0, 30)
        ax[1].set_ylim(0,state_df.tail(1)["Infected"].values[0] * 1.5)
        ax.flatten()[1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.12), ncol=2, borderaxespad=0.)
        st.pyplot()

# ------------------------------ Predict hospital bed shortage ------------------------------
st.header("Predict number of hospital beds needed for COVID-19 patients")   
#st.markdown("### Select hospital bed availability")
#st.write("* Select percentage of hospital beds available.  \n * As a reference, about 35% of hospital beds are available in the US.")  

# get percentage of beds available
if state_name != "Select state":
    if n_susceptible != 0:

        bed_percentage = st.sidebar.slider("Select percentage of hospital beds that are available. (As a reference, about 35% of hospital beds are available in the US)")
#st.write( bed_percentage, '% of hospital beds are available.')


st.markdown("### Hospital information")
if (state_name != "Select state") and (n_susceptible != 0) and (bed_percentage != 0):
    # get state demographics
    state_dem = US_dem[US_dem["GEO"] == state_name]

    # predict hospitalization and ICU admission
    hos_prediction = predict_hospitalization_ICU(state_dem, P)

    # compare to state-wide hospital capacity
    state_beds = int(US_hospitals[US_hospitals["State"].str.contains(state)]["StaffedBeds"].values[0])


    beds_available = state_beds * bed_percentage * 0.01
    st.write('*', bed_percentage, '% of hospital beds are available.  \n * Total number of hospital beds in ' + str(state_name) + " is ", state_beds, "  \n * Number of available beds in ", state_name, " is ", int(beds_available))
  
    
st.markdown("### Prediction")
if (state_name != "Select state") and (n_susceptible != 0) and (bed_percentage != 0):
    hospital_info(hos_prediction, beds_available, state_name)
    
    fig, ax = plt.subplots(ncols = 2, figsize = (15, 5))
    fig.subplots_adjust(bottom = 0.27)

    plot_hospital_capacity(ax[0], P, state_df, hos_prediction, beds_available, 'hospital bed prediction in ' + str(state) +' for 100 days', scatter = False)
    plot_hospital_capacity(ax[1], P, state_df, hos_prediction, beds_available, 'hospital bed prediction in ' + str(state) + ' for immediate future', scatter = False)
    
    ax[1].set_xlim(0, 50)
    ax[1].set_ylim(0,beds_available * 1.1)
    ax.flatten()[1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.12), ncol=2)
    st.pyplot()
        

# add my info
st.sidebar.markdown("### About")
st.sidebar.markdown("This app was created by [Iris Yoon.](https://irisyoon.org)  ")

















