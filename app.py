from unittest import result
import streamlit as st 
import pickle 
import pandas as pd
import sklearn
from PIL import Image

teams = [ 
 'Sunrisers Hyderabad','Mumbai Indians', 
 'Royal Challengers Bangalore', 
 'Kolkata Knight Riders', 
 'Kings XI Punjab', 
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Daredevils']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata',
       'Chandigarh', 'Delhi', 'Jaipur', 'Chennai', 'Cape Town',
       'Port Elizabeth', 'Durban', 'Centurion', 'East London',
       'Johannesburg', 'Kimberley', 'Ahmedabad', 'Cuttack', 'Nagpur',
       'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi',
       'Abu Dhabi', 'Sharjah', 'Bengaluru', 'Mohali']

pipe = pickle.load(open('pipe.pkl','rb'))

col1, col2 = st.columns(2)
image = Image.open('csk.png')
col1.image(image, caption='CSK', width=200, use_column_width=200)

image = Image.open('msd.jpg')
col2.image(image, caption='MSD', width=200, use_column_width=200)

st.title("IPL WINNING PREDICTOR")
st.write('Owner: Rajeev Kumar')
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',(teams))

    selected_city = st.selectbox('Select host city',sorted(cities))

target  = st.number_input('Target')

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score')

with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if(st.button('Predict Probability')):

    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left


    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],
    'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets],
    'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
    
    st.table(input_df)
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")

