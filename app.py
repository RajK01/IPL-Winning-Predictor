import streamlit as st
import pickle
import pandas as pd

# Define teams and cities
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata',
    'Chandigarh', 'Delhi', 'Jaipur', 'Chennai', 'Cape Town',
    'Port Elizabeth', 'Durban', 'Centurion', 'East London',
    'Johannesburg', 'Kimberley', 'Ahmedabad', 'Cuttack', 'Nagpur',
    'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi',
    'Abu Dhabi', 'Sharjah', 'Bengaluru', 'Mohali'
]

# Load the machine learning pipeline from a pickled file
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Streamlit UI
st.title("IPL WINNING PREDICTOR")
st.write('Owner: Rajeev Kumar')

# Layout with two columns for inputs
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
    target = st.number_input('Target', min_value=0, step=1)

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))
    selected_city = st.selectbox('Select host city', sorted(cities))

# Additional inputs
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', min_value=0, step=1)

with col4:
    overs = st.number_input('Overs completed')

with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

# Prediction button
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    # Create input dataframe for prediction
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],  # Use 'wickets' instead of 'wickets_left'
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Display input dataframe
    st.table(input_df)

    # Make prediction
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Display results
    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")
