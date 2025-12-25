import streamlit as st
import subprocess
import os
import pandas as pd
from PIL import Image as PILImage
import numpy as np
import matplotlib.pyplot as plt
import json
import streamlit.components.v1 as components
import re
from io import BytesIO
import pickle

st.set_page_config(
    page_title="IPL Winning Predictor",
    page_icon="🏏",
    layout='wide',  # Only 'centered' or 'wide' are valid options
)

# Inject custom CSS to reduce top padding/margin
st.markdown("""
    <style>
        /* Remove top padding from main container */
        .block-container {
            padding-top: 0rem !important;
        }

        /* Optional: Remove padding from header if it's showing */
        header {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }

        /* Optional: Remove margin from main content */
        main {
            margin-top: 0rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

button_style = """
    <style>
        .stButton>button {
            box-shadow: 1px 1px 1px rgba(0, 0, 0, 0.8);
        }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)
pipe = pickle.load(open('pipe.pkl','rb'))
def main(): 
    card_button_style = """
        <style>
        .card-button {
            width: 100%;
            padding: 20px;
            background-color: white;
            border: none;
            border-radius: 10px;
            box-shadow: 0 2px 2px rgba(0,0,0,0.2);
            transition: box-shadow 0.3s ease;
            text-align: center;
            font-size: 16px;
            cursor: pointer;
        }
        .card-button:hover {
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        }
        </style>
    """

    st.markdown(
        """
        <style>
        body {
            background-color: #bfe1ff;  /* Set your desired background color here */
            animation: changeColor 5s infinite;
        }
        .css-18e3th9 {
            padding-top: 0rem;  /* Adjust the padding at the top */
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .viewerBadge_container__1QSob {visibility: hidden;}
        .stActionButton {margin: 5px;} /* Optional: Adjust button spacing */
        header .stApp [title="View source on GitHub"] {
            display: none;
        }
        .stApp header, .stApp footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )
            
    _, col2, _ = st.columns([1,1,0.5])
    with col2:
        st.markdown("## :rainbow[IPL Winning Predictor]")

    st.markdown(
        """<div style='margin-top:-46px;'>
               <hr style="border:1px solid red">
           </div>""",
        unsafe_allow_html=True
    )
    st.markdown("""
        <style>
        .stButton button {
            height: 30px;
            width: 166px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    teams = sorted(['Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'])

    cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
        'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
        'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
        'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
        'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
        'Sharjah', 'Mohali', 'Bengaluru']

    col1,col2,col3 = st.columns(3)
    with col1:
        batting_team =st.selectbox('Select the batting team',teams)
    with col2:
        bowling_team = st.selectbox('Select the bowling team',teams)
    with col3 :
        target = st.number_input('Target',min_value=0, key="target")
    col4,col5,col6,col7 = st.columns(4)
    with col4:
        selected_city = st.selectbox('Cities',sorted(cities), key="city")
    with col5:
        score =st.number_input('Score',min_value=0)
    with col6:
        wickets =st.number_input('Wickets',min_value=0,max_value=9)
    with col7:
        overs = st.number_input('Overs completed',min_value=0,max_value=20)
    if st.button('Predict Probability'):
        runs_left = target - score
        balls_left = 120 - overs*6
        wickets = 10 - wickets
        
        if overs > 0:
            crr = score / overs
            rrr = runs_left * 6 / balls_left
            
            df = pd.DataFrame({
                'batting_team':[batting_team],
                'bowling_team':[bowling_team],
                'city':[selected_city],
                'runs_left':[runs_left],
                'balls_left':[balls_left],
                'wickets':[wickets],
                'total_runs_x':[target],
                'crr':[crr],
                'rrr':[rrr]
            })
            
            result = pipe.predict_proba(df)
            r_1 = round(result[0][0] * 100)
            r_2 = round(result[0][1] * 100)

            # Header
            st.markdown("<h2 style='text-align:center; color:#00eaff;'>🏆 Match Win Probability</h2>", unsafe_allow_html=True)

            # Cards
            colA, colB = st.columns(2)

            with colA:
                st.markdown("""
                    <div style='background:#0a0f24; padding:20px; border-radius:12px; border:1px solid #1f2b4d; text-align:center;'>
                        <h3 style='color:#00ff7f;'>{} 🏏</h3>
                        <h1 style='color:#00ff7f;'>{}%</h1>
                        <p style='color:#cccccc;'>Winning Chance</p>
                    </div>
                """.format(batting_team, r_2), unsafe_allow_html=True)

            with colB:
                st.markdown("""
                    <div style='background:#0a0f24; padding:20px; border-radius:12px; border:1px solid #1f2b4d; text-align:center;'>
                        <h3 style='color:#ff4d4d;'>{} 🛡️</h3>
                        <h1 style='color:#ff4d4d;'>{}%</h1>
                        <p style='color:#cccccc;'>Winning Chance</p>
                    </div>
                """.format(bowling_team, r_1), unsafe_allow_html=True)

            # Progress Bar Section
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📊 Probability Comparison")

            st.progress(r_2 / 100)
            st.write(f"**{batting_team}: {r_2}%**")

            st.progress(r_1 / 100)
            st.write(f"**{bowling_team}: {r_1}%**")
                
if __name__ == "__main__":
    main()
    
st.markdown('<hr style="border:1px solid black">', unsafe_allow_html=True)
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        .footer {
            background-color: #f8f9fa;
            padding: 20px 0;
            color: #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
            text-align: center;
        }
        .footer .logo {
            flex: 1;
        }
        .footer .logo img {
            max-width: 150px;
            height: auto;
        }
        .footer .social-media {
            flex: 2;
        }
        .footer .social-media p {
            margin: 0;
            font-size: 16px;
        }
        .footer .icons {
            margin-top: 10px;
        }
        .footer .icons a {
            margin: 0 10px;
            color: #666;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        .footer .icons a:hover {
            color: #0077b5; /* LinkedIn color as default */
        }
        .footer .icons a .fab {
            font-size: 28px;
        }
        .footer .additional-content {
            margin-top: 10px;
        }
        .footer .additional-content h4 {
            margin: 0;
            font-size: 18px;
            color: #007bff;
        }
        .footer .additional-content p {
            margin: 5px 0;
            font-size: 16px;
        }
    </style>
   <div class="footer">
        <div class="social-media" style="flex: 2;">
            <p>&copy; 2024. All Rights Reserved</p>
            <div class="icons">
                <a href="https://twitter.com/rajeev?lang=en" target="_blank"><i class="fab fa-twitter" style="color: #1DA1F2;"></i></a>
                <a href="https://www.facebook.com/rajeevkumar/" target="_blank"><i class="fab fa-facebook" style="color: #4267B2;"></i></a>
                <a href="https://www.instagram.com/_rajeevkumar1/?hl=en" target="_blank"><i class="fab fa-instagram" style="color: #E1306C;"></i></a>
                <a href="https://www.linkedin.com/rajeev-kumar-nit/" target="_blank"><i class="fab fa-linkedin" style="color: #0077b5;"></i></a>
            </div>
            <div class="additional-content">
                <h4>Contact Us</h4>
                <p>Email: kumarrajeev66797@gmail.com | Phone: 7091895623</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True
)

with st.container():
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 8px; font-size: 19px;'>
            <span style='font-weight: 600;'>Website Visitors Count:</span>
            <a href="https://equest-utilities-edsglobal.streamlit.app/" target="_blank">
                <img src="https://hitwebcounter.com/counter/counter.php?page=15322595&style=0019&nbdigits=5&type=ip&initCount=0"
                     title="Counter Widget" alt="Visit counter For Websites" border="0"
                     style="height: 20px;" />
            </a>
        </div>
        """,
        unsafe_allow_html=True

    )


