@echo off
title SignAI Automated Launcher
echo  Initializing SignAI Assistive Translation Engine  
echo.
echo Step 1: Checking and installing dependencies...
pip install -r requirements.txt
echo.
echo Step 2: Launching Real-Time Streamlit Dashboard...
streamlit run app_web.py
pause
