@echo off
title SIGN LANGUAGE ENGINE - WEB DASHBOARD
cd /d D:\SIGN-LANGUAGE-PROJECT
call venv\Scripts\activate
echo 🔥 Launching Web UI Dashboard...
streamlit run app_web.py
pause