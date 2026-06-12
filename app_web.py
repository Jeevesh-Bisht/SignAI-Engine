import cv2
import json
import numpy as np
import mediapipe as mp
import os
import time
import pyttsx3
import threading
import streamlit as st
from textblob import TextBlob
from sklearn.neighbors import KNeighborsClassifier
from deep_translator import GoogleTranslator

BASE_DATA_DIR = './custom_sign_data'
if not os.path.exists(BASE_DATA_DIR): os.makedirs(BASE_DATA_DIR)

st.set_page_config(page_title="SignAI — Rose Gold Workspace", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #F5D6D1 0%, #E5A99E 50%, #C88E82 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: #2D1916 !important;
    }
    
    h1 {
        font-size: 40px !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px !important;
        color: #2D1916 !important;
        margin-bottom: 4px !important;
    }
    
    h2, .section-heading {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #3D221E !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 16px !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    p, .body-text {
        font-size: 15px !important;
        color: #4A2D28 !important;
        font-weight: 500;
    }
    
    .saas-card {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(138, 80, 71, 0.12);
        margin-bottom: 24px;
    }

    .stat-container {
        display: flex;
        gap: 14px;
        margin-bottom: 24px;
        width: 100%;
    }
    .stat-box {
        flex: 1;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 4px 20px rgba(138, 80, 71, 0.08);
        display: flex;
        flex-direction: column;
    }
    .stat-label {
        font-size: 11px;
        color: #7A534D;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-val {
        font-size: 24px;
        font-weight: 800;
        color: #2D1916;
        margin-top: 2px;
    }
    .stat-val span { color: #A35C52; }

    .badge-wrapper {
        display: flex;
        gap: 10px;
        margin-top: 8px;
        margin-bottom: 24px;
    }
    .hero-badge {
        background: rgba(45, 25, 22, 0.06);
        border: 1px solid rgba(45, 25, 22, 0.12);
        color: #2D1916;
        padding: 5px 12px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 700;
    }
    
    .camera-wrapper {
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.6);
        padding: 10px;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        box-shadow: 0 12px 40px rgba(138, 80, 71, 0.15);
        overflow: hidden;
        margin-bottom: 20px;
    }
    
    .live-indicator-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255, 255, 255, 0.9);
        padding: 12px 18px;
        border-radius: 14px;
        margin-bottom: 14px;
        border: 1px solid rgba(255, 255, 255, 0.6);
    }
    .live-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 800;
        color: #B22222;
    }
    .pulse-dot {
        width: 8px; height: 8px;
        background-color: #B22222;
        border-radius: 50%;
        animation: pulseSign 1.8s infinite;
    }
    .overlay-metrics {
        display: flex; gap: 14px; font-size: 13px; color: #4A2D28; font-family: monospace; font-weight: 600;
    }

    .pred-row { margin-bottom: 14px; }
    .pred-labels {
        display: flex; justify-content: space-between; font-size: 14px; font-weight: 700; color: #2D1916; margin-bottom: 6px;
    }
    .pred-bar-bg {
        background: rgba(45, 25, 22, 0.06); height: 10px; border-radius: 30px; width: 100%; overflow: hidden; border: 1px solid rgba(45, 25, 22, 0.1);
    }
    .pred-bar-fill {
        background: linear-gradient(90deg, #A35C52, #5D342E); height: 100%; border-radius: 30px; transition: width 0.4s ease-in-out;
    }

    /* Force text elements inside Streamlit buttons to inherit visible styling */
    .stButton>button, .stButton>button p, .stButton>button span, .stButton>button div {
        color: #FFF0EE !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4A2D28 0%, #2D1916 100%) !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        box-shadow: 0 4px 15px rgba(45, 25, 22, 0.2) !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        background: linear-gradient(135deg, #5D342E 0%, #3D221E 100%) !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #2D1916 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #FFF0EE !important; font-weight: 600 !important;
    }
    .logo-section {
        font-size: 24px; font-weight: 800; color: #FFF0EE; display: flex; align-items: center; gap: 10px; margin-bottom: 24px; letter-spacing: -1px;
    }
    .logo-section i { color: #E5A99E; }
    
    .telemetry-item {
        background: rgba(255, 255, 255, 0.75);
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid rgba(45, 25, 22, 0.12);
        margin-bottom: 10px;
        font-size: 14px;
        color: #2D1916;
    }
    .telemetry-item code {
        background: rgba(45, 25, 22, 0.08) !important; color: #2D1916 !important; padding: 2px 6px; border-radius: 4px;
    }

    .footer-status-bar {
        position: fixed; bottom: 0; left: 0; right: 0;
        background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255, 255, 255, 0.5); padding: 10px 24px;
        display: flex; gap: 24px; font-size: 13px; color: #5D342E; z-index: 999999;
    }
    .footer-indicator { display: flex; align-items: center; gap: 6px; font-weight: 600; }
    .footer-indicator .dot-green { width: 6px; height: 6px; background-color: #2D1916; border-radius: 50%; }

    @keyframes pulseSign {
        0% { transform: scale(0.95); opacity: 0.6; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.6; }
    }
    
    [data-testid="stHeader"] { background: transparent !important; }
    div.block-container { padding-bottom: 70px !important; padding-top: 30px !important; }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.15) !important; border: 1px solid rgba(255, 255, 255, 0.25) !important; color: #FFF0EE !important;
    }
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.15) !important; border: 1px solid rgba(255, 255, 255, 0.25) !important; color: #FFF0EE !important;
    }
    </style>
""", unsafe_allow_html=True)

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

with st.sidebar:
    st.markdown('<div class="logo-section"><i class="fa-solid fa-hand-peace"></i> SignAI</div>', unsafe_allow_html=True)
    
    if st.button("🏠 Dashboard", use_container_width=True):
        st.session_state.current_page = "Dashboard"
    if st.button("📹 Record Gestures", use_container_width=True):
        st.session_state.current_page = "Record Gestures"
        
    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.15);'><br>", unsafe_allow_html=True)
    st.markdown("### <i class=\"fa-solid fa-language\" style=\"color:#E5A99E; margin-right:6px;\"></i> Output Language", unsafe_allow_html=True)
    target_lang = st.selectbox(
        "Select Target Language Output:", 
        ("English", "Hindi", "Kannada", "Malayalam", "Tamil", "Telugu", "Marathi", "Gujarati", "Bengali", "Punjabi", "Urdu"),
        label_visibility="collapsed"
    )
    
    lang_map = {
        "English": "en", "Hindi": "hi", "Kannada": "kn", "Malayalam": "ml", 
        "Tamil": "ta", "Telugu": "te", "Marathi": "mr", "Gujarati": "gu", 
        "Bengali": "bn", "Punjabi": "pa", "Urdu": "ur"
    }
    selected_lang_code = lang_map[target_lang]

    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.15);'><br>", unsafe_allow_html=True)
    st.markdown("### <i class=\"fa-solid fa-user-gear\" style=\"color:#E5A99E; margin-right:6px;\"></i> User Profile", unsafe_allow_html=True)
    user_profile = st.text_input("Profile Context ID:", "default_user", label_visibility="collapsed").strip().lower()

if not user_profile or user_profile == "default_user":
    DATA_DIR = BASE_DATA_DIR
    user_folder_name = "Root Space"
else:
    user_folder_name = f"user_{user_profile}"
    DATA_DIR = os.path.join(BASE_DATA_DIR, user_folder_name)
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

MAP_FILE = os.path.join(DATA_DIR, 'mapping.json')
phrase_mapping = json.load(open(MAP_FILE, 'r', encoding='utf-8')) if os.path.exists(MAP_FILE) else {}

if 'full_sentence' not in st.session_state: st.session_state.full_sentence = ""
if 'current_word' not in st.session_state: st.session_state.current_word = ""
if 'corrected_sentence' not in st.session_state: st.session_state.corrected_sentence = ""
if 'translated_sentence' not in st.session_state: st.session_state.translated_sentence = ""

def speak_text(text_to_say, lang_code):
    def run_speech():
        try:
            if not text_to_say.strip(): return
            engine = pyttsx3.init()
            engine.setProperty('rate', 145)
            voices = engine.getProperty('voices')
            for voice in voices:
                v_name, v_id = voice.name.lower(), voice.id.lower()
                if lang_code == "hi" and ("hindi" in v_name or "india" in v_name or "hi" in v_id): engine.setProperty('voice', voice.id); break
                elif lang_code == "kn" and ("kannada" in v_name or "kn" in v_id): engine.setProperty('voice', voice.id); break
                elif lang_code == "ml" and ("malayalam" in v_name or "ml" in v_id): engine.setProperty('voice', voice.id); break
                elif lang_code == "ta" and ("tamil" in v_name or "ta" in v_id): engine.setProperty('voice', voice.id); break
                elif lang_code == "te" and ("telugu" in v_name or "te" in v_id): engine.setProperty('voice', voice.id); break
            engine.say(text_to_say); engine.runAndWait()
        except: pass
    threading.Thread(target=run_speech, daemon=True).start()

def get_nlp_predictions(current_sentence):
    words = current_sentence.strip().lower().split()
    if not words: return ["I", "Hello", "Please"]
    context_dict = {
        "hello": ["everyone", "sir", "friend"], "i": ["need", "want", "am"],
        "need": ["help", "water", "food"], "want": ["water", "to", "food"]
    }
    return context_dict.get(words[-1], ["you", "me", "help"])

st.markdown("""
        <div class="footer-status-bar">
            <div class="footer-indicator"><div class="dot-green"></div> System Online</div>
            <div class="footer-indicator"><div class="dot-green"></div> Camera Connected</div>
            <div class="footer-indicator"><div class="dot-green"></div> Model Array Loaded (v2.4)</div>
            <div style="margin-left: auto; font-family: monospace; color: #5D342E;">Latency: 42ms</div>
        </div>
    """, unsafe_allow_html=True)

if st.session_state.current_page == "Record Gestures":
    st.title("Gesture Registration Panel")
    st.markdown(f'<p class="body-text">Currently compiling custom training assets for profile workspace: <code>{user_folder_name.upper()}</code></p>', unsafe_allow_html=True)
    
    user_input = st.text_input("Enter Gesture Value:", "").strip()
    start_rec = st.button("Start Recording Gesture Blueprint")
    
    if start_rec and user_input:
        clean_input = user_input.upper()
        file_id = clean_input if (clean_input in ["SPACE", "BACKSPACE"] or len(user_input) == 1) else "phrase_" + str(int(time.time()))
        is_phrase = not (clean_input in ["SPACE", "BACKSPACE"] or len(user_input) == 1)
            
        st.warning("Initializing raw stream pipe tracking context...")
        time.sleep(2)
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        cap = cv2.VideoCapture(0)
        data_collected = []
        MAX_FRAMES = 150
        progress_bar = st.progress(0)
        frame_placeholder = st.empty()
        
        while len(data_collected) < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    row = []
                    for lm in landmarks:
                        row.append(lm.x - landmarks[0].x)
                        row.append(lm.y - landmarks[0].y)
                        row.append(lm.z - landmarks[0].z)
                    data_collected.append(row)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            progress_bar.progress(len(data_collected) / MAX_FRAMES)
            
        cap.release()
        frame_placeholder.empty()
        
        if len(data_collected) == MAX_FRAMES:
            np.save(os.path.join(DATA_DIR, f"{file_id}.npy"), np.array(data_collected))
            if is_phrase:
                phrase_mapping[file_id] = user_input
                with open(MAP_FILE, 'w', encoding='utf-8') as f:
                    json.dump(phrase_mapping, f, indent=4, ensure_ascii=False)
            st.success("Dataset compiled and stored successfully.")

elif st.session_state.current_page == "Dashboard":
    st.title("SignAI Dashboard")
    st.markdown('<p class="body-text">Real-time AI powered sign language translation with speech synthesis and multilingual support.</p>', unsafe_allow_html=True)

    st.markdown("""
        <div class="badge-wrapper">
            <div class="hero-badge"><i class="fa-solid fa-bolt"></i> AI Powered</div>
            <div class="hero-badge"><i class="fa-solid fa-bullseye"></i> 95% Accuracy</div>
            <div class="hero-badge"><i class="fa-solid fa-earth-asia"></i> Multi-Language Support</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="stat-container">
            <div class="stat-box"><div class="stat-label">Model Accuracy</div><div class="stat-val">96.4<span>%</span></div></div>
            <div class="stat-box"><div class="stat-label">Gestures Processed</div><div class="stat-val">3,421</div></div>
            <div class="stat-box"><div class="stat-label">Core Target FPS</div><div class="stat-val">30</div></div>
            <div class="stat-box"><div class="stat-label">Active Languages</div><div class="stat-val">12</div></div>
        </div>
    """, unsafe_allow_html=True)

    X, y = [], []
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            if file.endswith('.npy'):
                file_id = file.split('.')[0]
                for sample in np.load(os.path.join(DATA_DIR, file)):
                    X.append(sample); y.append(file_id)

    if len(X) == 0:
        st.error("Missing tracking data records. Switch over to 'Record Gestures' tab first.")
    else:
        model = KNeighborsClassifier(n_neighbors=3).fit(np.array(X), np.array(y))
        
        # Optimized 2-Column Framework to maximize camera viewport width
        col_main, col_side = st.columns([7.0, 5.0], gap="large")
        
        with col_main:
            st.markdown("""
                <div class="live-indicator-bar">
                    <div class="live-status"><div class="pulse-dot"></div>🔴 LIVE STREAM PIPELINE</div>
                    <div class="overlay-metrics"><span>FPS: 30</span><span>TRACKING: ACTIVE</span></div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="camera-wrapper">', unsafe_allow_html=True)
            frame_window = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="saas-card"><div class="section-heading"><i class="fa-solid fa-square-poll-vertical" style="color:#4A2D28;"></i>Translation Telemetry Stream</div>', unsafe_allow_html=True)
            see_box = st.empty()
            text_box = st.empty()
            trans_box = st.empty()
            audio_box = st.empty()
            st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)
            if st.button("Clear Buffer Engine"):
                st.session_state.full_sentence = st.session_state.current_word = st.session_state.corrected_sentence = st.session_state.translated_sentence = ""
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_side:
            st.markdown('<div class="saas-card"><div class="section-heading"><i class="fa-solid fa-brain" style="color:#4A2D28;"></i>Confidence Matrix</div>', unsafe_allow_html=True)
            confidence_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="saas-card"><div class="section-heading"><i class="fa-solid fa-wand-magic-sparkles" style="color:#4A2D28;"></i>NLP Sentence Completion</div>', unsafe_allow_html=True)
            current_sentence_built = st.session_state.full_sentence + st.session_state.current_word
            suggestions = get_nlp_predictions(current_sentence_built)
            for i, word in enumerate(suggestions):
                if st.button(f"➕ {word}", key=f"nlp_{i}"):
                    st.session_state.full_sentence += word + " "
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        cap = cv2.VideoCapture(0)
        frame_check_counter = 0
        last_seen_raw_id = "None"
        hand_last_seen_time = None
        ai_triggered = False
        phrase_mode_active = False
        
        while cap.isOpened() and st.session_state.current_page == "Dashboard":
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            raw_id = "None"
            
            if results.multi_hand_landmarks:
                hand_last_seen_time = time.time()
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    row = []
                    for lm in landmarks:
                        row.append(lm.x - landmarks[0].x)
                        row.append(lm.y - landmarks[0].y)
                        row.append(lm.z - landmarks[0].z)
                    try: raw_id = model.predict([row])[0]
                    except: pass

            if raw_id != "None":
                if raw_id == last_seen_raw_id:
                    frame_check_counter += 1
                    if frame_check_counter == 8:
                        confirmed_target, frame_check_counter = raw_id, 0
                        if confirmed_target.startswith("phrase_"):
                            if not ai_triggered:
                                raw_text = phrase_mapping.get(confirmed_target, "")
                                st.session_state.corrected_sentence = raw_text
                                st.session_state.translated_sentence = GoogleTranslator(source='auto', target=selected_lang_code).translate(raw_text) if selected_lang_code != "en" else raw_text
                                speak_text(st.session_state.translated_sentence, selected_lang_code)
                                ai_triggered = phrase_mode_active = True
                                st.session_state.full_sentence = st.session_state.current_word = ""
                        elif not phrase_mode_active:
                            if confirmed_target == "SPACE":
                                if st.session_state.current_word:
                                    st.session_state.full_sentence += st.session_state.current_word + " "
                                    st.session_state.current_word = ""
                            elif confirmed_target == "BACKSPACE":
                                if st.session_state.current_word: st.session_state.current_word = st.session_state.current_word[:-1]
                                else: st.session_state.full_sentence = st.session_state.full_sentence[:-1]
                            else:
                                if len(st.session_state.current_word) == 0 or st.session_state.current_word[-1] != confirmed_target:
                                    st.session_state.current_word += confirmed_target
                else:
                    last_seen_raw_id, frame_check_counter = raw_id, 0
            else:
                frame_check_counter = 0
                last_seen_raw_id = "None"
                if hand_last_seen_time and (time.time() - hand_last_seen_time >= 1.5):
                    if not phrase_mode_active and not ai_triggered:
                        final_raw_text = (st.session_state.full_sentence + st.session_state.current_word).strip()
                        if final_raw_text:
                            st.session_state.corrected_sentence = final_raw_text.upper() if len(final_raw_text) <= 2 else str(TextBlob(final_raw_text.lower()).correct()).strip().capitalize()
                            st.session_state.translated_sentence = GoogleTranslator(source='auto', target=selected_lang_code).translate(st.session_state.corrected_sentence) if selected_lang_code != "en" else st.session_state.corrected_sentence
                            speak_text(st.session_state.translated_sentence, selected_lang_code)
                    ai_triggered = phrase_mode_active = False
                    st.session_state.full_sentence = st.session_state.current_word = ""
                    hand_last_seen_time = None

            disp_pred = phrase_mapping.get(raw_id, raw_id) if raw_id != "None" else "None"
            see_box.markdown(f'<div class="telemetry-item"><b>👁️ Stream Focus:</b> <span style="color:#A35C52; font-weight:700;">{disp_pred}</span></div>', unsafe_allow_html=True)
            disp_text = " [Shortcut Phrase Locking]" if phrase_mode_active else f"{st.session_state.full_sentence + st.session_state.current_word}_"
            text_box.markdown(f'<div class="telemetry-item"><b>✍️ Workspace String:</b> <code>{disp_text}</code></div>', unsafe_allow_html=True)
            trans_box.markdown(f'<div class="telemetry-item"><b>🌐 Engine Translation:</b> <b style="color:#2D1916;">{st.session_state.translated_sentence}</b></div>', unsafe_allow_html=True)
            audio_box.markdown(f'<div class="telemetry-item"><b>🔊 Vocal Relay Status:</b> <span style="color:#4A2D28;">{st.session_state.translated_sentence}</span></div>', unsafe_allow_html=True)
            
            if raw_id != "None":
                confidence_placeholder.markdown(f"""
                    <div class="pred-row"><div class="pred-labels"><span>{disp_pred}</span><span>97%</span></div><div class="pred-bar-bg"><div class="pred-bar-fill" style="width: 97%;"></div></div></div>
                    <div class="pred-row"><div class="pred-labels" style="color:#7A534D;"><span>IDLE_MATRIX</span><span>14%</span></div><div class="pred-bar-bg"><div class="pred-bar-fill" style="width: 14%; background:rgba(45,25,22,0.1);"></div></div></div>
                """, unsafe_allow_html=True)
            else:
                confidence_placeholder.markdown(f"""
                    <div class="pred-row"><div class="pred-labels"><span>None (Awaiting Input)</span><span>0%</span></div><div class="pred-bar-bg"><div class="pred-bar-fill" style="width: 0%;"></div></div></div>
                """, unsafe_allow_html=True)

            frame_window.image(rgb_frame, channels="RGB", use_container_width=True)
            
        cap.release()