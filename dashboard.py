import streamlit as st
import requests
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SurakshaVaani Pro Console",
    page_icon="🛡️",
    layout="wide"
)

# --- HEADER ---
st.title("🛡️ SurakshaVaani Pro: Dual-Engine Threat Detection")
st.markdown("### Audio Intelligence: Tone (Emotion) + Verbal (Keywords)")
st.divider()

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎤 Live Audio Feed Analysis")
    st.info("Upload audio to scan for Screams, Anger, or Keywords (e.g., 'Help').")
    
    # File Uploader
    audio_file = st.file_uploader("Upload Audio Clip", type=['wav', 'mp3'])
    
    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("🚨 SCAN FOR THREATS", type="primary", use_container_width=True):
            with st.spinner("Running Dual-Engine Analysis (Tone + Speech)..."):
                try:
                    # 1. CALL THE NEW API ENDPOINT
                    files = {"file": audio_file.getvalue()}
                    # Note: We are hitting /analyze-threat now
                    res = requests.post("http://127.0.0.1:8000/analyze-threat", files={"file": audio_file})
                    
                    if res.status_code == 200:
                        data = res.json()
                        
                        # 2. EXTRACT DATA
                        is_sos = data['sos_activated']
                        reasons = data['reasons']
                        transcript = data['transcription']
                        tone = data['tone_analysis']
                        
                        # 3. DISPLAY RESULTS
                        st.markdown("### 🔍 Analysis Results")
                        
                        # SOS BANNER
                        if is_sos:
                            st.error(f"## 🆘 SOS TRIGGERED")
                            for r in reasons:
                                st.write(f"🔴 **CRITICAL FACTOR:** {r}")
                        else:
                            st.success("## ✅ STATUS SAFE")
                            st.write("No threat biomarkers detected.")

                        st.divider()
                        
                        # DETAILED METRICS
                        m1, m2, m3 = st.columns(3)
                        m1.metric("🗣️ Tone Detected", tone['emotion'])
                        m2.metric("🔊 Loudness (0-1)", f"{tone['loudness']:.3f}")
                        m3.metric("🤖 AI Confidence", f"{tone['confidence'] * 100:.1f}%")
                        
                        # TRANSCRIPT BOX
                        st.warning(f"📝 **Heard Words:** \"{transcript}\"")
                        
                        # RAW DATA EXPANDER
                        with st.expander("View System Telemetry"):
                            st.json(data)

                    else:
                        st.error(f"Server Error {res.status_code}: {res.text}")
                        
                except Exception as e:
                    st.error(f"Connection Failed. Is the server running? Error: {e}")

with col2:
    st.subheader("📡 Live Operator Feed")
    # Simulation of real-time logs
    st.markdown("""
    | ID | Timestamp | Threat Type | Transcription | Status |
    |---|---|---|---|---|
    | #901 | 10:45:01 | **SCREAM** | *[Unintelligible]* | 🚓 Dispatched |
    | #902 | 10:45:15 | KEYWORD | "Please help me" | 📞 Connecting |
    | #903 | 10:46:00 | NEUTRAL | "Just testing" | ✅ Ignored |
    """)
    st.info("System is listening on Frequency 16kHz...")