import sys
import os
import threading
import queue
import time
import socketserver
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import mediapipe as mp
import streamlit as st
import streamlit.components.v1 as components

import inference

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ASL Recognition", page_icon="🤟", layout="wide")
STABILITY_THRESHOLD = 5
MJPEG_PORT = 5001

st.markdown("""
<style>
  .stApp { background: #0d1117; }
  #MainMenu, footer, header { visibility: hidden; }
  .card {
    background: linear-gradient(145deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
  }
  .letter-display {
    font-size: 140px;
    font-weight: 900;
    text-align: center;
    line-height: 1;
  }
  .sentence-box {
    background: #010409;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px 22px;
    font-size: 30px;
    font-family: 'Consolas', monospace;
    color: #e6edf3;
    min-height: 72px;
    word-break: break-all;
    letter-spacing: 2px;
  }
  .stButton > button {
    background: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    padding: 10px !important;
  }
  .stButton > button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Shared state ──────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.lock          = threading.Lock()
        self.latest_frame  = None   # JPEG bytes ready to stream
        self.letter        = ""
        self.confidence    = 0.0
        self.stable_count  = 0
        self._stable_letter  = ""
        self._last_appended  = ""
        self.sentence      = ""

if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()
state: AppState = st.session_state.app_state

# ── MJPEG HTTP server ─────────────────────────────────────────────────────────
class _MJPEGHandler(BaseHTTPRequestHandler):
    shared: AppState = None

    def log_message(self, *_):
        pass  # silence request logs

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=--frame")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with self.shared.lock:
                    jpg = self.shared.latest_frame
                if jpg is None:
                    time.sleep(0.01)
                    continue
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpg + b"\r\n"
                )
        except Exception:
            pass

@st.cache_resource
def start_pipeline(_shared: AppState):
    """Starts the MJPEG server + capture thread + inference thread. Runs once."""
    shared = _shared

    # Wire shared state into the handler class
    _MJPEGHandler.shared = shared

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    server = ReusableTCPServer(("", MJPEG_PORT), _MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    # ── Inference thread ──────────────────────────────────────────────────────
    roi_q       = queue.Queue(maxsize=1)
    infer_buf   = ["", 0.0]
    infer_lock  = threading.Lock()

    def infer_worker():
        while True:
            roi = roi_q.get()
            try:
                letter, conf = inference.predict(roi)
            except Exception as e:
                print(f"[infer] {e}")
                letter, conf = "", 0.0
            with infer_lock:
                infer_buf[0], infer_buf[1] = letter, conf

    threading.Thread(target=infer_worker, daemon=True).start()

    # ── Capture + MediaPipe thread ────────────────────────────────────────────
    def capture_worker():
        hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.5,
        )
        mp_draw   = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
        PADDING   = 30

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab the freshest frame

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # MediaPipe
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_roi = None
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, lm, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                xs = [p.x * w for p in lm.landmark]
                ys = [p.y * h for p in lm.landmark]
                x1, y1 = max(0, int(min(xs)) - PADDING), max(0, int(min(ys)) - PADDING)
                x2, y2 = min(w, int(max(xs)) + PADDING), min(h, int(max(ys)) + PADDING)
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    hand_roi = roi

            # Submit ROI (non-blocking)
            if hand_roi is not None:
                try:
                    roi_q.put_nowait(hand_roi.copy())
                except queue.Full:
                    pass

            # Read latest inference result
            with infer_lock:
                current_letter = infer_buf[0] if hand_roi is not None else ""
                current_conf   = infer_buf[1] if hand_roi is not None else 0.0

            # Stability + sentence
            with shared.lock:
                if current_letter == shared._stable_letter and current_letter:
                    shared.stable_count += 1
                else:
                    shared._stable_letter = current_letter
                    shared.stable_count   = 1
                    shared._last_appended = ""

                if shared.stable_count == STABILITY_THRESHOLD:
                    if current_letter not in ("", "nothing", "del", "space") \
                            and shared._last_appended != current_letter:
                        shared.sentence      += current_letter
                        shared._last_appended = current_letter
                        shared.stable_count   = 0
                    elif current_letter == "del" and shared._last_appended != "del":
                        shared.sentence       = shared.sentence[:-1]
                        shared._last_appended = "del"
                        shared.stable_count   = 0
                    elif current_letter == "space" and shared._last_appended != "space":
                        shared.sentence      += " "
                        shared._last_appended = "space"
                        shared.stable_count   = 0

                shared.letter     = current_letter
                shared.confidence = current_conf
                stable_count      = shared.stable_count
                sentence          = shared.sentence

            # ── Overlays ──────────────────────────────────────────────────────
            ratio     = min(stable_count, STABILITY_THRESHOLD) / STABILITY_THRESHOLD
            bar_color = (50, 210, 100) if ratio < 1.0 else (0, 255, 128)
            cv2.rectangle(frame, (0, 0), (w, 6), (30, 30, 35), -1)
            if ratio > 0:
                cv2.rectangle(frame, (0, 0), (int(w * ratio), 6), bar_color, -1)

            cv2.rectangle(frame, (0, h - 52), (w, h), (12, 12, 18), -1)
            disp = sentence[-46:] if len(sentence) > 46 else sentence
            cv2.putText(frame, disp or "[ start signing... ]",
                        (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (230, 237, 243), 2, cv2.LINE_AA)

            if hand_roi is None:
                cv2.putText(frame, "Show your hand",
                            (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (90, 90, 110), 2, cv2.LINE_AA)

            # Encode as JPEG (fast) and store for MJPEG server
            _, jpg = cv2.imencode(".jpg", frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 75])
            with shared.lock:
                shared.latest_frame = jpg.tobytes()

    threading.Thread(target=capture_worker, daemon=True).start()
    return MJPEG_PORT

port = start_pipeline(state)

# ── Layout ────────────────────────────────────────────────────────────────────
col_vid, col_panel = st.columns([3, 2], gap="large")

with col_vid:
    # Browser pulls MJPEG frames directly — zero WebSocket overhead
    components.html(f"""
    <div style="background:#0d1117;border-radius:14px;overflow:hidden;">
      <img id="feed" src="http://localhost:{port}"
           style="width:100%;display:block;border-radius:14px;"
           onerror="setTimeout(()=>{{document.getElementById('feed').src='http://localhost:{port}?t='+Date.now()}},500)">
    </div>
    """, height=500)

with col_panel:

    @st.fragment(run_every=0.3)
    def prediction_panel():
        with state.lock:
            letter   = state.letter
            conf     = state.confidence
            stable   = min(state.stable_count, STABILITY_THRESHOLD)
            sentence = state.sentence

        if letter and letter != "nothing":
            color, label = "#00e676", letter
        elif letter == "nothing":
            color, label = "#546e7a", "·"
        else:
            color, label = "#37474f", "—"

        conf_pct = int(conf * 100)
        st.markdown(f"""
        <div class="card" style="text-align:center;">
          <div class="letter-display" style="color:{color};">{label}</div>
          <div style="color:#8b949e;font-size:16px;margin-top:6px;">{conf_pct}% confidence</div>
          <div style="background:#21262d;border-radius:8px;height:8px;margin:14px 0 4px;">
            <div style="background:{color};width:{conf_pct}%;height:100%;border-radius:8px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        dots = "".join(
            f'<div style="width:18px;height:18px;border-radius:50%;'
            f'background:{"#00e676" if i < stable else "#21262d"};'
            f'display:inline-block;margin:0 4px;"></div>'
            for i in range(STABILITY_THRESHOLD)
        )
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:16px 24px;">
          <div style="color:#8b949e;font-size:13px;margin-bottom:10px;
                      text-transform:uppercase;letter-spacing:1px;">Stability</div>
          {dots}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card" style="padding:16px 22px;">
          <div style="color:#8b949e;font-size:13px;margin-bottom:10px;
                      text-transform:uppercase;letter-spacing:1px;">Sentence</div>
          <div class="sentence-box">{sentence if sentence else "&nbsp;"}</div>
        </div>
        """, unsafe_allow_html=True)

    prediction_panel()

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("⎵  Space", use_container_width=True):
            with state.lock:
                state.sentence += " "
    with b2:
        if st.button("⌫  Delete", use_container_width=True):
            with state.lock:
                state.sentence = state.sentence[:-1]
    with b3:
        if st.button("✕  Clear", use_container_width=True):
            with state.lock:
                state.sentence = ""
