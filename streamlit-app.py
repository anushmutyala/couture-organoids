"""
Streamlit ECG Dashboard (4‑up real‑time view)

* Top‑left  : Human ECG trace
* Top‑right : Organoid ECG trace
* Bottom‑left: Bayesian update (error between traces)
* Bottom‑right: Opentrons camera feed placeholder

How to run:
    streamlit run streamlit_ecg_dashboard.py

Replace the `get_next_sample()` function with your own data ingest logic.
"""
import time
from typing import Tuple, Deque

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from collections import deque

# ————————————————————————————————————————————————
# CONFIGURATION
# ————————————————————————————————————————————————
BUFFER_SECONDS: int = 5            # visible window length in seconds
FS: int = 500                     # sampling rate [Hz]
MAX_POINTS: int = BUFFER_SECONDS * FS

# ————————————————————————————————————————————————
# DATA BUFFERS
# ————————————————————————————————————————————————
t_buf: Deque[float] = deque(maxlen=MAX_POINTS)
human_buf: Deque[float] = deque(maxlen=MAX_POINTS)
organoid_buf: Deque[float] = deque(maxlen=MAX_POINTS)


# ————————————————————————————————————————————————
# STUB FOR REAL DATA INGEST
# ————————————————————————————————————————————————
# ↓↓↓  REPLACE THIS WITH YOUR OWN ACQUISITION CODE  ↓↓↓ #
PHASE_OFFSET = 0.25 * np.pi  # radians – difference between traces
FREQ_HZ = 1.2               # ECG‑like frequency for demo
NOISE = 0.05                # additive Gaussian noise (demo)

def get_next_sample(t: float) -> Tuple[float, float]:
    """Return next (human, organoid) ECG samples.

    Replace this function with the callback that supplies your
    *actual* ECG samples in real time. Both values should be floats
    representing millivolts at the same timestamp `t`.
    """
    human = np.sin(2 * np.pi * FREQ_HZ * t) + NOISE * np.random.randn()
    organoid = np.sin(2 * np.pi * FREQ_HZ * t + PHASE_OFFSET) + NOISE * np.random.randn()
    return human, organoid
# ↑↑↑  --------------------------------------------------  ↑↑↑ #


# ————————————————————————————————————————————————
# STREAMLIT PAGE SETUP
# ————————————————————————————————————————————————
st.set_page_config(page_title="Real‑Time ECG Dashboard", layout="wide")
st.title("Real‑Time ECG Dashboard")

# HOLD PLACE FOR FIGURE (single placeholder → faster refresh)
fig_placeholder = st.empty()

# Simple controls in sidebar
with st.sidebar:
    st.markdown("### Data controls")
    run_button = st.button("▶️ Start", key="start")
    stop_button = st.button("⏹ Stop", key="stop")
    st.markdown("---")
    st.caption("Replace `get_next_sample()` with your live data source.")

# Flag stored in session_state
if "running" not in st.session_state:
    st.session_state.running = False
if run_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# ————————————————————————————————————————————————
# MAIN LOOP (runs only while running == True)
# ————————————————————————————————————————————————
prev_time = time.perf_counter()
while st.session_state.running:
    # 1️⃣  Acquire new sample
    current_time = time.perf_counter()
    dt = current_time - prev_time
    prev_time = current_time
    next_t = t_buf[-1] + dt if t_buf else 0.0
    human_val, organoid_val = get_next_sample(next_t)

    # 2️⃣  Update buffers
    t_buf.append(next_t)
    human_buf.append(human_val)
    organoid_buf.append(organoid_val)

    # 3️⃣  Compute error (Bayesian update placeholder)
    error_arr = np.asarray(human_buf) - np.asarray(organoid_buf)

    # 4️⃣  Build figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Top‑left: Human ECG
    axs[0, 0].plot(t_buf, human_buf)
    axs[0, 0].set_title("Human ECG")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("mV")

    # Top‑right: Organoid ECG
    axs[0, 1].plot(t_buf, organoid_buf)
    axs[0, 1].set_title("Organoid ECG")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("mV")

    # Bottom‑left: Error / Bayesian update placeholder
    axs[1, 0].plot(t_buf, error_arr)
    axs[1, 0].set_title("Bayesian Update (Error)")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("mV")

    # Bottom‑right: Opentrons Camera placeholder
    axs[1, 1].text(0.5, 0.5, "Opentrons Camera\n(feed here)",
                   ha="center", va="center", fontsize=12)
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_title("Opentrons Camera")

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # 5️⃣  Render & sleep
    fig_placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)

    time.sleep(1.0 / FS)  # maintain sampling rate

# When stopped, show message
if not st.session_state.running:
    st.info("Click **Start** to begin streaming.")
