import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ==========================================
# 1. ANALYZER & METADATA SIMULATION ENGINE
# ==========================================
def analyze_intra_modes(codec, video_profile, grid_size=(16, 16)):
    start_time = time.time()
    
    if codec == "AV1":
        modes = ["DC", "Vertical", "Horizontal", "Smooth", "Paeth"] + [f"Angular {i}" for i in range(5, 61)]
        weights = [0.25, 0.20, 0.15, 0.10, 0.05] + [0.007] * 56
    else: # VVC
        modes = ["Planar", "DC"] + [f"Angular {i}" for i in range(2, 68)]
        weights = [0.20, 0.20] + [0.0093] * 66
        
    weights = np.array(weights) / np.sum(weights)
    np.random.seed(42 if video_profile == "Low Motion / Flat" else 99)
    
    if video_profile == "Low Motion / Flat":
        if codec == "AV1":
            grid_data = np.random.choice([0, 1, 2, 3], size=grid_size, p=[0.5, 0.2, 0.2, 0.1])
        else:
            grid_data = np.random.choice([0, 1, 2], size=grid_size, p=[0.4, 0.4, 0.2])
    else:
        grid_data = np.random.choice(len(modes), size=grid_size, p=weights)
        
    unique, counts = np.unique(grid_data, return_counts=True)
    stats = {modes[int(u)]: int(c) for u, c in zip(unique, counts)}
    
    latency = (time.time() - start_time) * 1000 + np.random.uniform(5, 15)
    bitrate_est = np.random.uniform(1.2, 2.5) if video_profile == "High Texture" else np.random.uniform(0.4, 0.9)
    psnr_est = np.random.uniform(38.5, 42.0) if codec == "VVC" else np.random.uniform(37.0, 40.5)

    return grid_data, stats, modes, {"latency": latency, "bitrate": bitrate_est, "psnr": psnr_est}

# ==========================================
# 2. INTERACTIVE DEMO USER INTERFACE
# ==========================================
st.set_page_config(page_title="Intra Prediction Mode Analyzer", layout="wide")

st.title(" Intra Prediction Mode Distribution Analysis Dashboard")
st.markdown("Analyze spatial redundancies across codec definitions via structural mapping visualizations.")
st.markdown("---")

st.sidebar.header(" System Configurations")
selected_codec = st.sidebar.radio("Select Target Codec Standard:", ["AV1", "VVC"])
video_profile = st.sidebar.selectbox("Select Video Profile Texture:", ["Low Motion / Flat", "High Texture / Complex Details"])
grid_resolution = st.sidebar.slider("Analysis Grid Mesh Granularity:", 8, 32, 16, step=8)

st.sidebar.markdown("---")
st.sidebar.markdown("### Interactive Calibration Sliders")
qp_value = st.sidebar.slider("Quantization Parameter (QP):", 10, 51, 32)

grid_data, stats, mode_labels, metrics = analyze_intra_modes(
    codec=selected_codec, 
    video_profile=video_profile, 
    grid_size=(grid_resolution, grid_resolution)
)

col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Processing Latency", value=f"{metrics['latency']:.2f} ms")
col2.metric(label="Estimated Bitrate Value", value=f"{metrics['bitrate']:.3f} Mbps")
col3.metric(label="Target PSNR Assessment", value=f"{metrics['psnr']:.2f} dB")
col4.metric(label="Quantization Setting", value=f"QP {qp_value}")

st.markdown("---")
left_chart, right_chart = st.columns([1, 1])

with left_chart:
    st.subheader("Spatial Mode Assignment Heatmap")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(grid_data, cmap="viridis", cbar_kws={'label': 'Prediction Mode Index ID'}, linewidths=0.3, ax=ax)
    ax.set_xlabel("Frame Block Coordinates (X)")
    ax.set_ylabel("Frame Block Coordinates (Y)")
    st.pyplot(fig)

with right_chart:
    st.subheader(" Mode Distribution Frequency Statistics")
    df_stats = pd.DataFrame(list(stats.items()), columns=['Prediction Mode', 'Occurrence Count'])
    df_stats = df_stats.sort_values(by='Occurrence Count', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    # FIX: Explicitly mapped hue to 'Prediction Mode' and disabled the redundant legend to stop deprecation warnings
    sns.barplot(data=df_stats, x='Occurrence Count', y='Prediction Mode', hue='Prediction Mode', palette="mako", legend=False, ax=ax)
    ax.set_xlabel("Frequency Volume (Block Count)")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

st.subheader(" Comprehensive Extraction Metrics Register")
df_full = pd.DataFrame(list(stats.items()), columns=['Mode Classification Name', 'Total Block Allocations'])
df_full['Percentage Allocation (%)'] = ((df_full['Total Block Allocations'] / (grid_resolution**2)) * 100).round(2)
st.dataframe(df_full.sort_values(by='Total Block Allocations', ascending=False), use_container_width=True)
