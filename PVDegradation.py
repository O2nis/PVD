import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from degradation import (
    degradation_ols,
    degradation_classical_decomposition,
    degradation_year_on_year
)
from filtering import (
    normalized_filter,
    poa_filter,
    tcell_filter,
    clearsky_filter,
    clip_filter,
    two_way_window_filter,
    hampel_filter,
    insolation_filter,
    directional_tukey_filter
)

# App configuration
st.set_page_config(page_title="PV Degradation Analyzer", layout="wide")
st.title("Photovoltaic System Degradation Analysis")

# Sidebar - Data Upload and Configuration
with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        st.success("Data successfully loaded!")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Column selection
        energy_col = st.selectbox("Energy Column", df.columns)
        poa_col = st.selectbox("POA Irradiance Column", df.columns, index=1)
        temp_col = st.selectbox("Cell Temperature Column", df.columns, index=2)
        clearsky_col = st.selectbox("Clearsky POA Column (optional)", 
                                   [""] + list(df.columns), index=0)
        power_col = st.selectbox("AC Power Column (for clipping)", 
                                [""] + list(df.columns), index=0)
    
    st.header("Analysis Settings")
    degradation_method = st.selectbox(
        "Degradation Method",
        ["OLS Regression", "Classical Decomposition", "Year-on-Year"]
    )
    confidence_level = st.slider("Confidence Level", 50, 99, 68)
    recenter = st.checkbox("Recent Year-on-Year Data", value=True)

# Main content area
if not uploaded_file:
    st.info("Please upload a CSV file to begin analysis")
    st.stop()

# Data Filtering Section
st.header("Data Filtering")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Filters")
    norm_low = st.number_input("Normalized Energy Low", value=0.01)
    norm_high = st.number_input("Normalized Energy High", value=1.5)
    poa_low = st.number_input("POA Irradiance Low (W/m²)", value=200)
    poa_high = st.number_input("POA Irradiance High (W/m²)", value=1200)
    temp_low = st.number_input("Cell Temp Low (°C)", value=-50)
    temp_high = st.number_input("Cell Temp High (°C)", value=110)

with col2:
    st.subheader("Advanced Filters")
    use_clearsky = st.checkbox("Apply Clearsky Filter")
    use_clipping = st.checkbox("Apply Clipping Filter")
    post_agg_filter = st.selectbox(
        "Post-Aggregation Filter",
        ["None", "Two-way Window", "Hampel", "Directional Tukey", "Insolation"]
    )
    
    if use_clearsky and clearsky_col:
        clearsky_model = st.radio("Clearsky Model", ["pvlib", "csi"])
        csi_threshold = st.slider("CSI Threshold", 0.05, 0.5, 0.15)
    
    if use_clipping and power_col:
        clip_model = st.radio("Clipping Model", ["logic", "quantile", "xgboost"])

# Apply filters
filter_mask = pd.Series(True, index=df.index)

# Basic filters
filter_mask &= normalized_filter(df[energy_col], norm_low, norm_high)
filter_mask &= poa_filter(df[poa_col], poa_low, poa_high)
filter_mask &= tcell_filter(df[temp_col], temp_low, temp_high)

# Advanced filters
if use_clearsky and clearsky_col:
    if clearsky_model == "csi":
        filter_mask &= clearsky_filter(
            df[poa_col], df[clearsky_col], model="csi", threshold=csi_threshold
        )
    else:
        filter_mask &= clearsky_filter(
            df[poa_col], df[clearsky_col], model="pvlib"
        )

if use_clipping and power_col:
    filter_mask &= clip_filter(df[power_col], model=clip_model)

# Apply filters to data
filtered_energy = df[energy_col][filter_mask]

# Post-aggregation filtering
if post_agg_filter != "None":
    daily_energy = filtered_energy.resample('D').mean()
    
    if post_agg_filter == "Two-way Window":
        mask = two_way_window_filter(daily_energy)
    elif post_agg_filter == "Hampel":
        mask = hampel_filter(daily_energy)
    elif post_agg_filter == "Directional Tukey":
        mask = directional_tukey_filter(daily_energy)
    elif post_agg_filter == "Insolation" and poa_col in df:
        daily_poa = df[poa_col][filter_mask].resample('D').sum()
        mask = insolation_filter(daily_poa)
    
    filtered_energy = daily_energy[mask].dropna()

# Degradation Analysis
st.header("Degradation Analysis")

if st.button("Run Degradation Analysis"):
    with st.spinner("Calculating degradation..."):
        try:
            if degradation_method == "OLS Regression":
                rd, ci, info = degradation_ols(
                    filtered_energy, 
                    confidence_level=confidence_level
                )
            elif degradation_method == "Classical Decomposition":
                rd, ci, info = degradation_classical_decomposition(
                    filtered_energy,
                    confidence_level=confidence_level
                )
            else:  # Year-on-Year
                rd, ci, info = degradation_year_on_year(
                    filtered_energy,
                    recenter=recenter,
                    confidence_level=confidence_level
                )
            
            # Display results
            st.success(f"Degradation Rate: {rd:.2f} %/year")
            st.info(f"Confidence Interval ({confidence_level}%): "
                    f"[{ci[0]:.2f}, {ci[1]:.2f}] %/year")
            
            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot filtered data
            if post_agg_filter != "None":
                ax.plot(daily_energy.index, daily_energy, 'o', 
                        alpha=0.3, label="Original Data")
                ax.plot(filtered_energy.index, filtered_energy, 'o',
                        label="Filtered Data")
            else:
                ax.plot(filtered_energy.index, filtered_energy, 'o',
                        label="Normalized Energy")
            
            # Plot trend line
            if degradation_method != "Year-on-Year":
                years = (filtered_energy.index - filtered_energy.index[0]).days / 365.0
                trend = info['intercept'] + info['slope'] * years
                ax.plot(filtered_energy.index, trend, 'r-', linewidth=2,
                        label="Degradation Trend")
            
            ax.set_title("Degradation Analysis")
            ax.set_ylabel("Normalized Energy")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)
            
            # Show additional info
            with st.expander("Detailed Results"):
                st.write(info)
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# Data summary
st.subheader("Filtered Data Summary")
col1, col2 = st.columns(2)

with col1:
    st.metric("Original Data Points", len(df))
    st.metric("Filtered Data Points", len(filtered_energy))

with col2:
    st.metric("Data Retention", 
              f"{len(filtered_energy)/len(df)*100:.1f}%")
    st.metric("Time Span", 
              f"{(filtered_energy.index[-1] - filtered_energy.index[0]).days / 365:.1f} years")

st.line_chart(filtered_energy)
