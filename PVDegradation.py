import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch.bootstrap import CircularBlockBootstrap  # Requires arch package

st.set_page_config(layout="wide", page_title="PV Degradation Analyzer")

# ---- Bootstrap Functions ----
def _make_time_series_bootstrap_samples(
    signal, model_fit, sample_nr=1000, block_length=90,
    decomposition_type='multiplicative', bootstrap_seed=None
):
    '''
    Generate bootstrap samples based on a time series signal and its model fit
    using circular block bootstrapping.
    '''
    if decomposition_type == 'multiplicative':
        residuals = signal / model_fit
    elif decomposition_type == 'additive':
        residuals = signal - model_fit
    else:
        raise ValueError(
            "decomposition_type needs to be either 'multiplicative' or 'additive'")

    bootstrap_samples = pd.DataFrame(
        index=signal.index, columns=range(sample_nr))

    bs = CircularBlockBootstrap(block_length, residuals, seed=bootstrap_seed)
    for b, bootstrapped_residuals in enumerate(bs.bootstrap(sample_nr)):
        if decomposition_type == 'multiplicative':
            bootstrap_samples.loc[:, b] = model_fit * bootstrapped_residuals[0][0].values
        elif decomposition_type == 'additive':
            bootstrap_samples.loc[:, b] = model_fit + bootstrapped_residuals[0][0].values

    return bootstrap_samples

def _construct_confidence_intervals(
    bootstrap_samples, fitting_function, exceedance_prob=95, confidence_level=68.2, **kwargs
):
    '''
    Construct confidence intervals based on bootstrap samples and a fitting function.
    '''
    metrics = bootstrap_samples.apply(fitting_function, **kwargs)
    half_ci = confidence_level / 2.0
    confidence_interval = np.percentile(metrics, [50.0 - half_ci, 50.0 + half_ci])
    exceedance_level = np.percentile(metrics, 100.0 - exceedance_prob)
    return confidence_interval, exceedance_level, metrics

# ---- Degradation Calculation Functions ----
def robust_median(series):
    return np.median(series.dropna())

def _mk_test(x, alpha=0.05):
    n = len(x)
    x = np.array(x)
    s = np.sum(np.triu(np.sign(-np.subtract.outer(x, x)), axis=1))
    
    unique_x = np.unique(x)
    g = len(unique_x)
    
    if n == g:  # no ties
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:  # ties exist
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n * (n - 1) * (2 * n + 5) +
                 np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)

    p = 2 * (1 - norm.cdf(abs(z)))
    h = abs(z) > norm.ppf(1 - alpha / 2)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return trend, h, p, z

def _degradation_CI(results, confidence_level):
    '''Monte Carlo estimation of degradation rate uncertainty'''
    sampled_normal = np.random.multivariate_normal(
        results.params, results.cov_params(), 10000
    )
    dist = sampled_normal[:, 1] / sampled_normal[:, 0]
    half_ci = confidence_level / 2.0
    Rd_CI = np.percentile(dist, [50.0 - half_ci, 50.0 + half_ci]) * 100.0
    return Rd_CI

def degradation_ols(energy_normalized, confidence_level=68.2):
    '''Ordinary Least Squares degradation calculation'''
    df = energy_normalized.to_frame(name='energy_normalized')
    
    day_diffs = (df.index - df.index[0])
    df['days'] = day_diffs / pd.Timedelta('1d')
    df['years'] = df.days / 365.0
    df = sm.add_constant(df)

    ols_model = sm.OLS(
        endog=df.energy_normalized,
        exog=df.loc[:, ['const', 'years']],
        hasconst=True,
        missing='drop'
    )
    results = ols_model.fit()
    b, m = results.params
    Rd_pct = 100.0 * m / b
    Rd_CI = _degradation_CI(results, confidence_level)

    calc_info = {
        'slope': m,
        'intercept': b,
        'rmse': np.sqrt(results.mse_resid),
        'slope_stderr': results.bse[1],
        'intercept_stderr': results.bse[0],
        'ols_result': results,
    }
    return (Rd_pct, Rd_CI, calc_info)

def degradation_classical_decomposition(energy_normalized, confidence_level=68.2):
    '''Classical decomposition degradation calculation'''
    df = energy_normalized.to_frame(name='energy_normalized')
    
    if df.dropna().index.freq is None:
        st.warning('Data may have gaps. Results might be unreliable.')
        df = df.asfreq('D').interpolate()

    day_diffs = (df.index - df.index[0])
    df['days'] = day_diffs / pd.Timedelta('1d')
    df['years'] = df.days / 365.0

    energy_ma = df['energy_normalized'].rolling('365d', center=True).mean()
    has_full_year = (df["years"] >= df["years"].iloc[0] + 0.5) & (
        df["years"] <= df["years"].iloc[-1] - 0.5
    )
    energy_ma[~has_full_year] = np.nan
    df['energy_ma'] = energy_ma
    df = sm.add_constant(df)

    ols_model = sm.OLS(
        endog=df.energy_ma.dropna(),
        exog=df.loc[df.energy_ma.notna(), ['const', 'years']],
        hasconst=True
    )
    results = ols_model.fit()
    b, m = results.params
    Rd_pct = 100.0 * m / b
    Rd_CI = _degradation_CI(results, confidence_level)
    trend, h, p, z = _mk_test(df.energy_ma.dropna())

    calc_info = {
        'slope': m,
        'intercept': b,
        'rmse': np.sqrt(results.mse_resid),
        'slope_stderr': results.bse[1],
        'intercept_stderr': results.bse[0],
        'ols_result': results,
        'series': df.energy_ma,
        'mk_test_trend': trend,
        'mk_p_value': p
    }
    return (Rd_pct, Rd_CI, calc_info)

def degradation_year_on_year(energy_normalized, recenter=True,
                            exceedance_prob=95, confidence_level=68.2,
                            uncertainty_method='simple', block_length=30):
    '''Year-on-Year degradation calculation with bootstrap options'''
    energy_normalized = energy_normalized.sort_index()
    energy_normalized.name = 'energy'
    energy_normalized.index.name = 'dt'
    
    # Detect frequency and validate data length
    if energy_normalized.index.inferred_freq is not None:
        step = pd.tseries.frequencies.to_offset(energy_normalized.index.inferred_freq)
    else:
        step = energy_normalized.index.to_series().diff().median()

    if energy_normalized.index[-1] < energy_normalized.index[0] + pd.DateOffset(years=2) - step:
        raise ValueError('Must provide at least two years of normalized energy')

    # Recenter data
    if recenter:
        start = energy_normalized.index[0]
        oneyear = start + pd.Timedelta('364d')
        renorm = robust_median(energy_normalized[start:oneyear])
    else:
        renorm = 1.0

    energy_normalized = energy_normalized / renorm
    df = pd.DataFrame({'energy': energy_normalized})
    df['dt_shifted'] = df.index + pd.DateOffset(years=1)
    
    merged = pd.merge_asof(
        df[['energy']].reset_index(),
        df.reset_index().rename(columns={'energy': 'energy_right', 'dt': 'dt_right'}),
        left_on='dt',
        right_on='dt_shifted',
        tolerance=pd.Timedelta('8D'),
        suffixes=('', '_right')
    )
    
    merged['time_diff_years'] = (merged.dt - merged.dt_right) / pd.Timedelta('365d')
    merged['yoy'] = 100.0 * (merged.energy - merged.energy_right) / merged.time_diff_years
    yoy_result = merged['yoy'].dropna()
    
    if yoy_result.empty:
        st.error("No year-over-year data pairs found!")
        return None, None, None
    
    Rd_pct = yoy_result.median()
    
    # Simple bootstrap
    if uncertainty_method == 'simple':
        n = len(yoy_result)
        reps = 10000
        xb = np.random.choice(yoy_result, (n, reps), replace=True)
        mb = np.median(xb, axis=0)
        half_ci = confidence_level / 2.0
        Rd_CI = np.percentile(mb, [50.0 - half_ci, 50.0 + half_ci])
        P_level = np.percentile(mb, 100.0 - exceedance_prob)
        
        calc_info = {
            'YoY_values': yoy_result,
            'renormalizing_factor': renorm,
            'bootstrap_medians': mb,
            'exceedance_level': P_level
        }
        return (Rd_pct, Rd_CI, calc_info)
    
    # Circular block bootstrap
    elif uncertainty_method == 'circular_block':
        reps = 1000
        N = len(energy_normalized)
        numeric_index = np.arange(N)
        days_per_index = (df.index[-1] - df.index[0]).days / N
        degradation_trend = 1 + (Rd_pct / 100 / 365.0 * numeric_index * days_per_index)
        degradation_trend = pd.Series(index=df.index, data=degradation_trend)

        bootstrap_samples = _make_time_series_bootstrap_samples(
            energy_normalized, degradation_trend,
            sample_nr=reps, block_length=block_length
        )

        def yoy_wrapper(series):
            return degradation_year_on_year(
                series, recenter=False, 
                uncertainty_method='none',
                confidence_level=confidence_level
            )[0]

        Rd_CI, exceedance_level, bootstrap_rates = _construct_confidence_intervals(
            bootstrap_samples, yoy_wrapper,
            exceedance_prob=exceedance_prob,
            confidence_level=confidence_level
        )

        calc_info = {
            'renormalizing_factor': renorm,
            'exceedance_level': exceedance_level,
            'bootstrap_rates': bootstrap_rates
        }
        return (Rd_pct, Rd_CI, calc_info)
    
    else:  # No uncertainty calculation
        return (Rd_pct, None, {'renormalizing_factor': renorm})

# ---- Streamlit UI ----
st.title("📉 Photovoltaic Degradation Analyzer")
st.markdown("""
Upload normalized PV system output data to calculate degradation rates using:
- **OLS Regression**: Linear trend fitting
- **Classical Decomposition**: Moving average trend
- **Year-on-Year**: Paired annual comparisons with bootstrap options
""")

with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    confidence_level = st.slider("Confidence Level (%)", 50, 99, 68, step=1)
    method = st.radio("Analysis Method", 
                      ("OLS Regression", 
                       "Classical Decomposition", 
                       "Year-on-Year"))
    
    # Year-on-Year specific options
    if method == "Year-on-Year":
        uncertainty_method = st.radio("Uncertainty Method",
                                     ("Simple Bootstrap", 
                                      "Circular Block Bootstrap"))
        if uncertainty_method == "Circular Block Bootstrap":
            block_length = st.slider("Block Length (days)", 1, 180, 30)
        exceedance_prob = st.slider("Exceedance Probability (%)", 80, 99, 95)
    
    st.subheader("Data Requirements")
    st.markdown("""
    - Daily or lower frequency data
    - Datetime index column
    - Column with normalized energy values
    - At least 2 years of data
    """)
    st.divider()
    st.markdown("**Dependencies**: statsmodels, scipy, arch, pandas, numpy")
    st.markdown("Developed by PV Analytics Team")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        energy_col = st.selectbox("Select energy column", df.columns)
        
        if st.checkbox("Show raw data"):
            st.dataframe(df.head(), use_container_width=True)
            
        energy_series = df[energy_col].dropna()
        
        if len(energy_series) < 730:  # 2 years
            st.warning("At least 2 years of data recommended for accurate results")
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Preview")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(energy_series.index, energy_series, 'b-', alpha=0.7)
            ax.set_title("Normalized Energy Time Series")
            ax.set_ylabel("Normalized Output")
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Annual Trends")
            annual = energy_series.resample('Y').mean()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(annual.index.year, annual.values, width=0.8)
            ax.set_title("Annual Average Normalized Output")
            ax.set_ylabel("Normalized Output")
            ax.set_xlabel("Year")
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        
        # Run selected analysis
        st.divider()
        st.subheader(f"Results: {method}")
        
        if method == "OLS Regression":
            Rd, CI, info = degradation_ols(energy_series, confidence_level)
            st.metric("Degradation Rate", f"{Rd:.3f} %/year", 
                      delta=f"CI: [{CI[0]:.3f}, {CI[1]:.3f}]")
            
            st.write("**Regression Statistics**")
            st.dataframe({
                "Parameter": ["Slope", "Intercept", "RMSE"],
                "Value": [info['slope'], info['intercept'], info['rmse']]
            })
            
            # Plot results
            fig, ax = plt.subplots(figsize=(10, 5))
            years = (energy_series.index - energy_series.index[0]).days / 365.0
            ax.scatter(years, energy_series, alpha=0.5, label="Data")
            
            # Regression line
            reg_line = info['intercept'] + info['slope'] * years
            ax.plot(years, reg_line, 'r-', lw=2, label="OLS Fit")
            
            ax.set_title("OLS Regression Analysis")
            ax.set_xlabel("Years")
            ax.set_ylabel("Normalized Output")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        elif method == "Classical Decomposition":
            Rd, CI, info = degradation_classical_decomposition(energy_series, confidence_level)
            st.metric("Degradation Rate", f"{Rd:.3f} %/year", 
                      delta=f"CI: [{CI[0]:.3f}, {CI[1]:.3f}]")
            
            st.write(f"**Mann-Kendall Test:** Trend is **{info['mk_test_trend']}** " 
                     f"(p-value: {info['mk_p_value']:.4f})")
            
            # Plot results
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(energy_series.index, energy_series, alpha=0.3, label="Daily Data")
            ax.plot(info['series'].index, info['series'], 'r-', lw=2, label="365-day Moving Avg")
            
            # Trend line
            years = (info['series'].dropna().index - info['series'].dropna().index[0]).days / 365.0
            trend_line = info['intercept'] + info['slope'] * years
            ax.plot(info['series'].dropna().index, trend_line, 'g--', lw=2, label="Trend")
            
            ax.set_title("Classical Decomposition Analysis")
            ax.set_ylabel("Normalized Output")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        elif method == "Year-on-Year":
            uncertainty_param = 'simple' if uncertainty_method == "Simple Bootstrap" else 'circular_block'
            
            Rd, CI, info = degradation_year_on_year(
                energy_series,
                confidence_level=confidence_level,
                exceedance_prob=exceedance_prob,
                uncertainty_method=uncertainty_param,
                block_length=block_length if uncertainty_param == 'circular_block' else 30
            )
            
            st.metric("Degradation Rate", f"{Rd:.3f} %/year")
            
            if CI is not None:
                st.metric(f"Confidence Interval ({confidence_level}%)", 
                         f"[{CI[0]:.3f}, {CI[1]:.3f}]")
            
            if 'exceedance_level' in info:
                st.metric(f"Exceedance Level ({exceedance_prob}%)", 
                         f"{info.get('exceedance_level', 0):.3f} %/year")
            
            st.write(f"Renormalizing factor: {info.get('renormalizing_factor', 1.0):.4f}")
            
            # Plot results
            fig, ax = plt.subplots(figsize=(10, 5))
            
            if uncertainty_param == 'simple':
                ax.hist(info.get('bootstrap_medians', []), bins=30, alpha=0.7)
                if CI is not None:
                    ax.axvline(CI[0], color='g', linestyle='dashed', linewidth=1, label='CI Lower')
                    ax.axvline(CI[1], color='g', linestyle='dashed', linewidth=1, label='CI Upper')
                ax.axvline(Rd, color='r', linestyle='dashed', linewidth=2, label='Median')
                ax.set_title("Bootstrap Degradation Rate Distribution")
                ax.set_xlabel("%/year")
                ax.legend()
                
            else:  # Circular block bootstrap
                if 'bootstrap_rates' in info:
                    ax.hist(info['bootstrap_rates'], bins=30, alpha=0.7, color='orange')
                    if CI is not None:
                        ax.axvline(CI[0], color='g', linestyle='dashed', linewidth=1, label='CI Lower')
                        ax.axvline(CI[1], color='g', linestyle='dashed', linewidth=1, label='CI Upper')
                    ax.axvline(Rd, color='r', linestyle='dashed', linewidth=2, label='Median')
                    ax.set_title("Circular Block Bootstrap Distribution")
                    ax.set_xlabel("%/year")
                    ax.legend()
            
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.info("Please upload a CSV file to begin analysis")
    st.image("https://images.unsplash.com/photo-1613665813446-82a78c468a1d?auto=format&fit=crop&w=1200", 
             caption="Photovoltaic System - Image by Unsplash")