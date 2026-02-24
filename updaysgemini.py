import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="NSE Momentum & RS Scanner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CONSTANTS & UNIVERSES
# ================================
BENCHMARK = "^NSEI"  # Nifty 50

NSE_SECTORS = {
    "Railway": ["IRFC.NS","IRCTC.NS", "RVNL.NS", "CONCOR.NS", "IRCON.NS","RAILTEL.NS","TITAGARH.NS","JWL.NS","BEML.NS","RITES.NS","TEXRAIL.NS", "RKFORGE.NS"],
    "Defense": ["HAL.NS", "BEL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "BDL.NS", "DATA_PATT.NS"],
    "Finance/Banking": ["BAJFINANCE.NS", "SBIN.NS", "HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS"],
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS"],
    "Power/Energy": ["NTPC.NS", "TATAPOWER.NS", "POWERGRID.NS", "ADANIPOWER.NS", "RELIANCE.NS", "ONGC.NS", "JSWENERGY.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "ASHOKLEY.NS"],
    "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "MAXHEALTH.NS", "APOLLOHOSP.NS"],
    "Realty": ["DLF.NS", "PRESTIGE.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "MACROTECH.NS"]
}

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .main-header { font-size: 2.2rem; color: #0E1117; font-weight: 800; margin-bottom: 0.5rem; }
    .status-tag { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ================================
# DATA ENGINE
# ================================
@st.cache_data(ttl=3600)
def fetch_all_data(tickers, start_date):
    tickers = list(set(tickers + [BENCHMARK]))
    data = yf.download(tickers, start=start_date, progress=False, group_by='ticker')
    return data

def calculate_momentum_metrics(data, ticker, benchmark_data):
    try:
        # Get Stock Close and Benchmark Close
        df_close = data[ticker]['Close'].dropna()
        bench_close = benchmark_data['Close'].dropna()
        
        # ALIGNMENT: Ensure both series have the same dates
        combined = pd.concat([df_close, bench_close], axis=1, join='inner').dropna()
        combined.columns = ['Stock', 'Bench']
        
        if len(combined) < 200: return None
        
        curr_price = combined['Stock'].iloc[-1]
        
        # Moving Averages
        sma_50 = combined['Stock'].rolling(50).mean().iloc[-1]
        sma_150 = combined['Stock'].rolling(150).mean().iloc[-1]
        sma_200 = combined['Stock'].rolling(200).mean().iloc[-1]
        
        # Mansfield Relative Strength (RS)
        ratio = combined['Stock'] / combined['Bench']
        base_line = ratio.rolling(200).mean()
        rs_score = ((ratio / base_line) - 1) * 10 
        curr_rs = rs_score.iloc[-1]
        
        # Momentum
        high_52w = combined['Stock'].rolling(252).max().iloc[-1]
        dist_from_high = ((curr_price - high_52w) / high_52w) * 100
        roc_3m = ((curr_price - combined['Stock'].iloc[-63]) / combined['Stock'].iloc[-63]) * 100
        
        # Minervini Stage 2 logic
        is_stage_2 = (
            curr_price > sma_150 > sma_200 and 
            curr_price > sma_50 and
            dist_from_high > -30 # Loosened to 30% from high
        )
        
        setup = "NEUTRAL"
        if is_stage_2:
            if curr_rs > 1.0 and roc_3m > 15: setup = "SUPER MOMENTUM"
            elif curr_rs > 0: setup = "BULLISH"
        elif curr_price < sma_200:
            setup = "BEARISH"

        return {
            'Stock': ticker.replace('.NS', ''),
            'Price': round(curr_price, 2),
            'RS Score': round(curr_rs, 2),
            '3M ROC %': round(roc_3m, 1),
            'Dist 52W High %': round(dist_from_high, 1),
            'Stage 2': "✅" if is_stage_2 else "❌",
            'Trend': setup
        }
    except Exception as e:
        return None

# ================================
# SIDEBAR
# ================================
st.sidebar.header("🛠️ Momentum Controls")

selected_sectors = st.sidebar.multiselect(
    "Select Sectors", list(NSE_SECTORS.keys()), default=list(NSE_SECTORS.keys())[:3]
)

rs_threshold = st.sidebar.slider("Min RS Score (Outperformance)", -2.0, 5.0, 0.0, 0.5)
roc_filter = st.sidebar.checkbox("Filter by 3M Momentum (>15%)", value=True)

# ================================
# APP LOGIC
# ================================
st.markdown('<div class="main-header">🚀 NSE High-Momentum Scanner</div>', unsafe_allow_html=True)
st.write(f"Scanning for stocks in **Stage 2 Uptrends** with positive **Relative Strength** vs {BENCHMARK}")

if st.sidebar.button("Run Global Scan", type="primary"):
    # Prepare tickers
    tickers_to_scan = []
    for s in selected_sectors:
        tickers_to_scan.extend(NSE_SECTORS[s])
    
    # Fetch Data
    with st.spinner("Fetching historical data..."):
        start_date = datetime.now() - timedelta(days=500)
        raw_data = fetch_all_data(tickers_to_scan, start_date)
        bench_data = raw_data[BENCHMARK]
        
    # Process
    results = []
    for s in selected_sectors:
        for t in NSE_SECTORS[s]:
            if t in raw_data:
                m = calculate_momentum_metrics(raw_data, t, bench_data)
                if m:
                    m['Sector'] = s
                    results.append(m)
    
    if results:
        full_df = pd.DataFrame(results)
        
        # Apply Filters
        mask = (full_df['RS Score'] >= rs_threshold)
        if roc_filter:
            mask = mask & (full_df['3M ROC %'] > 15)
        
        filtered_df = full_df[mask]
        
        if filtered_df.empty:
            st.warning("⚠️ No stocks match your current RS and ROC filters. Showing all Stage 2 stocks instead.")
            filtered_df = full_df[full_df['Stage 2'] == "✅"]
            
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Leaders Found", len(filtered_df))
        m2.metric("Avg RS Score", round(filtered_df['RS Score'].mean(), 2))
        m3.metric("Top Performer", filtered_df.iloc[filtered_df['3M ROC %'].argmax()]['Stock'] if not filtered_df.empty else "N/A")
        m4.metric("Market Breadth", f"{len(full_df[full_df['Trend'] != 'BEARISH'])} / {len(full_df)}")

        # --- UI OPTIMIZATION: SECTOR DRILL-DOWN ---
        st.markdown("---")
        
        st.subheader("🔥 Momentum Heatmap (RS vs ROC)")
        fig = px.scatter(
            filtered_df, x="RS Score", y="3M ROC %", 
            size="Price", color="Sector", hover_name="Stock",
            text="Stock", height=550,
            labels={"RS Score": "Mansfield Relative Strength", "3M ROC %": "3-Month Return (%)"}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col_rank, col_detail = st.columns([1, 1])

        with col_rank:
            st.subheader("📊 Sector Momentum Rank")
            # Sort ascending=True so highest values appear at the top of the horizontal bar chart
            sec_rank = full_df.groupby('Sector')['RS Score'].mean().sort_values(ascending=True).reset_index()
            
            # Create the horizontal chart
            fig_sec = px.bar(
                sec_rank, x='RS Score', y='Sector', orientation='h', 
                color='RS Score', color_continuous_scale='RdYlGn',
                height=350,
                text='RS Score'
            )
            fig_sec.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_sec.update_layout(showlegend=False, yaxis_title="", xaxis_title="Average RS Score", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_sec, use_container_width=True)

        with col_detail:
            st.subheader("🎯 Sector Deep Dive")
            # This is the "Click" equivalent in Streamlit
            sector_list_order = full_df.groupby('Sector')['RS Score'].mean().sort_values(ascending=False).index.tolist()
            target_sector = st.selectbox(
                "Select a Sector to see its Momentum Leaders:", 
                ["All Sectors"] + sector_list_order
            )
            
            # Filter logic for drill-down
            if target_sector == "All Sectors":
                display_df = full_df[full_df['Stage 2'] == "✅"]
            else:
                display_df = full_df[full_df['Sector'] == target_sector]

            # Mini-stats for the selected sector
            s_m1, s_m2 = st.columns(2)
            s_m1.metric("Stage 2 Count", len(display_df[display_df['Stage 2'] == "✅"]))
            if not display_df.empty:
                s_m2.metric("Sector Avg RS", round(display_df['RS Score'].mean(), 2))
            else:
                s_m2.metric("Sector Avg RS", "N/A")

        # --- UPDATED LEADERBOARD ---
        st.subheader(f"🏆 {target_sector} Leaderboard")
        
        # Apply the specific trend coloring
        def color_trend(val):
            if val == 'SUPER MOMENTUM': return 'background-color: #10B981; color: white'
            if val == 'BULLISH': return 'background-color: #3B82F6; color: white'
            if val == 'BEARISH': return 'background-color: #EF4444; color: white'
            return ''

        # Sort by RS Score to show leaders at the top
        final_df = display_df.sort_values('RS Score', ascending=False)
        
        st.dataframe(
            final_df.style.format({
                'Price': '₹{:.2f}',
                'RS Score': '{:.2f}',
                '3M ROC %': '{:.1f}%',
                'Dist 52W High %': '{:.1f}%'
            }).map(color_trend, subset=['Trend']),
            use_container_width=True,
            height=400
        )

        # --- BREADTH ANALYSIS CHART ---
        st.subheader("📈 Market Breadth Trend")
        # Visualizing the ratio of Stage 2 stocks across sectors
        breadth_df = full_df.groupby('Sector')['Stage 2'].apply(lambda x: (x == "✅").sum()).reset_index()
        fig_breadth = px.pie(breadth_df, values='Stage 2', names='Sector', hole=.4,
                             title="Distribution of Stage 2 Leaders",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_breadth, use_container_width=True)

        # Trading Insights Logic
        st.info("💡 **Strategy Guide:** Focus on stocks with '✅' in Stage 2. These have the structural support of institutions. 'Super Momentum' indicates the stock is accelerating away from the benchmark.")
    else:
        st.error("No data found for selected criteria.")

else:
    st.image("https://images.unsplash.com/photo-1611974717482-980096c6183c?auto=format&fit=crop&q=80&w=1000", caption="Ready to scan for institutional footprints.")
    st.warning("Click 'Run Global Scan' in the sidebar to begin analysis.")

# ================================
# FOOTER
# ================================
st.markdown("---")
st.caption("Custom Momentum Scanner | Built for NSE Swing Traders | Data source: Yahoo Finance")
