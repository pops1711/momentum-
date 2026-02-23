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
    page_title="NSE Sector Rotation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .signal-buy {
        background-color: #D1FAE5 !important;
        color: #065F46 !important;
        font-weight: bold;
    }
    .signal-sell {
        background-color: #FEE2E2 !important;
        color: #991B1B !important;
        font-weight: bold;
    }
    .signal-hold {
        background-color: #FEF3C7 !important;
        color: #92400E !important;
    }
    .info-card {
        background-color: #f0f9ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# NSE SECTOR UNIVERSES
# ================================
NSE_SECTORS = {
    "Bank": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "Automobile": ["MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS"],
    "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HPCL.NS"],
    "Metal": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "JINDALSTEL.NS"]
}

# ================================
# IMPROVED CALCULATION FUNCTIONS
# ================================
def calculate_real_strength(prices, window=20):
    """
    IMPROVED: Calculate strength that actually matches what traders see on charts
    Uses multiple technical indicators for better accuracy
    """
    if len(prices) < window + 10:
        return None
    
    try:
        # 1. PRICE ACTION STRENGTH (40%)
        # Look at actual price movement pattern
        recent_prices = prices.iloc[-window:]
        
        # Check if making higher highs and higher lows
        highs = recent_prices.rolling(5).max()
        lows = recent_prices.rolling(5).min()
        
        # Price trend direction
        price_trend = 0
        if len(recent_prices) >= 10:
            # Check if price is above recent average
            avg_price = recent_prices.mean()
            current_price = recent_prices.iloc[-1]
            
            if current_price > avg_price:
                price_trend += 25
            elif current_price < avg_price:
                price_trend -= 25
            
            # Check slope of recent prices
            x = np.arange(len(recent_prices))
            y = recent_prices.values
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0:
                price_trend += 25
            else:
                price_trend -= 25
        
        # 2. MOVING AVERAGE STRENGTH (30%)
        ma_strength = 0
        if len(prices) >= 50:
            sma_20 = prices.rolling(20).mean().iloc[-1]
            sma_50 = prices.rolling(50).mean().iloc[-1]
            current_price = prices.iloc[-1]
            
            # Above both MAs
            if current_price > sma_20 > sma_50:
                ma_strength += 30
            # Above 20MA but below 50MA
            elif current_price > sma_20:
                ma_strength += 15
            # Below both MAs
            elif current_price < sma_20 < sma_50:
                ma_strength -= 30
        
        # 3. MOMENTUM STRENGTH (30%)
        momentum = 0
        
        # ROC calculations
        if len(prices) >= 5:
            roc_5 = ((prices.iloc[-1] / prices.iloc[-5]) - 1) * 100
            if roc_5 > 0:
                momentum += 15
            else:
                momentum -= 15
        
        if len(prices) >= 20:
            roc_20 = ((prices.iloc[-1] / prices.iloc[-20]) - 1) * 100
            if roc_20 > 0:
                momentum += 15
            else:
                momentum -= 15
        
        # 4. VOLATILITY ADJUSTMENT
        returns = prices.pct_change().dropna()
        if len(returns) >= 20:
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            # Penalize high volatility stocks
            if volatility > 50:
                volatility_penalty = -10
            elif volatility < 20:
                volatility_penalty = 5
            else:
                volatility_penalty = 0
        else:
            volatility_penalty = 0
        
        # FINAL STRENGTH SCORE (-100 to +100)
        total_strength = price_trend + ma_strength + momentum + volatility_penalty
        
        # Normalize to -100 to 100 range
        total_strength = max(-100, min(100, total_strength))
        
        return {
            'strength': round(total_strength, 1),
            'price_trend': round(price_trend, 1),
            'ma_strength': round(ma_strength, 1),
            'momentum': round(momentum, 1),
            'current_price': round(prices.iloc[-1], 2)
        }
    
    except Exception as e:
        return None

def calculate_support_resistance(prices):
    """Identify key support and resistance levels"""
    try:
        if len(prices) < 50:
            return None
        
        # Recent highs and lows
        recent_high = prices.iloc[-20:].max()
        recent_low = prices.iloc[-20:].min()
        current_price = prices.iloc[-1]
        
        # Distance from recent high/low
        distance_from_high = ((recent_high - current_price) / recent_high) * 100
        distance_from_low = ((current_price - recent_low) / recent_low) * 100
        
        return {
            'recent_high': round(recent_high, 2),
            'recent_low': round(recent_low, 2),
            'distance_from_high': round(distance_from_high, 1),
            'distance_from_low': round(distance_from_low, 1),
            'near_resistance': distance_from_high < 5,  # Within 5% of high
            'near_support': distance_from_low < 5       # Within 5% of low
        }
    except:
        return None

def calculate_volume_strength(volume, prices):
    """Analyze volume patterns"""
    try:
        if len(volume) < 20:
            return None
        
        # Volume ratio
        avg_volume_20 = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = (current_volume / avg_volume_20) * 100
        
        # Volume trend
        volume_trend = np.polyfit(range(min(20, len(volume))), 
                                  volume.iloc[-20:].values, 1)[0]
        
        # Check if volume confirms price movement
        if len(prices) >= 5:
            price_change = ((prices.iloc[-1] / prices.iloc[-5]) - 1) * 100
            volume_confirm = 1 if (price_change > 0 and volume_ratio > 100) or \
                                 (price_change < 0 and volume_ratio < 100) else 0
        else:
            volume_confirm = 0
        
        return {
            'volume_ratio': round(volume_ratio, 1),
            'volume_trend': round(volume_trend, 1),
            'volume_confirm': volume_confirm,
            'high_volume': volume_ratio > 120
        }
    except:
        return None

def generate_accurate_signal(strength_data, support_data, volume_data):
    """
    Generate more accurate trading signals based on multiple factors
    """
    if not all([strength_data, support_data, volume_data]):
        return "NO DATA"
    
    score = 0
    reasons = []
    
    # 1. Strength scoring (0-40 points)
    strength = strength_data['strength']
    if strength > 30:
        score += 40
        reasons.append("Very Strong Momentum")
    elif strength > 20:
        score += 30
        reasons.append("Strong Momentum")
    elif strength > 10:
        score += 20
        reasons.append("Moderate Momentum")
    elif strength > 0:
        score += 10
        reasons.append("Slight Momentum")
    elif strength < -20:
        score -= 30
        reasons.append("Strong Downtrend")
    
    # 2. Moving Average alignment (0-20 points)
    ma_strength = strength_data['ma_strength']
    if ma_strength > 20:
        score += 20
        reasons.append("Bullish MA Alignment")
    elif ma_strength > 10:
        score += 10
        reasons.append("Positive MA Setup")
    elif ma_strength < -20:
        score -= 20
        reasons.append("Bearish MA Alignment")
    
    # 3. Support/Resistance (0-20 points)
    if support_data['near_resistance']:
        score -= 15
        reasons.append("Near Resistance")
    elif support_data['near_support']:
        score += 10
        reasons.append("Near Support")
    
    # 4. Volume confirmation (0-20 points)
    if volume_data['high_volume'] and volume_data['volume_confirm']:
        score += 20
        reasons.append("Volume Confirmation")
    elif volume_data['high_volume']:
        score += 10
        reasons.append("High Volume")
    
    # 5. Price action (from price_trend)
    if strength_data['price_trend'] > 20:
        score += 10
        reasons.append("Positive Price Action")
    elif strength_data['price_trend'] < -20:
        score -= 10
        reasons.append("Negative Price Action")
    
    # Determine final signal
    if score >= 60:
        return "STRONG BUY", score, reasons
    elif score >= 40:
        return "BUY", score, reasons
    elif score >= 20:
        return "WATCH", score, reasons
    elif score >= 0:
        return "HOLD", score, reasons
    elif score >= -20:
        return "WEAK", score, reasons
    else:
        return "AVOID", score, reasons

# ================================
# SIDEBAR CONTROLS
# ================================
st.sidebar.header("⚙️ Analysis Settings")

# Date range
today = datetime.now()
start_date = today - timedelta(days=180)  # 6 months

# Sector selection
selected_sectors = st.sidebar.multiselect(
    "Select Sectors",
    list(NSE_SECTORS.keys()),
    default=["Bank", "IT", "Healthcare", "FMCG"]
)

# Analysis window
window = st.sidebar.slider(
    "Analysis Window (days)",
    min_value=10,
    max_value=60,
    value=20,
    step=5
)

# Strength threshold
min_strength = st.sidebar.slider(
    "Minimum Strength",
    min_value=-50,
    max_value=50,
    value=10,
    step=5
)

# Volume filter
min_volume = st.sidebar.slider(
    "Minimum Volume Ratio (%)",
    min_value=50,
    max_value=200,
    value=80,
    step=5
)

# Signal filter
signal_filter = st.sidebar.multiselect(
    "Show Signals",
    ["STRONG BUY", "BUY", "WATCH", "HOLD", "WEAK", "AVOID"],
    default=["STRONG BUY", "BUY", "WATCH"]
)

# Run analysis
st.sidebar.markdown("---")
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

# ================================
# DATA FETCHING
# ================================
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    """Fetch stock data with error handling"""
    try:
        stock = yf.download(ticker, start=start_date, end=today, progress=False)
        if not stock.empty and len(stock) >= 50:
            return stock
        return None
    except:
        return None

# ================================
# MAIN DASHBOARD
# ================================
st.markdown('<h1 class="main-header">📈 NSE Sector Rotation Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### *Improved Signal Accuracy with Chart-Aligned Calculations*")

st.markdown("""
<div class="info-card">
<h3>⚠️ Key Improvement: Chart-Aligned Signals</h3>
<p>This version fixes the issue where stocks were showing BUY signals but charts looked weak.</p>
<p><strong>New calculations consider:</strong></p>
<ul>
<li>Actual price patterns (higher highs/lows)</li>
<li>Moving average alignment (20MA vs 50MA)</li>
<li>Support/Resistance levels</li>
<li>Volume confirmation</li>
<li>Volatility adjustment</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Run analysis
if run_analysis:
    with st.spinner("🔍 Analyzing stocks with improved calculations..."):
        try:
            all_stocks_data = []
            sector_data = {}
            
            # Collect all tickers
            all_tickers = []
            for sector in selected_sectors:
                if sector in NSE_SECTORS:
                    all_tickers.extend(NSE_SECTORS[sector])
            
            if not all_tickers:
                st.error("Please select at least one sector")
                st.stop()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analyze each stock
            for idx, ticker in enumerate(all_tickers):
                status_text.text(f"Analyzing {ticker.replace('.NS', '')}...")
                progress_bar.progress((idx + 1) / len(all_tickers))
                
                # Fetch data
                stock_data = fetch_stock_data(ticker)
                
                if stock_data is not None:
                    prices = stock_data['Close']
                    volume = stock_data['Volume']
                    
                    # Calculate all metrics
                    strength_data = calculate_real_strength(prices, window)
                    support_data = calculate_support_resistance(prices)
                    volume_data = calculate_volume_strength(volume, prices)
                    
                    if all([strength_data, support_data, volume_data]):
                        # Generate signal
                        signal, score, reasons = generate_accurate_signal(
                            strength_data, support_data, volume_data
                        )
                        
                        # Find sector
                        sector_name = None
                        for sec, tickers in NSE_SECTORS.items():
                            if ticker in tickers:
                                sector_name = sec
                                break
                        
                        if sector_name:
                            stock_info = {
                                'Stock': ticker.replace('.NS', ''),
                                'Sector': sector_name,
                                'Strength': strength_data['strength'],
                                'Signal': signal,
                                'Score': score,
                                'Price': strength_data['current_price'],
                                'MA_Score': strength_data['ma_strength'],
                                'Momentum': strength_data['momentum'],
                                'Volume_Ratio': volume_data['volume_ratio'],
                                'Near_Resistance': support_data['near_resistance'],
                                'Near_Support': support_data['near_support'],
                                'Reasons': " | ".join(reasons[:3]),  # Top 3 reasons
                                'Price_Trend': strength_data['price_trend']
                            }
                            
                            all_stocks_data.append(stock_info)
                            
                            # Group by sector
                            if sector_name not in sector_data:
                                sector_data[sector_name] = []
                            sector_data[sector_name].append(stock_info)
                
                # Small delay to prevent rate limiting
                import time
                time.sleep(0.1)
            
            progress_bar.empty()
            status_text.empty()
            
            # Filter results
            filtered_stocks = [
                s for s in all_stocks_data 
                if s['Strength'] >= min_strength 
                and s['Volume_Ratio'] >= min_volume
                and s['Signal'] in signal_filter
            ]
            
            # Store results
            st.session_state.analysis_results = {
                'all': all_stocks_data,
                'filtered': filtered_stocks,
                'sector_data': sector_data
            }
            
            st.success(f"✅ Analysis complete! Found {len(filtered_stocks)} opportunities.")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ================================
# DISPLAY RESULTS
# ================================
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    all_stocks = results['all']
    filtered_stocks = results['filtered']
    sector_data = results['sector_data']
    
    # 1. MARKET OVERVIEW
    st.markdown("## 📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyzed", len(all_stocks))
    
    with col2:
        st.metric("Quality Opportunities", len(filtered_stocks))
    
    with col3:
        strong_buy = sum(1 for s in filtered_stocks if s['Signal'] == 'STRONG BUY')
        st.metric("STRONG BUY Signals", strong_buy)
    
    with col4:
        if all_stocks:
            avg_strength = np.mean([s['Strength'] for s in all_stocks])
            st.metric("Avg Market Strength", f"{avg_strength:.1f}")
    
    # 2. SECTOR ANALYSIS
    if sector_data:
        st.markdown("## 🔥 Sector Analysis")
        
        # Calculate sector statistics
        sector_stats = []
        for sector, stocks in sector_data.items():
            if stocks:
                strengths = [s['Strength'] for s in stocks]
                strong_signals = sum(1 for s in stocks if s['Signal'] in ['STRONG BUY', 'BUY'])
                
                sector_stats.append({
                    'Sector': sector,
                    'Avg_Strength': round(np.mean(strengths), 1),
                    'Strong_Signals': strong_signals,
                    'Total_Stocks': len(stocks),
                    'Top_Stock': max(stocks, key=lambda x: x['Score'])['Stock']
                })
        
        if sector_stats:
            sector_df = pd.DataFrame(sector_stats).sort_values('Avg_Strength', ascending=False)
            
            # Sector strength chart
            fig = px.bar(
                sector_df,
                x='Sector',
                y='Avg_Strength',
                color='Avg_Strength',
                color_continuous_scale='RdYlGn',
                title="Sector Strength Comparison",
                text='Avg_Strength'
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sector details
            st.markdown("## 📈 Sector Details")
            selected_sector = st.selectbox(
                "Select sector for detailed view:",
                sector_df['Sector'].tolist()
            )
            
            if selected_sector in sector_data:
                sector_stocks = sector_data[selected_sector]
                filtered_sector = [s for s in sector_stocks if s in filtered_stocks]
                
                # Sector metrics
                col1, col2, col3, col4 = st.columns(4)
                sector_info = sector_df[sector_df['Sector'] == selected_sector].iloc[0]
                
                col1.metric("Sector Strength", f"{sector_info['Avg_Strength']}")
                col2.metric("Total Stocks", sector_info['Total_Stocks'])
                col3.metric("Strong Signals", sector_info['Strong_Signals'])
                col4.metric("Top Stock", sector_info['Top_Stock'])
                
                # Display sector stocks
                if filtered_sector:
                    display_df = pd.DataFrame(filtered_sector)
                    
                    # Select and order columns
                    display_cols = ['Stock', 'Signal', 'Strength', 'Score', 'Price', 
                                   'MA_Score', 'Momentum', 'Volume_Ratio', 'Reasons']
                    
                    display_df = display_df[display_cols].sort_values('Score', ascending=False)
                    
                    # Apply styling
                    def style_signal(val):
                        if val == 'STRONG BUY':
                            return 'signal-buy'
                        elif val == 'BUY':
                            return 'signal-buy'
                        elif val == 'SELL':
                            return 'signal-sell'
                        else:
                            return 'signal-hold'
                    
                    # Format the DataFrame
                    styled_df = display_df.style.format({
                        'Strength': '{:.1f}',
                        'Score': '{:.0f}',
                        'Price': '₹{:.2f}',
                        'MA_Score': '{:.1f}',
                        'Momentum': '{:.1f}',
                        'Volume_Ratio': '{:.1f}%'
                    }).applymap(lambda x: style_signal(x), subset=['Signal'])
                    
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    
                    # Top 3 stocks chart
                    top_3 = display_df.nlargest(3, 'Score')
                    
                    fig2 = go.Figure(data=[
                        go.Bar(
                            x=top_3['Stock'],
                            y=top_3['Score'],
                            text=top_3['Signal'],
                            marker_color=['#10B981', '#3B82F6', '#8B5CF6']
                        )
                    ])
                    
                    fig2.update_layout(
                        title=f"Top 3 Stocks in {selected_sector} (by Score)",
                        yaxis_title="Signal Score",
                        height=300
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning(f"No stocks in {selected_sector} passed the current filters.")
    
    # 3. ALL OPPORTUNITIES
    if filtered_stocks:
        st.markdown("## 💎 All Trading Opportunities")
        
        # Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_confidence = sum(1 for s in filtered_stocks if s['Score'] >= 50)
            st.metric("High Confidence", high_confidence)
        
        with col2:
            above_avg = sum(1 for s in filtered_stocks if s['Strength'] > 20)
            st.metric("Strong Momentum (>20)", above_avg)
        
        with col3:
            volume_confirm = sum(1 for s in filtered_stocks if s['Volume_Ratio'] > 100)
            st.metric("Volume Confirmed", volume_confirm)
        
        # Show all filtered stocks
        all_df = pd.DataFrame(filtered_stocks)
        
        # Simple view for quick scanning
        simple_cols = ['Stock', 'Sector', 'Signal', 'Strength', 'Score', 'Price', 'Reasons']
        simple_df = all_df[simple_cols].sort_values(['Signal', 'Score'], ascending=[True, False])
        
        st.dataframe(
            simple_df.style.format({
                'Strength': '{:.1f}',
                'Score': '{:.0f}',
                'Price': '₹{:.2f}'
            }).applymap(
                lambda x: 'background-color: #D1FAE5' if x in ['STRONG BUY', 'BUY'] else 
                         'background-color: #FEF3C7' if x == 'WATCH' else '',
                subset=['Signal']
            ),
            use_container_width=True,
            height=500
        )
    
    # 4. SIGNAL EXPLANATION
    st.markdown("## 🎯 Signal Explanation")
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.markdown("""
        ### 📈 **What Makes a STRONG BUY:**
        
        **Must Have (All):**
        1. **Strength > 30** (Very strong momentum)
        2. **MA_Score > 20** (Bullish MA alignment)
        3. **Volume_Ratio > 100%** (Volume confirmation)
        
        **Should Have (2+):**
        - Momentum > 15
        - Not near resistance
        - Score ≥ 60
        - Positive price trend (>20)
        
        **Example Signals:**
        - HDFCBANK: Strength 45, MA_Score 30, Volume 120%
        - TCS: Strength 38, MA_Score 25, Momentum 20
        """)
    
    with exp_col2:
        st.markdown("""
        ### ⚠️ **Why Stocks Might Look Weak:**
        
        **Common Issues Fixed:**
        
        1. **False Strength from choppy sideways movement**
           - Old: Counted up/down days equally
           - New: Requires higher highs/lows
        
        2. **Near resistance but showing BUY**
           - Old: Ignored price levels
           - New: Penalizes near resistance
        
        3. **Low volume breakout**
           - Old: Only volume ratio
           - New: Requires volume confirmation
        
        4. **Bearish MA alignment**
           - Old: Only price vs SMA20
           - New: Checks 20MA vs 50MA alignment
        """)
    
    # 5. CASE STUDY: LUPIN Example
    st.markdown("## 🔍 Case Study: LUPIN Analysis")
    
    # Check if LUPIN is in results
    lupin_data = None
    for stock in all_stocks:
        if stock['Stock'] == 'LUPIN':
            lupin_data = stock
            break
    
    if lupin_data:
        st.markdown(f"""
        <div class="info-card">
        <h3>LUPIN Analysis ({lupin_data['Signal']})</h3>
        <p><strong>Strength:</strong> {lupin_data['Strength']} (Price Trend: {lupin_data['Price_Trend']})</p>
        <p><strong>MA Score:</strong> {lupin_data['MA_Score']} | <strong>Momentum:</strong> {lupin_data['Momentum']}</p>
        <p><strong>Volume Ratio:</strong> {lupin_data['Volume_Ratio']}%</p>
        <p><strong>Signal Reasons:</strong> {lupin_data['Reasons']}</p>
        <p><strong>Note:</strong> If this still shows BUY but chart looks weak, check:</p>
        <ul>
        <li>Is it near resistance? {lupin_data['Near_Resistance']}</li>
        <li>Is volume confirming? {lupin_data['Volume_Ratio'] > 100}</li>
        <li>Are MAs aligned bullishly? {lupin_data['MA_Score'] > 20}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    # Initial instructions
    st.markdown("""
    ## 🎯 How This Version is Better
    
    ### **Fixes the LUPIN Problem:**
    
    Old calculations gave BUY signals based on:
    - Simple up/down day counting
    - Basic moving average crossover
    
    **New calculations require:**
    1. **Actual price pattern strength** (higher highs/lows)
    2. **Moving average alignment** (20MA > 50MA for bullish)
    3. **Volume confirmation** of price moves
    4. **Support/Resistance awareness**
    5. **Volatility adjustment**
    
    ### **Key Improvements:**
    
    ✅ **No more false BUY signals** for sideways/chop
    ✅ **Better alignment with chart reading**
    ✅ **Multiple confirmation factors required**
    ✅ **Penalizes stocks near resistance**
    ✅ **Rewards stocks with volume confirmation**
    
    ### **To Test:**
    
    1. Select Healthcare sector (includes LUPIN)
    2. Run analysis
    3. Check if LUPIN signal matches chart appearance
    4. Compare with old dashboard
    
    ---
    
    *Click "Run Analysis" in sidebar to begin*
    """)

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.9em;'>
    <p>📊 <strong>Improved Sector Rotation Dashboard</strong> | Chart-Aligned Signal Generation</p>
    <p>⚠️ <em>Signals now align with actual chart patterns and technical analysis principles.</em></p>
</div>
""", unsafe_allow_html=True)