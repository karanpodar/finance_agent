from typing import Union, Dict, List, TypedDict, Literal
import pandas as pd
import datetime as dt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from langgraph.graph import StateGraph, START, END
import os
import dotenv
from langchain_openai import ChatOpenAI
import logging
import sys
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import re


# Configure logging
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
        # logging.FileHandler('technical_analysis.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()
api_key = os.getenv("OpenAI_Alfred_API_KEY")

# Enhanced State definition with more detailed tracking
class TechnicalAnalysisState(TypedDict):
    messages: List
    stock_ticker: str
    current_price: float
    technical_data: Dict
    support_resistance_data: Dict
    analysis_summary: Dict
    analysis_complete: bool
    error_message: str

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    extra_headers={
        "HTTP-Referer": "https://karan-alfredchatbot.streamlit.app/",
        "X-Title": "Alfred Chatbot",
    },
    model="deepseek/deepseek-r1-0528-qwen3-8b:free",
    temperature=0.2
)

def safe_extract_value(series_or_value, default=None):
    """Safely extract scalar value from pandas Series or numpy array"""
    try:
        if hasattr(series_or_value, 'iloc'):
            # It's a pandas Series
            return series_or_value.iloc[-1]
        elif isinstance(series_or_value, np.ndarray):
            # It's a numpy array
            return series_or_value.flatten()[-1]
        else:
            # It's already a scalar
            return series_or_value
    except (IndexError, AttributeError) as e:
        logger.warning(f"Error extracting value: {e}, returning default: {default}")
        return default

def create_langgraph_visualization():
    """Create a visually enhanced LangGraph workflow visualization"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 9))

    # Define the workflow graph
    G = nx.DiGraph()

    # Define node positions
    nodes = {
        'START': (1, 4),
        'fetch_technical_data': (2.5, 4),
        'fetch_support_resistance': (4.5, 4),
        'analysis_synthesis': (6.5, 4),
        'report_generation': (8.5, 4),
        'visualization': (10.5, 4),
        'error_handler': (6, 2.5),
        'END': (12, 4)
    }

    # Add nodes and edges
    G.add_nodes_from(nodes)
    edges = [
        ('START', 'fetch_technical_data'),
        ('fetch_technical_data', 'fetch_support_resistance'),
        ('fetch_support_resistance', 'analysis_synthesis'),
        ('analysis_synthesis', 'report_generation'),
        ('report_generation', 'visualization'),
        ('visualization', 'END'),
        # Error paths
        ('fetch_technical_data', 'error_handler'),
        ('fetch_support_resistance', 'error_handler'),
        ('analysis_synthesis', 'error_handler'),
        ('report_generation', 'error_handler'),
        ('visualization', 'error_handler'),
        ('error_handler', 'END')
    ]
    G.add_edges_from(edges)

    # Get position dictionary
    pos = nodes

    # Node colors by type
    node_colors = {
        'START': '#4CAF50',
        'END': '#4CAF50',
        'fetch_technical_data': '#03A9F4',
        'fetch_support_resistance': '#03A9F4',
        'analysis_synthesis': '#FF9800',
        'report_generation': '#9C27B0',
        'visualization': '#7C4DFF',
        'error_handler': '#F44336'
    }

    # Dynamic label mapping
    node_labels = {
        'START': 'START',
        'END': 'END',
        'fetch_technical_data': 'Fetch\nTechnical Data',
        'fetch_support_resistance': 'Support/\nResistance',
        'analysis_synthesis': 'Analysis\nSynthesis',
        'report_generation': 'Generate\nReport',
        'visualization': 'Visualize\nData',
        'error_handler': 'Error\nHandler'
    }

    # Draw fancy boxes with dynamic widths
    for node, (x, y) in pos.items():
        label = node_labels[node]
        width = 1.4 if node not in ['START', 'END'] else 1
        height = 0.6 if node not in ['START', 'END'] else 0.4
        offset_x = width / 2
        offset_y = height / 2
        bbox = FancyBboxPatch(
            (x - offset_x, y - offset_y), width, height,
            boxstyle="round,pad=0.02",
            facecolor=node_colors[node],
            edgecolor='white',
            linewidth=2,
            mutation_aspect=1.5
        )
        ax.add_patch(bbox)
        ax.text(x, y, label, ha='center', va='center', fontsize=9.5, 
                color='white', fontweight='bold')

    # Draw main and error arrows
    for src, dst in edges:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        dx, dy = x2 - x1, y2 - y1

        if dst == 'error_handler':
            # Curved error edge to error handler
            ax.annotate('', xy=(x2, y2 + 0.1), xytext=(x1, y1 - 0.1),
                        arrowprops=dict(
                            arrowstyle='->', color='#FF5722', lw=2,
                            connectionstyle="arc3,rad=0.25"
                        ))
        elif src == 'error_handler' and dst == 'END':
            # Straight arrow from error handler to END
            ax.annotate('', xy=(x2 - 0.4, y2), xytext=(x1 + 0.2, y1),
                        arrowprops=dict(arrowstyle='->', color='#FF5722', lw=2))
        else:
            ax.annotate('', xy=(x2 - 0.4, y2), xytext=(x1 + 0.4, y1),
                        arrowprops=dict(arrowstyle='->', color="#C1C2C2", lw=2.2))

    # Title and axis setup
    ax.set_title('LangGraph Technical Analysis Workflow', fontsize=16, color='white', pad=20)
    ax.set_xlim(0, 13)
    ax.set_ylim(1.5, 5)
    ax.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#03A9F4', label='Data Processing'),
        mpatches.Patch(color='#FF9800', label='Analysis'),
        mpatches.Patch(color='#9C27B0', label='Report Generation'),
        mpatches.Patch(color='#7C4DFF', label='Visualization'),
        mpatches.Patch(color='#F44336', label='Error Handling'),
        mpatches.Patch(color='#4CAF50', label='Start/End'),
        mpatches.FancyArrow(0, 0, 1, 0, color='#C1C2C2', lw=2.2, label='Main Flow'),
        mpatches.FancyArrow(0, 0, 1, 0, color='#FF5722', lw=2, label='Error Flow')
    ]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 1), fontsize=9)

    # Tight layout & save
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return encoded

def create_technical_charts(df, technical_data, support_resistance_data):
    """Create comprehensive technical analysis charts with error handling"""
    
    try:
        # Validate inputs
        if df is None or df.empty:
            logger.warning("Empty or None dataframe provided to create_technical_charts")
            return "<div>No price data available for charting</div>"
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return f"<div>Error: Missing required columns: {', '.join(missing_columns)}</div>"
        
        # Create subplots with proper height ratios
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Moving Averages', 'Volume', 'RSI', 'MACD'),
            row_heights=[0.5, 0.2, 0.15, 0.15]  # Fixed: use row_heights instead of row_width
        )
        
        # Price and Moving Averages
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # Moving Averages with error handling
        try:
            # Check if we have enough data for moving averages
            if len(df) >= 50:  # Need at least 50 points for SMA 50
                sma_20 = SMAIndicator(df['Close'], window=20).sma_indicator()
                sma_50 = SMAIndicator(df['Close'], window=50).sma_indicator()
                ema_12 = EMAIndicator(df['Close'], window=12).ema_indicator()
                
                fig.add_trace(
                    go.Scatter(x=df.index, y=sma_20, name='SMA 20', 
                              line=dict(color='orange', width=2)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=sma_50, name='SMA 50', 
                              line=dict(color='red', width=2)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=ema_12, name='EMA 12', 
                              line=dict(color='cyan', width=1, dash='dash')),
                    row=1, col=1
                )
            elif len(df) >= 20:  # At least show SMA 20 and EMA 12
                sma_20 = SMAIndicator(df['Close'], window=20).sma_indicator()
                ema_12 = EMAIndicator(df['Close'], window=12).ema_indicator()
                
                fig.add_trace(
                    go.Scatter(x=df.index, y=sma_20, name='SMA 20', 
                              line=dict(color='orange', width=2)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=ema_12, name='EMA 12', 
                              line=dict(color='cyan', width=1, dash='dash')),
                    row=1, col=1
                )
            else:
                logger.warning("Insufficient data for moving averages (need at least 20 data points)")
                
        except Exception as e:
            logger.error(f"Error calculating moving averages: {str(e)}")
        
        # Support and Resistance levels with error handling
        try:
            if support_resistance_data and isinstance(support_resistance_data, dict):
                resistance_levels = support_resistance_data.get('resistance_levels', [])
                support_levels = support_resistance_data.get('support_levels', [])
                
                # Ensure levels are numeric and valid
                for level in resistance_levels:
                    if isinstance(level, (int, float)) and not pd.isna(level):
                        fig.add_hline(
                            y=level, 
                            line_dash="dash", 
                            line_color="red", 
                            annotation_text=f"R: ${level:.2f}",
                            row=1, col=1
                        )
                
                for level in support_levels:
                    if isinstance(level, (int, float)) and not pd.isna(level):
                        fig.add_hline(
                            y=level, 
                            line_dash="dash", 
                            line_color="green", 
                            annotation_text=f"S: ${level:.2f}",
                            row=1, col=1
                        )
        except Exception as e:
            logger.error(f"Error adding support/resistance levels: {str(e)}")
        
        # Volume with error handling
        try:
            # Handle potential NaN values in Open/Close
            colors = []
            for close, open_price in zip(df['Close'], df['Open']):
                if pd.isna(close) or pd.isna(open_price):
                    colors.append('gray')
                else:
                    colors.append('green' if close >= open_price else 'red')
            
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                       marker_color=colors, opacity=0.7),
                row=2, col=1
            )
        except Exception as e:
            logger.error(f"Error creating volume chart: {str(e)}")
        
        # RSI with error handling
        try:
            if len(df) >= 14:  # Need at least 14 points for RSI
                rsi_indicator = RSIIndicator(df['Close'], window=14)
                rsi_values = rsi_indicator.rsi()
                
                fig.add_trace(
                    go.Scatter(x=df.index, y=rsi_values, name='RSI', 
                              line=dict(color='purple', width=2)),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
            else:
                logger.warning("Insufficient data for RSI calculation (need at least 14 data points)")
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
        
        # MACD with error handling
        try:
            if len(df) >= 26:  # Need at least 26 points for MACD
                macd_indicator = MACD(df['Close'])
                macd_line = macd_indicator.macd()
                macd_signal = macd_indicator.macd_signal()
                macd_histogram = macd_indicator.macd_diff()
                
                fig.add_trace(
                    go.Scatter(x=df.index, y=macd_line, name='MACD', 
                              line=dict(color='blue', width=2)),
                    row=4, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=macd_signal, name='Signal', 
                              line=dict(color='red', width=2)),
                    row=4, col=1
                )
                fig.add_trace(
                    go.Bar(x=df.index, y=macd_histogram, name='Histogram', 
                           marker_color='gray', opacity=0.6),
                    row=4, col=1
                )
            else:
                logger.warning("Insufficient data for MACD calculation (need at least 26 data points)")
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
        
        # Update layout
        ticker = technical_data.get("ticker", "Stock") if technical_data else "Stock"
        fig.update_layout(
            title=f'Technical Analysis - {ticker}',
            template='plotly_dark',
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            autosize=True
        )
        
        # Update x-axes to remove range slider from all subplots
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        logger.error(f"Critical error in create_technical_charts: {str(e)}")
        return f"<div>Error creating technical charts: {str(e)}</div>"


def generate_html_report(state: TechnicalAnalysisState, graph_image: str, chart_html: str):
    """Generate comprehensive HTML report"""
    
    ticker = state['stock_ticker']
    current_price = state.get('current_price', 0)
    tech_data = state.get('technical_data', {})
    sr_data = state.get('support_resistance_data', {})
    analysis = state.get('analysis_summary', {})
    
    # Extract LLM response content
    llm_response = ""
    if state.get('messages'):
        last_message = state['messages'][-1]
        if hasattr(last_message, 'content'):
            llm_response = last_message.content
    
    # Parse the LLM response to extract structured information
    def parse_llm_response(response_text):
        sections = {}
        current_section = None
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('**') and line.endswith('**'):
                current_section = line.strip('*').strip()
                sections[current_section] = []
            elif current_section and line:
                sections[current_section].append(line)
        
        return sections
    
    parsed_response = parse_llm_response(llm_response)
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Technical Analysis Report - {ticker}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: #ffffff;
                line-height: 1.6;
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .header {{
                text-align: center;
                padding: 30px 0;
                border-bottom: 2px solid #ffffff20;
                margin-bottom: 30px;
            }}
            
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }}
            
            .header .subtitle {{
                font-size: 1.2em;
                opacity: 0.8;
                margin-bottom: 10px;
            }}
            
            .current-price {{
                font-size: 2em;
                font-weight: bold;
                color: #00ff88;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }}
            
            .section {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 25px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            
            .section h2 {{
                color: #00ff88;
                margin-bottom: 20px;
                font-size: 1.8em;
                border-bottom: 2px solid #00ff88;
                padding-bottom: 10px;
            }}
            
            .section h3 {{
                color: #ffd700;
                margin: 20px 0 10px 0;
                font-size: 1.3em;
            }}
            
            .workflow-diagram {{
                text-align: center;
                margin: 20px 0;
            }}
            
            .workflow-diagram img {{
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            
            .metric-card {{
                background: rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                text-align: center;
            }}
            
            .metric-value {{
                font-size: 1.5em;
                font-weight: bold;
                margin: 10px 0;
            }}
            
            .bullish {{ color: #00ff88; }}
            .bearish {{ color: #ff4444; }}
            .neutral {{ color: #ffd700; }}
            
            .recommendation-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                margin: 20px 0;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            
            .recommendation-signal {{
                font-size: 2.5em;
                font-weight: bold;
                margin: 15px 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }}
            
            .trading-levels {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            
            .level-card {{
                background: rgba(255, 255, 255, 0.05);
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #00ff88;
            }}
            
            .charts-container {{
                margin: 30px 0;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            
            .technical-indicators {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }}
            
            .indicator-card {{
                background: rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .support-resistance {{
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                gap: 20px;
            }}
            
            .sr-column {{
                flex: 1;
                min-width: 200px;
            }}
            
            .sr-level {{
                background: rgba(255, 255, 255, 0.05);
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                border-left: 3px solid #00ff88;
            }}
            
            .resistance .sr-level {{
                border-left-color: #ff4444;
            }}
            
            .footer {{
                text-align: center;
                padding: 30px 0;
                margin-top: 50px;
                border-top: 2px solid #ffffff20;
                opacity: 0.7;
            }}
            
            .timestamp {{
                background: rgba(255, 255, 255, 0.1);
                padding: 10px 20px;
                border-radius: 20px;
                display: inline-block;
                margin: 10px 0;
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    padding: 10px;
                }}
                
                .header h1 {{
                    font-size: 2em;
                }}
                
                .metrics-grid,
                .technical-indicators {{
                    grid-template-columns: 1fr;
                }}
                
                .support-resistance {{
                    flex-direction: column;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="header">
                <h1>📈 Technical Analysis Report</h1>
                <div class="subtitle">Stock: {ticker}</div>
                <div class="subtitle">Time Frame: {ticker}</div>
                <div class="current-price">${current_price:.2f}</div>
                <div class="timestamp">Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            
            <!-- LangGraph Workflow -->
            <div class="section">
                <h2>🔄 Analysis Workflow</h2>
                <p>This analysis was generated using an advanced LangGraph workflow that processes multiple data sources and applies sophisticated technical analysis algorithms.</p>
                <div class="workflow-diagram">
                    <img loading="lazy" src="data:image/png;base64,{graph_image}" alt="LangGraph Workflow" />
                </div>
            </div>
            
            <!-- Executive Summary -->
            <div class="section">
                <h2>📊 Executive Summary</h2>
                <div class="recommendation-box">
                    <div class="recommendation-signal {analysis.get('signal', 'HOLD').lower()}">{analysis.get('signal', 'HOLD')}</div>
                    <div>Confidence: {analysis.get('confidence_level', 'Medium')} ({analysis.get('confidence_score', 0.5):.2f})</div>
                    <div>Risk/Reward Ratio: {analysis.get('risk_reward_ratio', 'N/A')}</div>
                </div>
            </div>
            
            <!-- Trading Levels -->
            <div class="section">
                <h2>🎯 Trading Levels</h2>
                <div class="trading-levels">
                    <div class="level-card">
                        <h4>Entry Point</h4>
                        <div class="metric-value">${analysis.get('entry_point', 0):.2f}</div>
                    </div>
                    <div class="level-card">
                        <h4>Stop Loss</h4>
                        <div class="metric-value">${analysis.get('stop_loss', 0):.2f}</div>
                    </div>
                    <div class="level-card">
                        <h4>Take Profit</h4>
                        <div class="metric-value">${analysis.get('take_profit', 0):.2f}</div>
                    </div>
                </div>
            </div>
            
            <!-- Technical Indicators -->
            <div class="section">
                <h2>📈 Technical Indicators</h2>
                <div class="technical-indicators">
    """
    
    # Add technical indicators
    if 'technical_indicators' in tech_data:
        indicators = tech_data['technical_indicators']
        
        # RSI
        if 'RSI' in indicators:
            rsi_data = indicators['RSI']
            rsi_class = 'bearish' if rsi_data['current'] > 70 else 'bullish' if rsi_data['current'] < 30 else 'neutral'
            html_content += f"""
                    <div class="indicator-card">
                        <h3>RSI (14)</h3>
                        <div class="metric-value {rsi_class}">{rsi_data['current']}</div>
                        <div>{rsi_data['interpretation']}</div>
                    </div>
            """
        
        # MACD
        if 'MACD' in indicators:
            macd_data = indicators['MACD']
            macd_class = 'bullish' if macd_data['signal'] == 'Bullish' else 'bearish'
            html_content += f"""
                    <div class="indicator-card">
                        <h3>MACD</h3>
                        <div class="metric-value {macd_class}">{macd_data['signal']}</div>
                        <div>MACD: {macd_data['macd_line']:.4f}</div>
                        <div>Signal: {macd_data['signal_line']:.4f}</div>
                    </div>
            """
        
        # Moving Averages
        if 'Moving_Averages' in indicators:
            ma_data = indicators['Moving_Averages']
            html_content += f"""
                    <div class="indicator-card">
                        <h3>Moving Averages</h3>
                        <div>SMA 20: ${ma_data['SMA_20']:.2f}</div>
                        <div>SMA 50: ${ma_data['SMA_50']:.2f}</div>
                        <div>Price vs SMA20: {ma_data['price_vs_sma20']}</div>
                        <div>Price vs SMA50: {ma_data['price_vs_sma50']}</div>
                    </div>
            """
    
    html_content += """
                </div>
            </div>
            
            <!-- Support & Resistance -->
            <div class="section">
                <h2>📊 Support & Resistance Levels</h2>
                <div class="support-resistance">
                    <div class="sr-column support">
                        <h3>Support Levels</h3>
    """
    
    # Add support levels
    if sr_data and 'support_levels' in sr_data:
        for level in sr_data['support_levels']:
            html_content += f'<div class="sr-level">${level:.2f}</div>'
    
    html_content += """
                    </div>
                    <div class="sr-column resistance">
                        <h3>Resistance Levels</h3>
    """
    
    # Add resistance levels
    if sr_data and 'resistance_levels' in sr_data:
        for level in sr_data['resistance_levels']:
            html_content += f'<div class="sr-level">${level:.2f}</div>'
    
    html_content += f"""
                    </div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>52-Week High</h4>
                        <div class="metric-value">${sr_data.get('52_week_high', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>52-Week Low</h4>
                        <div class="metric-value">${sr_data.get('52_week_low', 0):.2f}</div>
                    </div>
                </div>
            </div>
            
            <!-- Interactive Charts -->
            <div class="section">
                <h2>📈 Interactive Charts</h2>
                <div class="charts-container">
                    {chart_html}
                </div>
            </div>
            
            <!-- Detailed Analysis -->
            <div class="section">
                <h2>🔍 Detailed Analysis</h2>
                <div style="background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 10px; white-space: pre-line; font-family: 'Courier New', monospace; line-height: 1.8;">
{llm_response}
                </div>
            </div>
            
            <!-- Analysis Breakdown -->
            <div class="section">
                <h2>📋 Analysis Breakdown</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Buy Signals</h4>
                        <div class="metric-value bullish">{analysis.get('signal_breakdown', {}).get('buy_signals', 0)}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Sell Signals</h4>
                        <div class="metric-value bearish">{analysis.get('signal_breakdown', {}).get('sell_signals', 0)}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Total Signals</h4>
                        <div class="metric-value">{analysis.get('signal_breakdown', {}).get('total_signals', 0)}</div>
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="footer">
                <p>🤖 Generated by Advanced Technical Analysis AI</p>
                <p>⚠️ This analysis is for informational purposes only and should not be considered as financial advice.</p>
                <p>📅 Report generated on {dt.datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
        </div>
        
        <script>
            // Add some interactivity
            document.addEventListener('DOMContentLoaded', function() {{
                // Smooth scrolling for internal links
                document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                    anchor.addEventListener('click', function (e) {{
                        e.preventDefault();
                        document.querySelector(this.getAttribute('href')).scrollIntoView({{
                            behavior: 'smooth'
                        }});
                    }});
                }});
                
                // Add fade-in animation for sections
                const sections = document.querySelectorAll('.section');
                const options = {{
                    threshold: 0.1,
                    rootMargin: '0px 0px -50px 0px'
                }};
                
                const observer = new IntersectionObserver(function(entries) {{
                    entries.forEach(entry => {{
                        if (entry.isIntersecting) {{
                            entry.target.style.opacity = '1';
                            entry.target.style.transform = 'translateY(0)';
                        }}
                    }});
                }}, options);
                
                sections.forEach(section => {{
                    section.style.opacity = '0';
                    section.style.transform = 'translateY(20px)';
                    section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
                    observer.observe(section);
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content

def fetch_technical_data_node(state: TechnicalAnalysisState) -> TechnicalAnalysisState:
    """Node that fetches comprehensive technical indicators and raw price data"""
    ticker = state['stock_ticker']
    logger.info(f"📊 Starting technical data fetch for {ticker}")
    
    try:
        # Download 6 months of daily data for better technical analysis
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=180)
        
        logger.info(f"Downloading data for {ticker} from {start_date.date()} to {end_date.date()}")
        
        # Download with progress disabled to reduce noise
        data = yf.download(
            ticker, 
            start=start_date, 
            end=end_date, 
            interval='1d',
            progress=False
        )
        
        if data.empty:
            logger.error(f"No data found for ticker {ticker}")
            return {
                **state,
                'error_message': f"No data found for ticker {ticker}",
                'analysis_complete': False
            }
        
        logger.info(f"Downloaded {len(data)} days of data for {ticker}")
        
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        df = data.copy()
        logger.debug(f"Data shape: {df.shape}, Columns: {list(df.columns)}")
        
        # Store raw price data for visualization
        raw_price_data = df.copy()
        # Reset index to make Date a column for easier plotting
        raw_price_data = raw_price_data.reset_index()
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return {
                **state,
                'error_message': f"Missing required columns: {missing_cols}",
                'analysis_complete': False
            }
        
        current_price = safe_extract_value(df['Close'], 0)
        logger.info(f"Current price for {ticker}: ${current_price:.2f}")
        
        # Calculate technical indicators with better error handling
        indicators = {}
        
        try:
            # RSI (14-day)
            logger.debug("Calculating RSI...")
            rsi_indicator = RSIIndicator(df['Close'], window=14)
            rsi_values = rsi_indicator.rsi()
            rsi_current = safe_extract_value(rsi_values, 50)  # Default to neutral
            
            # Add RSI values to raw data for visualization
            raw_price_data['RSI'] = rsi_values.reindex(raw_price_data.index, fill_value=np.nan)
            
            indicators["RSI"] = {
                "current": round(rsi_current, 2),
                "interpretation": "Oversold" if rsi_current < 30 else "Overbought" if rsi_current > 70 else "Neutral"
            }
            logger.debug(f"RSI: {rsi_current:.2f}")
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            indicators["RSI"] = {"current": 50, "interpretation": "Error"}
        
        try:
            # Stochastic Oscillator
            logger.debug("Calculating Stochastic...")
            stoch_indicator = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
            stoch_values = stoch_indicator.stoch()
            stoch_current = safe_extract_value(stoch_values, 50)
            
            # Add Stochastic values to raw data
            raw_price_data['Stochastic'] = stoch_values.reindex(raw_price_data.index, fill_value=np.nan)
            
            indicators["Stochastic"] = {
                "current": round(stoch_current, 2),
                "interpretation": "Oversold" if stoch_current < 20 else "Overbought" if stoch_current > 80 else "Neutral"
            }
            logger.debug(f"Stochastic: {stoch_current:.2f}")
        except Exception as e:
            logger.warning(f"Error calculating Stochastic: {e}")
            indicators["Stochastic"] = {"current": 50, "interpretation": "Error"}
        
        try:
            # MACD
            logger.debug("Calculating MACD...")
            macd_indicator = MACD(df['Close'])
            macd_line = macd_indicator.macd()
            macd_signal = macd_indicator.macd_signal()
            macd_histogram = macd_indicator.macd_diff()
            
            macd_current = safe_extract_value(macd_line, 0)
            signal_current = safe_extract_value(macd_signal, 0)
            histogram_current = safe_extract_value(macd_histogram, 0)
            
            # Add MACD values to raw data
            raw_price_data['MACD'] = macd_line.reindex(raw_price_data.index, fill_value=np.nan)
            raw_price_data['MACD_Signal'] = macd_signal.reindex(raw_price_data.index, fill_value=np.nan)
            raw_price_data['MACD_Histogram'] = macd_histogram.reindex(raw_price_data.index, fill_value=np.nan)
            
            indicators["MACD"] = {
                "macd_line": round(macd_current, 4),
                "signal_line": round(signal_current, 4),
                "histogram": round(histogram_current, 4),
                "signal": "Bullish" if macd_current > signal_current else "Bearish"
            }
            logger.debug(f"MACD: {macd_current:.4f}, Signal: {signal_current:.4f}")
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
            indicators["MACD"] = {
                "macd_line": 0, "signal_line": 0, "histogram": 0, "signal": "Error"
            }
        
        try:
            # Moving Averages
            logger.debug("Calculating Moving Averages...")
            sma_20_values = SMAIndicator(df['Close'], window=20).sma_indicator()
            sma_50_values = SMAIndicator(df['Close'], window=50).sma_indicator()
            ema_12_values = EMAIndicator(df['Close'], window=12).ema_indicator()
            ema_26_values = EMAIndicator(df['Close'], window=26).ema_indicator()
            
            sma_20 = safe_extract_value(sma_20_values, current_price)
            sma_50 = safe_extract_value(sma_50_values, current_price)
            ema_12 = safe_extract_value(ema_12_values, current_price)
            ema_26 = safe_extract_value(ema_26_values, current_price)
            
            # Add moving averages to raw data
            raw_price_data['SMA_20'] = sma_20_values.reindex(raw_price_data.index, fill_value=np.nan)
            raw_price_data['SMA_50'] = sma_50_values.reindex(raw_price_data.index, fill_value=np.nan)
            raw_price_data['EMA_12'] = ema_12_values.reindex(raw_price_data.index, fill_value=np.nan)
            raw_price_data['EMA_26'] = ema_26_values.reindex(raw_price_data.index, fill_value=np.nan)
            
            indicators["Moving_Averages"] = {
                "SMA_20": round(sma_20, 2),
                "SMA_50": round(sma_50, 2),
                "EMA_12": round(ema_12, 2),
                "EMA_26": round(ema_26, 2),
                "current_price": round(current_price, 2),
                "price_vs_sma20": "Above" if current_price > sma_20 else "Below",
                "price_vs_sma50": "Above" if current_price > sma_50 else "Below"
            }
            logger.debug(f"SMA20: {sma_20:.2f}, SMA50: {sma_50:.2f}")
        except Exception as e:
            logger.warning(f"Error calculating Moving Averages: {e}")
            indicators["Moving_Averages"] = {
                "SMA_20": current_price, "SMA_50": current_price,
                "EMA_12": current_price, "EMA_26": current_price,
                "current_price": current_price,
                "price_vs_sma20": "Error", "price_vs_sma50": "Error"
            }
        
        try:
            # Bollinger Bands
            logger.debug("Calculating Bollinger Bands...")
            bb_indicator = BollingerBands(df['Close'], window=20)
            bb_upper_values = bb_indicator.bollinger_hband()
            bb_lower_values = bb_indicator.bollinger_lband()
            bb_middle_values = bb_indicator.bollinger_mavg()
            
            bb_upper = safe_extract_value(bb_upper_values, current_price * 1.02)
            bb_lower = safe_extract_value(bb_lower_values, current_price * 0.98)
            bb_middle = safe_extract_value(bb_middle_values, current_price)
            
            # Add Bollinger Bands to raw data
            raw_price_data['BB_Upper'] = bb_upper_values.reindex(raw_price_data.index, fill_value=np.nan)
            raw_price_data['BB_Lower'] = bb_lower_values.reindex(raw_price_data.index, fill_value=np.nan)
            raw_price_data['BB_Middle'] = bb_middle_values.reindex(raw_price_data.index, fill_value=np.nan)
            
            indicators["Bollinger_Bands"] = {
                "upper_band": round(bb_upper, 2),
                "middle_band": round(bb_middle, 2),
                "lower_band": round(bb_lower, 2),
                "position": "Above Upper" if current_price > bb_upper else "Below Lower" if current_price < bb_lower else "Within Bands"
            }
            logger.debug(f"BB Upper: {bb_upper:.2f}, Lower: {bb_lower:.2f}")
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
            indicators["Bollinger_Bands"] = {
                "upper_band": current_price * 1.02, "middle_band": current_price,
                "lower_band": current_price * 0.98, "position": "Error"
            }
        
        try:
            # Volume Analysis
            logger.debug("Calculating Volume Analysis...")
            avg_volume_20 = safe_extract_value(df['Volume'].rolling(window=20).mean(), 1000000)
            current_volume = safe_extract_value(df['Volume'], 1000000)
            
            # Add volume moving average to raw data
            raw_price_data['Volume_MA_20'] = df['Volume'].rolling(window=20).mean().reindex(raw_price_data.index, fill_value=np.nan)
            
            indicators["Volume"] = {
                "current_volume": int(current_volume),
                "avg_volume_20d": int(avg_volume_20),
                "volume_ratio": round(current_volume / avg_volume_20, 2) if avg_volume_20 > 0 else 1.0
            }
            logger.debug(f"Current Volume: {current_volume:,}, Avg Volume: {avg_volume_20:,}")
        except Exception as e:
            logger.warning(f"Error calculating Volume: {e}")
            indicators["Volume"] = {"current_volume": 1000000, "avg_volume_20d": 1000000, "volume_ratio": 1.0}
        
        try:
            # Price momentum (% change over different periods)
            logger.debug("Calculating Price Momentum...")
            
            # Ensure we have enough data for momentum calculations
            if len(df) >= 30:
                price_1d = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100 if len(df) >= 2 else 0
                price_1w = ((df['Close'].iloc[-1] / df['Close'].iloc[-7]) - 1) * 100 if len(df) >= 7 else 0
                price_1m = ((df['Close'].iloc[-1] / df['Close'].iloc[-30]) - 1) * 100 if len(df) >= 30 else 0
            else:
                price_1d = price_1w = price_1m = 0
            
            # Add momentum indicators to raw data
            raw_price_data['Price_Change_1D'] = df['Close'].pct_change(1) * 100
            raw_price_data['Price_Change_1W'] = df['Close'].pct_change(7) * 100
            raw_price_data['Price_Change_1M'] = df['Close'].pct_change(30) * 100
            
            indicators["Price_Momentum"] = {
                "1_day_change": round(price_1d, 2),
                "1_week_change": round(price_1w, 2),
                "1_month_change": round(price_1m, 2)
            }
            logger.debug(f"1D: {price_1d:.2f}%, 1W: {price_1w:.2f}%, 1M: {price_1m:.2f}%")
        except Exception as e:
            logger.warning(f"Error calculating Price Momentum: {e}")
            indicators["Price_Momentum"] = {"1_day_change": 0, "1_week_change": 0, "1_month_change": 0}
        
        # Additional technical indicators for enhanced analysis
        try:
            # Williams %R
            logger.debug("Calculating Williams %R...")
            from ta.momentum import WilliamsRIndicator
            williams_r = WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=14)
            wr_values = williams_r.williams_r()
            wr_current = safe_extract_value(wr_values, -50)
            
            raw_price_data['Williams_R'] = wr_values.reindex(raw_price_data.index, fill_value=np.nan)
            
            indicators["Williams_R"] = {
                "current": round(wr_current, 2),
                "interpretation": "Oversold" if wr_current < -80 else "Overbought" if wr_current > -20 else "Neutral"
            }
        except Exception as e:
            logger.warning(f"Error calculating Williams %R: {e}")
            indicators["Williams_R"] = {"current": -50, "interpretation": "Error"}
        
        try:
            # Average True Range (ATR) for volatility
            logger.debug("Calculating ATR...")
            from ta.volatility import AverageTrueRange
            atr_indicator = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
            atr_values = atr_indicator.average_true_range()
            atr_current = safe_extract_value(atr_values, current_price * 0.02)
            
            raw_price_data['ATR'] = atr_values.reindex(raw_price_data.index, fill_value=np.nan)
            
            indicators["ATR"] = {
                "current": round(atr_current, 2),
                "volatility_pct": round((atr_current / current_price) * 100, 2) if current_price > 0 else 0
            }
        except Exception as e:
            logger.warning(f"Error calculating ATR: {e}")
            indicators["ATR"] = {"current": 0, "volatility_pct": 0}
        
        # Clean the raw price data - remove any NaN values in key columns and sort by date
        raw_price_data = raw_price_data.sort_values('Date')
        
        # Fill NaN values with forward fill for continuity in charts
        numeric_columns = raw_price_data.select_dtypes(include=[np.number]).columns
        raw_price_data[numeric_columns] = raw_price_data[numeric_columns].fillna(method='ffill')
        
        technical_data = {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'last_updated': end_date.strftime('%Y-%m-%d'),
            'technical_indicators': indicators,
            'raw_price_data': raw_price_data,  # Added raw price data for visualization
        }
        
        logger.info(f"✅ Successfully calculated technical indicators for {ticker}")
        logger.info(f"📈 Raw price data shape for visualization: {raw_price_data.shape}")

        return {
            **state,
            'current_price': round(current_price, 2),
            'technical_data': technical_data,
            'error_message': ""
        }

    except Exception as e:
        error_msg = f"Error fetching technical data for {ticker}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            'error_message': error_msg,
            'analysis_complete': False
        }

def fetch_support_resistance_node(state: TechnicalAnalysisState) -> TechnicalAnalysisState:
    """Node that calculates support and resistance levels and adds them to raw data"""
    ticker = state['stock_ticker']
    logger.info(f"📈 Starting support/resistance calculation for {ticker}")
    
    try:
        # Download more data for better support/resistance analysis
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=365)
        
        logger.info(f"Downloading extended data for {ticker} from {start_date.date()} to {end_date.date()}")
        
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        
        if data.empty:
            logger.error(f"No extended data found for ticker {ticker}")
            return {
                **state,
                'error_message': f"No data found for ticker {ticker} in support/resistance analysis"
            }
        
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        df = data.copy()
        logger.info(f"Downloaded {len(df)} days of extended data for {ticker}")
        
        current_price = safe_extract_value(df['Close'], state.get('current_price', 100))
        
        # Find local peaks and troughs with better error handling
        try:
            # Use a smaller window if we don't have enough data
            window_size = min(20, len(df) // 10)  # Adaptive window size
            logger.debug(f"Using window size of {window_size} for peak/trough detection")
            
            highs = df['High'].rolling(window=window_size, center=True, min_periods=1).max() == df['High']
            lows = df['Low'].rolling(window=window_size, center=True, min_periods=1).min() == df['Low']
            
            # Get resistance levels (peaks)
            resistance_prices = df.loc[highs.fillna(False), 'High'].values
            resistance_levels = sorted(set(resistance_prices), reverse=True)[:5] if len(resistance_prices) > 0 else [current_price * 1.05]
            
            # Get support levels (troughs)
            support_prices = df.loc[lows.fillna(False), 'Low'].values
            support_levels = sorted(set(support_prices))[-5:] if len(support_prices) > 0 else [current_price * 0.95]
            
            logger.debug(f"Found {len(resistance_levels)} resistance levels and {len(support_levels)} support levels")
            
        except Exception as e:
            logger.warning(f"Error in peak/trough detection: {e}, using price-based levels")
            resistance_levels = [current_price * 1.05, current_price * 1.10]
            support_levels = [current_price * 0.95, current_price * 0.90]
        
        # Filter levels close to current price (within 20%)
        price_range = current_price * 0.2
        nearby_resistance = [r for r in resistance_levels if current_price <= r <= current_price + price_range]
        nearby_support = [s for s in support_levels if current_price - price_range <= s <= current_price]
        
        # Ensure we have at least some levels
        if not nearby_resistance:
            nearby_resistance = [current_price * 1.05]
        if not nearby_support:
            nearby_support = [current_price * 0.95]
        
        # Calculate additional statistics
        try:
            all_time_high = df['High'].max()
            all_time_low = df['Low'].min()
            
            # 52-week high/low (or whatever data we have)
            weeks_52_data = df.iloc[-min(252, len(df)):]  # Up to 252 trading days
            week_52_high = weeks_52_data['High'].max()
            week_52_low = weeks_52_data['Low'].min()
            
        except Exception as e:
            logger.warning(f"Error calculating high/low statistics: {e}")
            all_time_high = week_52_high = current_price * 1.2
            all_time_low = week_52_low = current_price * 0.8
        
        support_resistance_data = {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'resistance_levels': [round(r, 2) for r in nearby_resistance[:3]],
            'support_levels': [round(s, 2) for s in nearby_support[:3]],
            'all_time_high': round(all_time_high, 2),
            'all_time_low': round(all_time_low, 2),
            '52_week_high': round(week_52_high, 2),
            '52_week_low': round(week_52_low, 2)
        }
        
        # Add support/resistance levels to raw price data if it exists
        raw_price_data = state['technical_data'].get('raw_price_data', pd.DataFrame())
        # print(raw_price_data.head())
        if not raw_price_data.empty:
            # Add horizontal lines for support and resistance levels
            for i, level in enumerate(nearby_resistance[:3]):
                raw_price_data[f'Resistance_{i+1}'] = level
            
            for i, level in enumerate(nearby_support[:3]):
                raw_price_data[f'Support_{i+1}'] = level
            
            # Update the state with enhanced raw data
            state['technical_data']['raw_price_data'] = raw_price_data
        
        # print(raw_price_data.head())
        
        logger.info(f"✅ Successfully calculated support/resistance for {ticker}")
        logger.debug(f"Resistance: {support_resistance_data['resistance_levels']}")
        logger.debug(f"Support: {support_resistance_data['support_levels']}")
        
        return {
            **state,
            'support_resistance_data': support_resistance_data
        }
        
    except Exception as e:
        error_msg = f"Error calculating support/resistance for {ticker}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            'error_message': error_msg
        }

def analysis_synthesis_node(state: TechnicalAnalysisState) -> TechnicalAnalysisState:
    """Node that synthesizes all data and creates trading signals"""
    print(f"🧠 Synthesizing analysis for {state['stock_ticker']}...")
    
    try:
        # Check if we have all required data
        if not state.get('technical_data') or not state.get('support_resistance_data'):
            return {
                **state,
                'error_message': "Missing required data for analysis synthesis"
            }
        
        tech_data = state['technical_data']
        sr_data = state['support_resistance_data']
        indicators = tech_data['technical_indicators']
        
        # Synthesize signals
        signals = []
        confidence_factors = []
        
        # RSI Signal
        rsi_val = indicators['RSI']['current']
        if rsi_val < 30:
            signals.append("BUY")
            confidence_factors.append(0.8)
        elif rsi_val > 70:
            signals.append("SELL")
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
            
        # MACD Signal
        macd_signal = indicators['MACD']['signal']
        if macd_signal == "Bullish":
            signals.append("BUY")
            confidence_factors.append(0.7)
        else:
            signals.append("SELL")
            confidence_factors.append(0.7)
            
        # Moving Average Signal
        ma_data = indicators['Moving_Averages']
        if ma_data['price_vs_sma20'] == "Above" and ma_data['price_vs_sma50'] == "Above":
            signals.append("BUY")
            confidence_factors.append(0.6)
        elif ma_data['price_vs_sma20'] == "Below" and ma_data['price_vs_sma50'] == "Below":
            signals.append("SELL")
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
            
        # Determine overall signal
        buy_signals = signals.count("BUY")
        sell_signals = signals.count("SELL")
        
        if buy_signals > sell_signals:
            overall_signal = "BUY"
        elif sell_signals > buy_signals:
            overall_signal = "SELL"
        else:
            overall_signal = "HOLD"
            
        # Calculate confidence
        avg_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        confidence_level = "High" if avg_confidence > 0.7 else "Medium" if avg_confidence > 0.4 else "Low"
        
        # Calculate entry/exit points
        current_price = state['current_price']
        
        # Support/Resistance based entry/exit
        resistance_levels = sr_data.get('resistance_levels', [])
        support_levels = sr_data.get('support_levels', [])
        
        # Entry point (slightly above support for BUY, slightly below resistance for SELL)
        if overall_signal == "BUY" and support_levels:
            entry_point = max(support_levels) * 1.01  # 1% above support
            stop_loss = max(support_levels) * 0.97    # 3% below support
            take_profit = min(resistance_levels) * 0.99 if resistance_levels else current_price * 1.15
        elif overall_signal == "SELL" and resistance_levels:
            entry_point = min(resistance_levels) * 0.99  # 1% below resistance
            stop_loss = min(resistance_levels) * 1.03    # 3% above resistance
            take_profit = max(support_levels) * 1.01 if support_levels else current_price * 0.85
        else:
            entry_point = current_price
            stop_loss = current_price * 0.95 if overall_signal == "BUY" else current_price * 1.05
            take_profit = current_price * 1.10 if overall_signal == "BUY" else current_price * 0.90
            
        # Risk/Reward calculation
        if overall_signal == "BUY":
            risk = abs(entry_point - stop_loss)
            reward = abs(take_profit - entry_point)
        else:
            risk = abs(stop_loss - entry_point)
            reward = abs(entry_point - take_profit)
            
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        analysis_summary = {
            'signal': overall_signal,
            'confidence_level': confidence_level,
            'confidence_score': round(avg_confidence, 2),
            'entry_point': round(entry_point, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'signal_breakdown': {
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'total_signals': len(signals)
            }
        }
        
        return {
            **state,
            'analysis_summary': analysis_summary
        }
        
    except Exception as e:
        return {
            **state,
            'error_message': f"Error in analysis synthesis: {str(e)}"
        }

def report_generation_node(state: TechnicalAnalysisState) -> TechnicalAnalysisState:
    """Node that generates the final comprehensive report using LLM"""
    print(f"📝 Generating comprehensive report for {state['stock_ticker']}...")
    
    try:
        # Prepare data for LLM
        tech_data = state.get('technical_data', {})
        sr_data = state.get('support_resistance_data', {})
        analysis_summary = state.get('analysis_summary', {})
        
        # Create comprehensive data summary for LLM
        data_summary = f"""
STOCK: {state['stock_ticker']}
CURRENT PRICE: ${state.get('current_price', 'N/A')}
ANALYSIS DATE: {dt.datetime.now().strftime('%Y-%m-%d')}

TECHNICAL INDICATORS:
{tech_data.get('technical_indicators', {})}

SUPPORT & RESISTANCE:
{sr_data}

ANALYSIS SUMMARY:
{analysis_summary}
"""
        
        # System prompt for report generation
        system_prompt = """You are an expert technical analyst. Generate a comprehensive technical analysis report based on the provided data. 

Format your response as follows:

**TECHNICAL ANALYSIS SUMMARY**
Stock: [Ticker]
Current Price: [Price]
Analysis Date: [Date]

**TREND ANALYSIS**
- Overall Trend: [Bullish/Bearish/Sideways]
- Trend Strength: [Strong/Moderate/Weak]
- Key Observations: [Key trend insights]

**TECHNICAL INDICATORS**
- RSI Signal: [Interpretation and signal]
- MACD Signal: [Interpretation and signal]  
- Moving Averages: [Price position relative to MAs]
- Bollinger Bands: [Position and volatility assessment]
- Volume Analysis: [Volume patterns and confirmation]

**SUPPORT & RESISTANCE**
- Key Resistance Levels: [List levels]
- Key Support Levels: [List levels]
- Critical Levels to Watch: [Most important levels]

**TRADING RECOMMENDATION**
- Signal: [BUY/SELL/HOLD]
- Confidence Level: [High/Medium/Low]
- Entry Point: [Suggested entry price/level]
- Stop Loss: [Risk management level]
- Take Profit: [Target price levels]
- Risk/Reward Ratio: [Assessment]

**ADDITIONAL INSIGHTS**
[Any other relevant technical observations, patterns, or warnings]

Be specific with numbers and provide clear reasoning for all recommendations."""

        # Generate report using LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate a comprehensive technical analysis report for the following data:\n\n{data_summary}")
        ]
        
        response = llm.invoke(messages)
        
        return {
            **state,
            'messages': state['messages'] + [response],
            'analysis_complete': True
        }
        
    except Exception as e:
        return {
            **state,
            'error_message': f"Error generating report: {str(e)}",
            'analysis_complete': False
        }
    
def visualization_node(state) -> dict:
    """Node that creates all visualizations and generates HTML report with comprehensive error logging"""
    
    ticker = state.get('stock_ticker', 'Unknown')
    logger.info(f"🎨 Creating visualizations and HTML report for {ticker}...")
    
    try:
        # Create LangGraph workflow visualization with error handling
        graph_image = None
        try:
            graph_image = create_langgraph_visualization()
            logger.info("✅ LangGraph visualization created successfully")
        except Exception as e:
            logger.error(f"Error creating LangGraph visualization: {str(e)}")
            graph_image = None
        
        # Create technical charts if we have price data
        chart_html = ""
        try:
            raw_price_data = state['technical_data'].get('raw_price_data', pd.DataFrame())
            if not raw_price_data.empty:
                logger.info(f"Creating technical charts with {len(raw_price_data)} data points")
                chart_html = create_technical_charts(
                    raw_price_data, 
                    state.get('technical_data', {}),
                    state.get('support_resistance_data', {})
                )
                logger.info("✅ Technical charts created successfully")
            else:
                logger.warning("No price data available for charting")
                chart_html = "<div>No price data available for technical analysis charts</div>"
        except Exception as e:
            logger.error(f"Error creating technical charts: {str(e)}")
            chart_html = f"<div>Error creating technical charts: {str(e)}</div>"
        
        # Generate HTML report with error handling
        html_report = ""
        try:
            html_report = generate_html_report(state, graph_image, chart_html)
            logger.info("✅ HTML report generated successfully")
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            html_report = f"<html><body><h1>Error generating report: {str(e)}</h1></body></html>"
        
        # Save HTML report with error handling
        filename = None
        try:
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"reports/technical_analysis_{ticker}_{timestamp}.html"
            
            # Ensure reports directory exists
            import os
            os.makedirs("reports", exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            logger.info(f"✅ HTML report saved as: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving HTML report: {str(e)}")
            filename = f"error_saving_report_{ticker}_{timestamp}.html"
        
        # Prepare visualization data
        visualization_data = {
            'graph_image': graph_image,
            'chart_html': chart_html,
            'html_report_filename': filename
        }
        
        logger.info(f"✅ Visualization node completed successfully for {ticker}")
        
        return {
            **state,
            'visualization_data': visualization_data
        }
        
    except Exception as e:
        error_msg = f"Critical error in visualization_node for {ticker}: {str(e)}"
        logger.error(error_msg)
        
        return {
            **state,
            'error_message': error_msg,
            'visualization_data': {
                'graph_image': None,
                'chart_html': f"<div>Error: {str(e)}</div>",
                'html_report_filename': None
            }
        }


def error_handler_node(state: TechnicalAnalysisState) -> TechnicalAnalysisState:
    """Node that handles errors and provides fallback responses"""
    print(f"❌ Handling error for {state['stock_ticker']}...")
    
    error_message = state.get('error_message', 'Unknown error occurred')
    
    fallback_response = AIMessage(content=f"""
**ERROR IN TECHNICAL ANALYSIS**

I encountered an issue while analyzing {state['stock_ticker']}:

{error_message}

**Possible Solutions:**
1. Verify the stock ticker symbol is correct
2. Check if the stock is actively traded
3. Ensure market data is available for this symbol
4. Try again in a few moments

Please double-check the ticker symbol and try again.
""")
    
    return {
        **state,
        'messages': state['messages'] + [fallback_response],
        'analysis_complete': True
    }

# Routing functions
def should_fetch_support_resistance(state: TechnicalAnalysisState) -> Literal["fetch_support_resistance", "error_handler"]:
    """Route to support/resistance fetching or error handling"""
    if state.get('error_message'):
        return "error_handler"
    return "fetch_support_resistance"

def should_synthesize_analysis(state: TechnicalAnalysisState) -> Literal["analysis_synthesis", "error_handler"]:
    """Route to analysis synthesis or error handling"""
    if state.get('error_message'):
        return "error_handler"
    return "analysis_synthesis"

def should_generate_report(state: TechnicalAnalysisState) -> Literal["report_generation", "error_handler"]:
    """Route to report generation or error handling"""
    if state.get('error_message'):
        return "error_handler"
    return "report_generation"

def should_generate_visualization(state: TechnicalAnalysisState) -> Literal["visualization", "error_handler"]:
    """Route to visualization or error handling"""
    if state.get('error_message'):
        return "error_handler"
    return "visualization"

def should_end(state: TechnicalAnalysisState) -> Literal[END]:
    """Always route to end after report generation or error handling"""
    return END

# Build the graph
graph_builder = StateGraph(TechnicalAnalysisState)

# Add nodes
graph_builder.add_node("fetch_technical_data", fetch_technical_data_node)
graph_builder.add_node("fetch_support_resistance", fetch_support_resistance_node)
graph_builder.add_node("analysis_synthesis", analysis_synthesis_node)
graph_builder.add_node("report_generation", report_generation_node)
graph_builder.add_node("visualization", visualization_node)
graph_builder.add_node("error_handler", error_handler_node)

# Add edges
graph_builder.add_edge(START, "fetch_technical_data")
graph_builder.add_conditional_edges(
    "fetch_technical_data",
    should_fetch_support_resistance,
    {"fetch_support_resistance": "fetch_support_resistance", "error_handler": "error_handler"}
)
graph_builder.add_conditional_edges(
    "fetch_support_resistance",
    should_synthesize_analysis,
    {"analysis_synthesis": "analysis_synthesis", "error_handler": "error_handler"}
)
graph_builder.add_conditional_edges(
    "analysis_synthesis",
    should_generate_report,
    {"report_generation": "report_generation", "error_handler": "error_handler"}
)
graph_builder.add_conditional_edges(
    "report_generation",
    should_generate_visualization,
    {"visualization": "visualization", "error_handler": "error_handler"}
)
graph_builder.add_conditional_edges(
    "visualization",
    should_end,
    {END: END}
)
graph_builder.add_conditional_edges(
    "error_handler",
    should_end,
    {END: END}
)

# Compile the graph
technical_analysis_graph = graph_builder.compile()

# Enhanced usage function
def analyze_stock(ticker: str, user_question: str = "Provide a comprehensive technical analysis"):
    """
    Analyze a stock using node-based technical analysis workflow
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        user_question: Specific question about the stock
    """
    
    initial_state = {
        'messages': [HumanMessage(content=user_question)],
        'stock_ticker': ticker.upper(),
        'current_price': 0.0,
        'technical_data': {},
        'support_resistance_data': {},
        'analysis_summary': {},
        'analysis_complete': False,
        'error_message': ""
    }
    
    print(f"🔍 Starting comprehensive technical analysis for {ticker.upper()}...")
    print("=" * 60)
    
    try:
        # Run the graph
        final_state = technical_analysis_graph.invoke(initial_state)
        
        # Print the final analysis
        if final_state['messages']:
            last_message = final_state['messages'][-1]
            if hasattr(last_message, 'content') and last_message.content:
                print(last_message.content)
                print("\n" + "=" * 60)
                
        # Return the final state for further processing if needed
        return final_state
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None

if __name__ == "__main__":

    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
    query = input("Enter your analysis question (or press Enter for default): ").strip()
    if not query:
        query = "Provide a comprehensive technical analysis"
    if not ticker:
        ticker = "AAPL"
    analysis = analyze_stock(ticker, query)
    
    # Example: Analyze Apple stock
    # apple_analysis = analyze_stock("MSFT", "What are the key support and resistance levels for AAPL? Is it a good time to enter a position?")