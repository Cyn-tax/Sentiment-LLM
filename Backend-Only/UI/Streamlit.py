"""
Streamlit UI for Sentiment Analysis
Provides a user-friendly interface for the OpenAI sentiment analysis model.
"""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path

# Add parent directory to path to import the model
sys.path.append(str(Path(__file__).parent.parent))
from Models.OpenAI import SentimentAnalyzer as OpenAIAnalyzer

# Import Groq analyzer - handle import error gracefully
# Note: Using importlib to handle hyphenated filename
import importlib.util
groq_module_path = Path(__file__).parent.parent / "Models" / "Groq-llama.py"
if groq_module_path.exists():
    spec = importlib.util.spec_from_file_location("Groq_llama", groq_module_path)
    groq_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(groq_module)
    GroqAnalyzer = groq_module.SentimentAnalyzer
    GROQ_AVAILABLE = True
else:
    GROQ_AVAILABLE = False
    GroqAnalyzer = None

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 1rem 0;
        letter-spacing: -0.02em;
    }
    
    /* Sentiment Cards */
    .sentiment-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .sentiment-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }
    
    .sentiment-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .sentiment-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sentiment-neutral {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Success Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Info Messages */
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 4px solid #17a2b8;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Error Messages */
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Text Area */
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File Uploader */
    .uploadedFile {
        border-radius: 12px;
        border: 2px dashed #cbd5e0;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 8px;
        padding: 1rem;
        font-weight: 600;
    }
    
    /* Remove default Streamlit styling */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Section Headers */
    h3 {
        color: #2d3748;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Number Input Styling */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Spinner Styling */
    .stSpinner > div {
        border-color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_analyzer(model_type: str = "OpenAI"):
    """Initialize and cache the sentiment analyzer."""
    try:
        if model_type == "Groq":
            analyzer = GroqAnalyzer(model="llama-3.1-8b-instant")
        else:  # OpenAI
            analyzer = OpenAIAnalyzer(model="gpt-5")
        return analyzer
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        if model_type == "Groq":
            st.info("Please ensure your .env file contains GROQ_API_KEY")
        else:
            st.info("Please ensure your .env file contains OPENAI_API_KEY")
        return None
    except Exception as e:
        st.error(f"Error initializing analyzer: {e}")
        return None


def display_confusion_matrix(confusion_matrix_data):
    """Display confusion matrix in a formatted way."""
    st.markdown("### Confusion Matrix")
    st.markdown("---")
    
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Accuracy", f"{confusion_matrix_data['accuracy']:.2f}%")
    with col2:
        st.metric("Total Correct", f"{confusion_matrix_data['total_correct']}/{confusion_matrix_data['total_processed']}")
    with col3:
        accuracy_pct = (confusion_matrix_data['total_correct'] / confusion_matrix_data['total_processed'] * 100) if confusion_matrix_data['total_processed'] > 0 else 0
        st.metric("Accuracy Rate", f"{accuracy_pct:.2f}%")
    
    # Confusion matrix table
    st.markdown("### Confusion Matrix Table")
    matrix = confusion_matrix_data['matrix']
    
    # Create DataFrame for better visualization
    df_matrix = pd.DataFrame(matrix).T
    df_matrix.index.name = "Actual \\ Predicted"
    df_matrix.columns.name = "Predicted"
    
    st.dataframe(df_matrix, use_container_width=True)
    
    # Per-class metrics
    st.markdown("### Per-Class Performance Metrics")
    metrics = confusion_matrix_data['class_metrics']
    
    for sentiment in ["Positive", "Negative", "Neutral"]:
        if sentiment in metrics:
            metric = metrics[sentiment]
            with st.expander(f"{sentiment} Class Metrics"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Precision", f"{metric['precision']:.4f}")
                with col2:
                    st.metric("Recall", f"{metric['recall']:.4f}")
                with col3:
                    st.metric("F1-Score", f"{metric['f1_score']:.4f}")
                with col4:
                    st.metric("TP", metric['true_positives'])


def display_results_table(results):
    """Display results in a formatted table."""
    if not results:
        st.warning("No results to display.")
        return
    
    df = pd.DataFrame(results)
    
    # Add color coding for sentiments
    def color_sentiment(val):
        if val == "Positive":
            return "background-color: #d4edda; color: #155724"
        elif val == "Negative":
            return "background-color: #f8d7da; color: #721c24"
        elif val == "Neutral":
            return "background-color: #fff3cd; color: #856404"
        return ""
    
    # Display with styling
    st.markdown("### Analysis Results")
    st.markdown("---")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_sentiment = st.selectbox(
            "Filter by Predicted Sentiment",
            ["All"] + list(df['predicted_sentiment'].unique()),
            key="pred_filter"
        )
    with col2:
        filter_true = st.selectbox(
            "Filter by True Sentiment",
            ["All"] + list(df['true_sentiment'].unique()),
            key="true_filter"
        )
    
    # Apply filters
    filtered_df = df.copy()
    if filter_sentiment != "All":
        filtered_df = filtered_df[filtered_df['predicted_sentiment'] == filter_sentiment]
    if filter_true != "All":
        filtered_df = filtered_df[filtered_df['true_sentiment'] == filter_true]
    
    # Display filtered results
    st.dataframe(
        filtered_df.style.applymap(color_sentiment, subset=['predicted_sentiment', 'true_sentiment']),
        use_container_width=True,
        hide_index=True
    )
    
    # Show match accuracy
    if 'true_sentiment' in filtered_df.columns and 'predicted_sentiment' in filtered_df.columns:
        matches = (filtered_df['true_sentiment'] == filtered_df['predicted_sentiment']).sum()
        total = len(filtered_df)
        match_rate = (matches / total * 100) if total > 0 else 0
        st.info(f"Match Rate: {matches}/{total} ({match_rate:.2f}%)")


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<div class="main-header">Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### Configuration")
        st.markdown("---")
        
        # Model selection
        model_options = ["OpenAI"]
        if GROQ_AVAILABLE:
            model_options.append("Groq")
        
        model_type = st.selectbox(
            "Select Model",
            model_options,
            index=0,
            help="Choose between OpenAI GPT-5 or Groq Llama-3.1-8b-instant"
        )
        
        if not GROQ_AVAILABLE and model_type == "Groq":
            st.error("Groq model not available. Please install groq package.")
            model_type = "OpenAI"
        
        if model_type == "OpenAI":
            model_name = "GPT-5"
            api_key_name = "OPENAI_API_KEY"
        else:
            model_name = "Llama-3.1-8b-instant"
            api_key_name = "GROQ_API_KEY"
        
        st.info(f"**Model:** {model_name}")
        st.markdown("---")
        st.markdown(f"""
        <div style="padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); 
                    border-radius: 12px; border-left: 4px solid #667eea;">
            <p style="margin: 0; font-size: 0.9rem; color: #4a5568;">
                This dashboard uses {model_name} for sentiment analysis. 
                Upload a CSV file or analyze individual feedback texts.
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #718096;">
                API Key: {api_key_name}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = get_analyzer(model_type)
    if analyzer is None:
        st.stop()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Single Text Analysis", "CSV Processing", "View Results"])
    
    # Tab 1: Single Text Analysis
    with tab1:
        st.markdown("### Analyze Single Feedback Text")
        st.markdown("---")
        
        feedback_text = st.text_area(
            "Enter customer feedback:",
            height=150,
            placeholder="Enter the customer feedback text here..."
        )
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)
        
        if analyze_button:
            if not feedback_text.strip():
                st.warning("Please enter some feedback text to analyze.")
            else:
                with st.spinner("Analyzing sentiment..."):
                    try:
                        sentiment = analyzer.analyze_sentiment(feedback_text)
                        
                        # Display result
                        st.success("Analysis Complete!")
                        
                        # Color-coded sentiment display with enhanced styling
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col2:
                            sentiment_class = {
                                "Positive": "sentiment-positive",
                                "Negative": "sentiment-negative",
                                "Neutral": "sentiment-neutral"
                            }.get(sentiment, "sentiment-neutral")
                            
                            st.markdown(f"""
                            <div class="sentiment-card">
                                <div class="sentiment-label">Predicted Sentiment</div>
                                <div class="sentiment-value {sentiment_class}">{sentiment}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show feedback text
                        with st.expander("View Feedback Text"):
                            st.write(feedback_text)
                    
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
    
    # Tab 2: CSV Processing
    with tab2:
        st.markdown("### Process CSV File")
        st.markdown("---")
        
        # Show results summary if available
        if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
            results = st.session_state['analysis_results']
            st.success(f"Results available: {len(results)} rows processed")
            st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with 'feedback_text' column (and optionally 'true_sentiment' column)",
            key="csv_uploader"
        )
        
        if uploaded_file is not None:
            # Read CSV
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"File loaded successfully! ({len(df)} rows)")
                
                # Display preview
                with st.expander("Preview CSV Data"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Check required columns
                if 'feedback_text' not in df.columns:
                    st.error("CSV file must contain a 'feedback_text' column")
                else:
                    # Processing options
                    col1, col2 = st.columns(2)
                    with col1:
                        start_row = st.number_input(
                            "Start Row",
                            min_value=1,
                            max_value=len(df),
                            value=1,
                            help="Row number to start processing from (1-indexed)"
                        )
                    with col2:
                        end_row = st.number_input(
                            "End Row",
                            min_value=1,
                            max_value=len(df),
                            value=len(df),
                            help="Row number to end processing at (leave as max for all rows)"
                        )
                    
                    # Save uploaded file temporarily
                    temp_path = Path("temp_uploaded_file.csv")
                    df.to_csv(temp_path, index=False)
                    
                    process_button = st.button("Process CSV", type="primary", use_container_width=True)
                    
                    if process_button:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Calculate total rows to process
                        total_rows_to_process = end_row - start_row + 1
                        
                        # Define progress callback
                        def update_progress(current, total, row_num):
                            progress = current / total if total > 0 else 0
                            progress_bar.progress(progress)
                            status_text.text(f"Processing row {row_num} of {end_row} ({current}/{total} rows completed)...")
                        
                        try:
                            status_text.text("Initializing processing...")
                            
                            # Process CSV with progress callback
                            analyzer.process_csv(
                                str(temp_path), 
                                start_row=start_row, 
                                end_row=end_row,
                                progress_callback=update_progress
                            )
                            
                            status_text.text("Calculating metrics...")
                            
                            # Save results to session state
                            st.session_state['analysis_results'] = analyzer.results.copy()
                            
                            # Calculate and save confusion matrix
                            confusion_matrix = analyzer.calculate_confusion_matrix()
                            st.session_state['confusion_matrix'] = confusion_matrix
                            
                            progress_bar.progress(1.0)
                            status_text.text("Complete!")
                            
                            # Small delay to show completion
                            import time
                            time.sleep(0.5)
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success(f"Successfully processed {len(analyzer.results)} rows!")
                            
                            # Show quick summary
                            st.markdown("---")
                            st.markdown("### Quick Summary")
                            summary_col1, summary_col2, summary_col3 = st.columns(3)
                            with summary_col1:
                                positive = sum(1 for r in analyzer.results if r.get('predicted_sentiment') == 'Positive')
                                st.metric("Positive", positive)
                            with summary_col2:
                                negative = sum(1 for r in analyzer.results if r.get('predicted_sentiment') == 'Negative')
                                st.metric("Negative", negative)
                            with summary_col3:
                                neutral = sum(1 for r in analyzer.results if r.get('predicted_sentiment') == 'Neutral')
                                st.metric("Neutral", neutral)
                            
                            if confusion_matrix:
                                st.metric("Overall Accuracy", f"{confusion_matrix['accuracy']:.2f}%")
                            
                            st.info("Processing complete! Switch to the 'View Results' tab for detailed analysis.")
                            
                        except Exception as e:
                            progress_bar.empty()
                            status_text.empty()
                            st.error(f"Error processing CSV: {e}")
                            import traceback
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())
                        finally:
                            # Clean up temp file
                            if temp_path.exists():
                                temp_path.unlink()
            
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
        
        # Option to use default Data.csv
        st.markdown("---")
        if st.button("Use Default Data.csv"):
            default_path = Path(__file__).parent.parent.parent / "Data.csv"
            if default_path.exists():
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Initializing processing...")
                    
                    # Read CSV to get total row count
                    df_default = pd.read_csv(default_path)
                    total_rows_default = len(df_default)
                    
                    # Define progress callback
                    def update_progress_default(current, total, row_num):
                        progress = current / total if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(f"Processing row {row_num} of {total_rows_default} ({current}/{total} rows completed)...")
                    
                    # Process CSV with progress callback
                    analyzer.process_csv(
                        str(default_path), 
                        start_row=1, 
                        end_row=None,
                        progress_callback=update_progress_default
                    )
                    
                    status_text.text("Calculating metrics...")
                    
                    # Save results to session state
                    st.session_state['analysis_results'] = analyzer.results.copy()
                    
                    # Calculate and save confusion matrix
                    confusion_matrix = analyzer.calculate_confusion_matrix()
                    st.session_state['confusion_matrix'] = confusion_matrix
                    
                    progress_bar.progress(1.0)
                    status_text.text("Complete!")
                    
                    # Small delay to show completion
                    import time
                    time.sleep(0.5)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("Default CSV processed successfully!")
                    
                    # Show quick summary
                    st.markdown("---")
                    st.markdown("### Quick Summary")
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    with summary_col1:
                        positive = sum(1 for r in analyzer.results if r.get('predicted_sentiment') == 'Positive')
                        st.metric("Positive", positive)
                    with summary_col2:
                        negative = sum(1 for r in analyzer.results if r.get('predicted_sentiment') == 'Negative')
                        st.metric("Negative", negative)
                    with summary_col3:
                        neutral = sum(1 for r in analyzer.results if r.get('predicted_sentiment') == 'Neutral')
                        st.metric("Neutral", neutral)
                    
                    if confusion_matrix:
                        st.metric("Overall Accuracy", f"{confusion_matrix['accuracy']:.2f}%")
                    
                    st.info("Processing complete! Switch to the 'View Results' tab for detailed analysis.")
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Error processing default CSV: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Error processing default CSV: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
            else:
                st.warning("Default Data.csv not found in project root.")
    
    # Tab 3: View Results
    with tab3:
        st.markdown("### Analysis Results & Metrics")
        st.markdown("---")
        
        if 'analysis_results' not in st.session_state or not st.session_state['analysis_results']:
            st.info("Process a CSV file first to see results here.")
        else:
            results = st.session_state['analysis_results']
            confusion_matrix = st.session_state.get('confusion_matrix')
            
            # Summary statistics
            st.markdown("### Summary Statistics")
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Processed", len(results))
            with col2:
                positive_count = sum(1 for r in results if r.get('predicted_sentiment') == 'Positive')
                st.metric("Positive", positive_count)
            with col3:
                negative_count = sum(1 for r in results if r.get('predicted_sentiment') == 'Negative')
                st.metric("Negative", negative_count)
            with col4:
                neutral_count = sum(1 for r in results if r.get('predicted_sentiment') == 'Neutral')
                st.metric("Neutral", neutral_count)
            
            # Display results table
            display_results_table(results)
            
            # Display confusion matrix if available
            if confusion_matrix:
                st.markdown("---")
                display_confusion_matrix(confusion_matrix)
            
            # Download results
            st.markdown("---")
            st.markdown("### Download Results")
            
            if st.button("Download Results as JSON"):
                output_data = {
                    "total_processed": len(results),
                    "results": results,
                    "confusion_matrix": confusion_matrix if confusion_matrix else None
                }
                
                json_str = json.dumps(output_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download JSON File",
                    data=json_str,
                    file_name="sentiment_analysis_results.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()

