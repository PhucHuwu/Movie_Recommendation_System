"""
Model evaluation page - Optimized
"""
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from config import BACKEND_URL

@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_model_metrics():
    """Fetch model metrics with caching"""
    response = requests.get(f"{BACKEND_URL}/api/admin/models/metrics", timeout=30)
    if response.status_code == 200:
        return response.json()
    return None

@st.cache_data(ttl=600)
def fetch_model_comparison():
    """Fetch model comparison with caching"""
    response = requests.get(f"{BACKEND_URL}/api/admin/models/comparison", timeout=30)
    if response.status_code == 200:
        return response.json()
    return None

def show():
    """Show model evaluation page"""
    st.title("Model Evaluation")
    st.caption("Performance metrics of recommendation models")
    
    with st.spinner("Loading metrics..."):
        metrics_data = fetch_model_metrics()
        comparison_data = fetch_model_comparison()
    
    if not metrics_data or not metrics_data.get('metrics'):
        st.warning("No metrics available. Please run evaluation.py first.")
        return
    
    metrics = metrics_data['metrics']
    
    # Metrics table
    st.subheader("Performance Metrics")
    
    df = pd.DataFrame(metrics)
    df.columns = ['Model', 'RMSE', 'MAE', 'Precision@10', 'Recall@10']
    
    # Highlight best values
    st.dataframe(
        df.style.highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
               .highlight_max(subset=['Precision@10', 'Recall@10'], color='lightgreen')
               .format({
                   'RMSE': '{:.4f}',
                   'MAE': '{:.4f}',
                   'Precision@10': '{:.4f}',
                   'Recall@10': '{:.4f}'
               }),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Charts
    if comparison_data and comparison_data.get('models'):
        st.subheader("Model Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error metrics
            fig = go.Figure(data=[
                go.Bar(name='RMSE', x=comparison_data['models'], y=comparison_data['rmse']),
                go.Bar(name='MAE', x=comparison_data['models'], y=comparison_data['mae'])
            ])
            fig.update_layout(
                title='Error Metrics (Lower is Better)',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Ranking metrics
            fig = go.Figure(data=[
                go.Bar(name='Precision@10', x=comparison_data['models'], y=comparison_data['precision']),
                go.Bar(name='Recall@10', x=comparison_data['models'], y=comparison_data['recall'])
            ])
            fig.update_layout(
                title='Ranking Metrics (Higher is Better)',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Metric explanations
    with st.expander("📖 Metric Explanations"):
        st.markdown("""
        - **RMSE**: Root Mean Square Error - measures rating prediction accuracy
        - **MAE**: Mean Absolute Error - average prediction error
        - **Precision@10**: % of recommended items that are relevant
        - **Recall@10**: % of relevant items that were recommended
        """)
