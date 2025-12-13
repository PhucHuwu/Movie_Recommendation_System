"""
Model evaluation page - Full charts for model comparison
"""
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from config import BACKEND_URL

@st.cache_data(ttl=600)
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
    st.subheader("Performance Metrics Table")
    
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
    
    # Charts in tabs
    if comparison_data and comparison_data.get('models'):
        tab1, tab2, tab3 = st.tabs(["Error Metrics", "Ranking Metrics", "Radar Chart"])
        
        with tab1:
            st.subheader("Error Metrics Comparison")
            
            # Grouped bar chart for RMSE and MAE
            fig = go.Figure(data=[
                go.Bar(name='RMSE', x=comparison_data['models'], y=comparison_data['rmse'], 
                       marker_color='indianred'),
                go.Bar(name='MAE', x=comparison_data['models'], y=comparison_data['mae'],
                       marker_color='lightsalmon')
            ])
            fig.update_layout(
                title='RMSE and MAE by Model (Lower is Better)',
                barmode='group',
                xaxis_title='Model',
                yaxis_title='Error Value',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model highlight
            best_rmse_idx = np.argmin(comparison_data['rmse'])
            best_mae_idx = np.argmin(comparison_data['mae'])
            st.success(f"Best RMSE: **{comparison_data['models'][best_rmse_idx]}** ({comparison_data['rmse'][best_rmse_idx]:.4f})")
            st.success(f"Best MAE: **{comparison_data['models'][best_mae_idx]}** ({comparison_data['mae'][best_mae_idx]:.4f})")
        
        with tab2:
            st.subheader("Ranking Metrics Comparison")
            
            # Grouped bar chart for Precision and Recall
            fig = go.Figure(data=[
                go.Bar(name='Precision@10', x=comparison_data['models'], y=comparison_data['precision'],
                       marker_color='steelblue'),
                go.Bar(name='Recall@10', x=comparison_data['models'], y=comparison_data['recall'],
                       marker_color='lightblue')
            ])
            fig.update_layout(
                title='Precision@10 and Recall@10 by Model (Higher is Better)',
                barmode='group',
                xaxis_title='Model',
                yaxis_title='Score',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model highlight
            best_p_idx = np.argmax(comparison_data['precision'])
            best_r_idx = np.argmax(comparison_data['recall'])
            st.success(f"Best Precision@10: **{comparison_data['models'][best_p_idx]}** ({comparison_data['precision'][best_p_idx]:.4f})")
            st.success(f"Best Recall@10: **{comparison_data['models'][best_r_idx]}** ({comparison_data['recall'][best_r_idx]:.4f})")
        
        with tab3:
            st.subheader("Model Performance Radar Chart")
            
            # Normalize metrics for radar chart (0-1 scale)
            rmse_norm = [1 - (r / max(comparison_data['rmse'])) for r in comparison_data['rmse']]
            mae_norm = [1 - (m / max(comparison_data['mae'])) for m in comparison_data['mae']]
            p_norm = [p / max(comparison_data['precision']) if max(comparison_data['precision']) > 0 else 0 
                      for p in comparison_data['precision']]
            r_norm = [r / max(comparison_data['recall']) if max(comparison_data['recall']) > 0 else 0 
                      for r in comparison_data['recall']]
            
            categories = ['RMSE (inv)', 'MAE (inv)', 'Precision@10', 'Recall@10']
            
            fig = go.Figure()
            
            colors = ['blue', 'green', 'red', 'purple']
            for i, model in enumerate(comparison_data['models']):
                values = [rmse_norm[i], mae_norm[i], p_norm[i], r_norm[i]]
                values.append(values[0])  # Close the polygon
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=model,
                    opacity=0.6
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title='Normalized Model Performance (Higher is Better)',
                showlegend=True,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Note: RMSE and MAE are inverted so higher values indicate better performance")
    
    st.markdown("---")
    
    # Metric explanations
    with st.expander("Metric Explanations"):
        st.markdown("""
        **Error Metrics (Lower is Better):**
        - **RMSE**: Root Mean Square Error - penalizes large errors more heavily
        - **MAE**: Mean Absolute Error - average absolute difference between predicted and actual ratings
        
        **Ranking Metrics (Higher is Better):**
        - **Precision@10**: Proportion of recommended items that are relevant (user actually rated highly)
        - **Recall@10**: Proportion of relevant items that appear in recommendations
        
        **Model Types:**
        - **User-Based CF**: Recommends based on similar users' preferences
        - **Item-Based CF**: Recommends based on similar items the user liked
        - **Neural CF**: Uses neural networks to learn complex user-item interactions
        - **Hybrid**: Combines predictions from all other models
        """)
