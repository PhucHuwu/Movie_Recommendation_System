"""
Model evaluation page
"""
import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from config import BACKEND_URL

def show():
    """Show model evaluation and comparison"""
    st.title("🤖 Model Evaluation")
    
    try:
        # Get model metrics
        metrics_response = requests.get(f"{BACKEND_URL}/api/admin/models/metrics")
        
        if metrics_response.status_code == 200:
            metrics_data = metrics_response.json()
            
            if 'message' in metrics_data:
                st.warning(metrics_data['message'])
                st.info("Run `python scripts/evaluation.py` to generate evaluation metrics.")
                return
            
            metrics = metrics_data['metrics']
            
            # Display metrics table
            st.markdown("### 📊 Model Performance Metrics")
            
            df = pd.DataFrame(metrics)
            
            # Rename columns for display
            display_df = df.copy()
            display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
            
            # Format numeric columns
            for col in display_df.columns:
                if display_df[col].dtype in ['float64', 'float32']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Model comparison charts
            st.markdown("### 📈 Model Comparison")
            
            comparison_response = requests.get(f"{BACKEND_URL}/api/admin/models/comparison")
            
            if comparison_response.status_code == 200:
                comparison_data = comparison_response.json()
                
                if 'message' not in comparison_data:
                    models = comparison_data['models']
                    
                    # Create tabs for different metrics
                    tab1, tab2, tab3 = st.tabs(["Error Metrics", "Ranking Metrics", "All Metrics"])
                    
                    with tab1:
                        # RMSE and MAE
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name='RMSE',
                            x=models,
                            y=comparison_data['rmse']
                        ))
                        fig.add_trace(go.Bar(
                            name='MAE',
                            x=models,
                            y=comparison_data['mae']
                        ))
                        fig.update_layout(
                            title='Error Metrics Comparison (Lower is Better)',
                            xaxis_title='Model',
                            yaxis_title='Error',
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        # Precision and Recall
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name='Precision@10',
                            x=models,
                            y=comparison_data['precision']
                        ))
                        fig.add_trace(go.Bar(
                            name='Recall@10',
                            x=models,
                            y=comparison_data['recall']
                        ))
                        fig.update_layout(
                            title='Ranking Metrics Comparison (Higher is Better)',
                            xaxis_title='Model',
                            yaxis_title='Score',
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        # Radar chart - all metrics
                        # Normalize metrics for visualization
                        # For error metrics, use 1 - normalized value (so higher is better)
                        import numpy as np
                        
                        max_rmse = max(comparison_data['rmse'])
                        max_mae = max(comparison_data['mae'])
                        
                        fig = go.Figure()
                        
                        for i, model in enumerate(models):
                            fig.add_trace(go.Scatterpolar(
                                r=[
                                    1 - (comparison_data['rmse'][i] / max_rmse),  # Normalized RMSE (inverted)
                                    1 - (comparison_data['mae'][i] / max_mae),  # Normalized MAE (inverted)
                                    comparison_data['precision'][i],
                                    comparison_data['recall'][i]
                                ],
                                theta=['RMSE', 'MAE', 'Precision@10', 'Recall@10'],
                                fill='toself',
                                name=model
                            ))
                        
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            title='Overall Model Performance (Higher is Better)',
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Metric explanations
            with st.expander("📚 Metric Explanations"):
                st.markdown("""
                **RMSE (Root Mean Squared Error):** Measures prediction accuracy. Lower is better.
                
                **MAE (Mean Absolute Error):** Average prediction error. Lower is better.
                
                **Precision@10:** Of the top 10 recommendations, how many are relevant? Higher is better.
                
                **Recall@10:** Of all relevant items, how many are in the top 10 recommendations? Higher is better.
                
                **Hybrid Model:** Combines all three models (User-Based, Item-Based, Neural CF) for better overall performance.
                """)
        
        else:
            st.error("Error loading model metrics")
            
    except Exception as e:
        st.error(f"Connection error: {e}")
        st.info("Make sure the backend server is running.")
