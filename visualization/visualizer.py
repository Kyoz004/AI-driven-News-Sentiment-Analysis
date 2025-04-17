import plotly.express as px
import streamlit as st

# Visualize sentiment distribution
def plot_sentiment_distribution(sentences_sentiment):
    """
    Create a pie chart showing the distribution of sentiments
    
    Args:
        sentences_sentiment (list): List of dictionaries containing sentence sentiment data
        
    Returns:
        plotly.graph_objects.Figure: A pie chart figure
    """
    sentiment_counts = {
        'POSITIVE': 0,
        'NEUTRAL': 0,
        'NEGATIVE': 0
    }
    
    for item in sentences_sentiment:
        label = item['sentiment']
        if label in sentiment_counts:
            sentiment_counts[label] += 1
        elif 'positive' in label.lower():
            sentiment_counts['POSITIVE'] += 1
        elif 'negative' in label.lower():
            sentiment_counts['NEGATIVE'] += 1
        else:
            sentiment_counts['NEUTRAL'] += 1
    
    fig = px.pie(
        names=list(sentiment_counts.keys()),
        values=list(sentiment_counts.values()),
        title="Sentiment Distribution",
        color=list(sentiment_counts.keys()),
        color_discrete_map={'POSITIVE': 'green', 'NEUTRAL': 'gray', 'NEGATIVE': 'red'}
    )
    return fig

# Visualize entity distribution
def plot_entities(entities):
    """
    Create a bar chart showing the distribution of entity types
    
    Args:
        entities (list): List of dictionaries containing entity data
        
    Returns:
        plotly.graph_objects.Figure: A bar chart figure or None if no entities
    """
    if not entities:
        return None
    
    # Group by entity type
    entity_types = {}
    for entity in entities:
        if entity['entity_type'] not in entity_types:
            entity_types[entity['entity_type']] = 0
        entity_types[entity['entity_type']] += entity['count']
    
    fig = px.bar(
        x=list(entity_types.keys()),
        y=list(entity_types.values()),
        title="Entity Types Distribution",
        labels={'x': 'Entity Type', 'y': 'Count'}
    )
    return fig

# Generate dashboard card for sentiment
def sentiment_card(sentiment):
    """
    Create a styled card displaying sentiment information
    
    Args:
        sentiment (dict): Dictionary containing sentiment label and score
    """
    score = sentiment.get('score', 0)
    label = sentiment.get('label', 'NEUTRAL')
    
    if 'positive' in label.lower():
        color = "green"
    elif 'negative' in label.lower():
        color = "red"
    else:
        color = "gray"
    
    st.markdown(f"""
    <div style="padding: 1rem; border-radius: 0.5rem; background-color: {color}25; border: 1px solid {color};">
        <h3 style="color: {color}; margin-top: 0;">{label}</h3>
        <p style="font-size: 2rem; margin: 0;">{score:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

def temporal_sentiment_analysis(sentiment_data):
    """
    Create a line chart showing sentiment changes over time
    
    Args:
        sentiment_data (list): List of dictionaries containing sentiment data with timestamps
        
    Returns:
        plotly.graph_objects.Figure: A line chart figure or None if not enough data
    """
    if not sentiment_data or len(sentiment_data) < 2:
        return None
    
    # Convert data to DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame(sentiment_data)
    
    # Ensure datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    fig = px.line(
        df, x='timestamp', y='score', color='source',
        title="Sentiment Trend Over Time",
        labels={'timestamp': 'Time', 'score': 'Sentiment Score', 'source': 'Source'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    # Add horizontal lines for sentiment boundaries
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="gray"),
        y0=0.5, y1=0.5, x0=df['timestamp'].min(), x1=df['timestamp'].max()
    )
    
    return fig

def model_metrics_card(model_metrics):
    """
    Tạo thẻ hiển thị các metrics của mô hình
    
    Args:
        model_metrics (dict): Dictionary chứa các metrics của mô hình
    """
    accuracy = model_metrics.get('accuracy', 0)
    precision = model_metrics.get('precision', 0)
    recall = model_metrics.get('recall', 0)
    f1_score = model_metrics.get('f1_score', 0)
    
    # Tính màu dựa trên giá trị F1-score
    if f1_score >= 0.8:
        color = "green"
    elif f1_score >= 0.6:
        color = "orange"
    else:
        color = "red"
    
    st.markdown(f"""
    <div style="padding: 1rem; border-radius: 0.5rem; background-color: {color}15; border: 1px solid {color};">
        <h3 style="color: {color}; margin-top: 0;">Model Metrics</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem;">
            <div>
                <p style="margin: 0; font-weight: bold;">Accuracy</p>
                <p style="font-size: 1.2rem; margin: 0;">{accuracy:.4f}</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: bold;">Precision</p>
                <p style="font-size: 1.2rem; margin: 0;">{precision:.4f}</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: bold;">Recall</p>
                <p style="font-size: 1.2rem; margin: 0;">{recall:.4f}</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: bold;">F1-Score</p>
                <p style="font-size: 1.2rem; margin: 0;">{f1_score:.4f}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def plot_model_metrics(model_evaluations_df):
    """
    Tạo biểu đồ hiển thị các metrics đánh giá mô hình
    
    Args:
        model_evaluations_df (DataFrame): DataFrame chứa dữ liệu đánh giá mô hình
        
    Returns:
        plotly.graph_objects.Figure: Biểu đồ đánh giá mô hình
    """
    if model_evaluations_df is None or model_evaluations_df.empty:
        return None
    
    # Tính giá trị trung bình cho mỗi mô hình
    metrics_df = model_evaluations_df.groupby('model_name')[
        ['accuracy', 'precision', 'recall', 'f1_score']
    ].mean().reset_index()
    
    # Tạo dữ liệu cho biểu đồ
    df_melted = metrics_df.melt(
        id_vars=['model_name'],
        value_vars=['accuracy', 'precision', 'recall', 'f1_score'],
        var_name='Metric', value_name='Value'
    )
    
    # Tạo biểu đồ cột
    fig = px.bar(
        df_melted, 
        x='model_name', 
        y='Value', 
        color='Metric',
        title='Model Performance Metrics',
        barmode='group',
        labels={'model_name': 'Model', 'Value': 'Score'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    return fig

def plot_user_ratings(model_evaluations_df):
    """
    Tạo biểu đồ hiển thị đánh giá của người dùng
    
    Args:
        model_evaluations_df (DataFrame): DataFrame chứa dữ liệu đánh giá mô hình
        
    Returns:
        plotly.graph_objects.Figure: Biểu đồ đánh giá người dùng
    """
    if model_evaluations_df is None or model_evaluations_df.empty:
        return None
    
    # Tính giá trị trung bình đánh giá cho mỗi mô hình
    ratings_df = model_evaluations_df.groupby('model_name')[
        ['user_rating']
    ].mean().reset_index()
    
    # Tạo biểu đồ cột
    fig = px.bar(
        ratings_df, 
        x='model_name', 
        y='user_rating',
        title='User Satisfaction Ratings (1-5)',
        labels={'model_name': 'Model', 'user_rating': 'Average Rating'},
        color='user_rating',
        color_continuous_scale=px.colors.sequential.Viridis,
        range_color=[1, 5]
    )
    
    return fig 