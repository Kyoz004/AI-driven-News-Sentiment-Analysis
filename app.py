import streamlit as st
import pandas as pd
import json
import datetime
import os
import numpy as np

# Import from our modules
from database.db_manager import (
    init_db, save_to_db, get_analysis_history, 
    clear_database, backup_database, check_database_exists,
    SQLITE_PATH, get_database_info, DB_TYPE,
    get_model_evaluations, get_model_evaluation_stats,
    save_model_evaluation, get_article_by_id, get_analysis_result,
    deserialize_json
)
from utils.text_extractor import extract_text_from_url, detect_language
from analysis.nlp_analyzer import (
    load_sentiment_pipeline, 
    load_ner_pipeline, 
    analyze_sentiment_by_sentence, 
    extract_entities,
    standardize_sentiment_label,
    calculate_overall_sentiment_from_sentences
)
from visualization.visualizer import (
    plot_sentiment_distribution, 
    plot_entities, 
    sentiment_card,
    model_metrics_card,
    plot_model_metrics,
    plot_user_ratings
)
from config.model_config import (
    SENTIMENT_MODEL_OPTIONS, 
    ENTITY_MODEL_OPTIONS, 
    DEFAULT_SETTINGS
)

def safe_json_loads(json_data):
    """Safely convert JSON string to Python object, handling cases where data is already a Python object"""
    if not json_data:
        return None
    
    if isinstance(json_data, (dict, list)):
        return json_data
    
    try:
        return json.loads(json_data)
    except (TypeError, json.JSONDecodeError) as e:
        print(f"JSON conversion error: {e}")
        return json_data

def main():
    st.set_page_config(
        page_title="News Analysis Dashboard",
        page_icon="📰",
        layout="wide"
    )
    
    # Initialize database
    conn = init_db()
    
    # Sidebar navigation
    st.sidebar.title("News Analysis System")
    page = st.sidebar.radio("Navigate", ["Dashboard", "Analysis", "History", "Settings"])
    
    if page == "Dashboard":
        st.title("📊 News Analysis Dashboard")
        
        # Panel đánh giá mô hình
        st.subheader("Model Performance Overview")
        
        # Lấy thống kê đánh giá mô hình
        model_stats = get_model_evaluation_stats(conn)
        
        if model_stats is not None and not model_stats.empty:
            # Tạo tabs cho từng loại mô hình
            sentiment_tab, entity_tab = st.tabs(["Sentiment Models", "Entity Recognition Models"])
            
            with sentiment_tab:
                # Lọc dữ liệu cho mô hình sentiment
                sentiment_stats = model_stats[model_stats['model_type'] == 'sentiment']
                if not sentiment_stats.empty:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Hiển thị thẻ đánh giá cho mô hình tốt nhất
                        best_model = sentiment_stats.iloc[sentiment_stats['avg_f1_score'].argmax()]
                        model_metrics_card({
                            'accuracy': best_model['avg_accuracy'],
                            'precision': best_model['avg_precision'],
                            'recall': best_model['avg_recall'],
                            'f1_score': best_model['avg_f1_score']
                        })
                        st.markdown(f"**Best Sentiment Model:** {best_model['model_type']}")
                        st.markdown(f"**Average User Rating:** ⭐ {best_model['avg_user_rating']:.1f}/5.0")
                    
                    with col2:
                        # Lấy tất cả đánh giá mô hình sentiment
                        sentiment_evals = get_model_evaluations(conn, model_type='sentiment')
                        if sentiment_evals is not None and not sentiment_evals.empty:
                            # Hiển thị biểu đồ metrics
                            metrics_fig = plot_model_metrics(sentiment_evals)
                            if metrics_fig:
                                st.plotly_chart(metrics_fig, use_container_width=True)
                else:
                    st.info("No sentiment model evaluations available.")
            
            with entity_tab:
                # Lọc dữ liệu cho mô hình entity
                entity_stats = model_stats[model_stats['model_type'] == 'entity']
                if not entity_stats.empty:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Hiển thị thẻ đánh giá cho mô hình tốt nhất
                        best_model = entity_stats.iloc[entity_stats['avg_f1_score'].argmax()]
                        model_metrics_card({
                            'accuracy': best_model['avg_accuracy'],
                            'precision': best_model['avg_precision'],
                            'recall': best_model['avg_recall'],
                            'f1_score': best_model['avg_f1_score']
                        })
                        st.markdown(f"**Best Entity Recognition Model:** {best_model['model_type']}")
                        st.markdown(f"**Average User Rating:** ⭐ {best_model['avg_user_rating']:.1f}/5.0")
                    
                    with col2:
                        # Lấy tất cả đánh giá mô hình entity
                        entity_evals = get_model_evaluations(conn, model_type='entity')
                        if entity_evals is not None and not entity_evals.empty:
                            # Hiển thị biểu đồ metrics
                            metrics_fig = plot_model_metrics(entity_evals)
                            if metrics_fig:
                                st.plotly_chart(metrics_fig, use_container_width=True)
                else:
                    st.info("No entity recognition model evaluations available.")
        else:
            st.info("No model evaluations available yet. Rate model performance after analysis to see metrics here.")
            
            # Hiển thị ví dụ về biểu đồ mẫu nếu không có dữ liệu thực
            # Tạo dữ liệu mẫu
            sample_models = ['ViSoBERT', 'PhoBERT', 'Vietnamese BERT', 'Vietnamese ELECTRA']
            sample_data = []
            for model in sample_models:
                sample_data.append({
                    'model_name': model,
                    'model_type': 'sentiment',
                    'accuracy': np.random.uniform(0.75, 0.95),
                    'precision': np.random.uniform(0.7, 0.9),
                    'recall': np.random.uniform(0.7, 0.9),
                    'f1_score': np.random.uniform(0.75, 0.93),
                    'user_rating': np.random.randint(3, 6)
                })
            
            sample_df = pd.DataFrame(sample_data)
            
            st.markdown("### Sample Model Performance (Simulated Data)")
            st.caption("This is example data. Actual metrics will appear here after model evaluation.")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                model_metrics_card({
                    'accuracy': 0.89,
                    'precision': 0.86,
                    'recall': 0.85,
                    'f1_score': 0.87
                })
            with col2:
                metrics_fig = plot_model_metrics(sample_df)
                if metrics_fig:
                    st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Thêm phần recent analyses như đang có
        st.subheader("Recent Analyses")
        history_df = get_analysis_history(conn, limit=5)
        
        if history_df is not None and not history_df.empty:
            for _, row in history_df.iterrows():
                with st.expander(f"{row['title']} ({row['source']})"):
                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown(f"**Source:** {row['source']}")
                        st.markdown(f"**Language:** {row['language']}")
                    with cols[1]:
                        st.markdown(f"**Sentiment:** {row['sentiment_label']}")
                        st.markdown(f"**Score:** {row['sentiment_score']:.2f}")
                    with cols[2]:
                        st.markdown(f"**Analyzed on:** {row['fetch_date']}")
                        st.markdown(f"[View full analysis](#)")
        else:
            st.info("No analysis history found. Start by analyzing a URL in the Analysis tab.")
    
    elif page == "Analysis":
        st.title("🔍 News Text Analysis")
        
        # Input tab selection
        input_method = st.radio("Input Method", ["URL", "Text Input", "Bulk Analysis"], horizontal=True)
        
        if input_method == "URL":
            url = st.text_input("Enter URL", placeholder="https://example.com")
            col1, col2 = st.columns(2)
            with col1:
                selected_sentiment_model = st.selectbox(
                    "Select Sentiment Analysis Model", 
                    list(SENTIMENT_MODEL_OPTIONS.keys())
                )
            with col2:
                selected_entity_model = st.selectbox(
                    "Select Entity Recognition Model", 
                    list(ENTITY_MODEL_OPTIONS.keys())
                )
            
            if st.button("Analyze Content"):
                if not url:
                    st.error("Please provide a valid URL!")
                else:
                    with st.spinner("Fetching and analyzing content..."):
                        try:
                            # Extract text from URL
                            article_data = extract_text_from_url(url)
                            
                            # Determine language
                            detected_language = detect_language(article_data['content'])
                            
                            article_data['url'] = url
                            article_data['language'] = detected_language
                            
                            # Load models
                            model_name, tokenizer_name = SENTIMENT_MODEL_OPTIONS[selected_sentiment_model]
                            sentiment_pipe = load_sentiment_pipeline(model_name, tokenizer_name)
                            
                            ner_model = ENTITY_MODEL_OPTIONS[selected_entity_model]
                            ner_pipe = load_ner_pipeline(ner_model)
                            
                            # Perform analysis
                            sentence_sentiments = analyze_sentiment_by_sentence(article_data['content'], sentiment_pipe)
                            overall_sentiment = calculate_overall_sentiment_from_sentences(sentence_sentiments)
                            entities = extract_entities(article_data['content'], ner_pipe)
                            
                            # Prepare results
                            analysis_results = {
                                'overall_sentiment': overall_sentiment,
                                'sentence_sentiments': sentence_sentiments,
                                'entities': entities,
                                'keywords': [],
                                'model_used': selected_sentiment_model
                            }
                            
                            # Save to database
                            article_id = save_to_db(conn, article_data, analysis_results)
                            
                            # Display results
                            st.success(f"Analysis complete! Article ID: {article_id}")
                            
                            # Display dashboard
                            st.subheader("Analysis Dashboard")
                            
                            # Top-level metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Language", detected_language.upper())
                            with col2:
                                st.metric("Overall Sentiment", overall_sentiment['label'])
                            with col3:
                                st.metric("Confidence", f"{overall_sentiment['score']:.2f}")
                            
                            # Article preview
                            st.subheader("Article Preview")
                            st.markdown(f"**Title:** {article_data['title']}")
                            st.markdown(f"**Source:** {article_data['source']}")
                            with st.expander("Show full content"):
                                st.write(article_data['content'])
                            
                            # Sentiment analysis
                            st.subheader("Sentiment Analysis")
                            
                            # Sentiment distribution
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                sentiment_card(overall_sentiment)
                            with col2:
                                sentiment_fig = plot_sentiment_distribution(sentence_sentiments)
                                st.plotly_chart(sentiment_fig, use_container_width=True)
                            
                            # Sentiment Statistics
                            st.subheader("Sentiment Statistics")
                            sentiment_counts = {
                                'POSITIVE': sum(1 for item in sentence_sentiments if item['sentiment'] == 'POSITIVE'),
                                'NEUTRAL': sum(1 for item in sentence_sentiments if item['sentiment'] == 'NEUTRAL'),
                                'NEGATIVE': sum(1 for item in sentence_sentiments if item['sentiment'] == 'NEGATIVE')
                            }
                            total_sentences = len(sentence_sentiments)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Positive", f"{sentiment_counts['POSITIVE']} ({sentiment_counts['POSITIVE']*100/total_sentences if total_sentences else 0:.1f}%)")
                            with col2:
                                st.metric("Neutral", f"{sentiment_counts['NEUTRAL']} ({sentiment_counts['NEUTRAL']*100/total_sentences if total_sentences else 0:.1f}%)")
                            with col3:
                                st.metric("Negative", f"{sentiment_counts['NEGATIVE']} ({sentiment_counts['NEGATIVE']*100/total_sentences if total_sentences else 0:.1f}%)")
                            
                            # Sentence-level sentiment
                            st.subheader("Sentence-level Sentiment")
                            df_sentences = pd.DataFrame(sentence_sentiments)
                            if not df_sentences.empty:
                                for _, row in df_sentences.iterrows():
                                    sentiment_color = "green" if row['sentiment'] == "POSITIVE" else "red" if row['sentiment'] == "NEGATIVE" else "gray"
                                    st.markdown(f"""
                                    <div style="padding: 0.5rem; border-radius: 0.25rem; background-color: {sentiment_color}15; margin-bottom: 0.5rem;">
                                        <p style="margin-bottom: 0.25rem;">{row['sentence']}</p>
                                        <p style="margin: 0; font-size: 0.8rem; color: {sentiment_color};">{row['sentiment']} ({row['score']:.2f})</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Entity extraction
                            st.subheader("Named Entity Recognition")
                            if entities:
                                entity_fig = plot_entities(entities)
                                st.plotly_chart(entity_fig, use_container_width=True)
                                
                                # Display entity list
                                st.markdown("### Detected Entities")
                                for entity_type in set(entity['entity_type'] for entity in entities):
                                    with st.expander(f"{entity_type}"):
                                        filtered_entities = [e for e in entities if e['entity_type'] == entity_type]
                                        filtered_entities.sort(key=lambda x: x['count'], reverse=True)
                                        
                                        for entity in filtered_entities[:10]:  # Show top 10
                                            st.markdown(f"- **{entity['word']}** (Score: {entity['score']:.2f}, Count: {entity['count']})")
                            else:
                                st.info("No entities detected in this content.")
                            
                            # Model Evaluation Section
                            st.subheader("Model Evaluation")
                            st.markdown("Please rate the performance of the models used in this analysis:")

                            col1, col2 = st.columns(2)
                            with col1:
                                # Đánh giá mô hình sentiment
                                st.markdown(f"**Sentiment Analysis Model:** {selected_sentiment_model}")
                                sentiment_rating = st.slider("Rate sentiment analysis accuracy (1-5)", 1, 5, 4, key="sentiment_rating")
                                sentiment_accuracy = st.slider("Sentiment Accuracy", 0.0, 1.0, 0.8, 0.01, key="sentiment_accuracy", format="%.2f")
                                sentiment_precision = st.slider("Sentiment Precision", 0.0, 1.0, 0.75, 0.01, key="sentiment_precision", format="%.2f")
                                sentiment_recall = st.slider("Sentiment Recall", 0.0, 1.0, 0.78, 0.01, key="sentiment_recall", format="%.2f")
                                sentiment_f1 = st.slider("Sentiment F1-Score", 0.0, 1.0, 0.76, 0.01, key="sentiment_f1", format="%.2f")
                                sentiment_feedback = st.text_area("Feedback on sentiment analysis", key="sentiment_feedback")

                            with col2:
                                # Đánh giá mô hình entity
                                st.markdown(f"**Entity Recognition Model:** {selected_entity_model}")
                                entity_rating = st.slider("Rate entity recognition accuracy (1-5)", 1, 5, 4, key="entity_rating")
                                entity_accuracy = st.slider("Entity Accuracy", 0.0, 1.0, 0.83, 0.01, key="entity_accuracy", format="%.2f")
                                entity_precision = st.slider("Entity Precision", 0.0, 1.0, 0.79, 0.01, key="entity_precision", format="%.2f")
                                entity_recall = st.slider("Entity Recall", 0.0, 1.0, 0.81, 0.01, key="entity_recall", format="%.2f")
                                entity_f1 = st.slider("Entity F1-Score", 0.0, 1.0, 0.80, 0.01, key="entity_f1", format="%.2f") 
                                entity_feedback = st.text_area("Feedback on entity recognition", key="entity_feedback")

                            # Nút lưu đánh giá
                            if st.button("Submit Evaluation"):
                                # Chuẩn bị dữ liệu đánh giá mô hình sentiment
                                sentiment_eval = {
                                    'model_name': selected_sentiment_model,
                                    'model_type': 'sentiment',
                                    'accuracy': sentiment_accuracy,
                                    'precision': sentiment_precision,
                                    'recall': sentiment_recall,
                                    'f1_score': sentiment_f1,
                                    'user_rating': sentiment_rating,
                                    'user_feedback': sentiment_feedback
                                }
                                
                                # Chuẩn bị dữ liệu đánh giá mô hình entity
                                entity_eval = {
                                    'model_name': selected_entity_model,
                                    'model_type': 'entity',
                                    'accuracy': entity_accuracy,
                                    'precision': entity_precision,
                                    'recall': entity_recall,
                                    'f1_score': entity_f1,
                                    'user_rating': entity_rating,
                                    'user_feedback': entity_feedback
                                }
                                
                                # Lưu đánh giá vào database
                                sentiment_eval_id = save_model_evaluation(conn, sentiment_eval)
                                entity_eval_id = save_model_evaluation(conn, entity_eval)
                                
                                if sentiment_eval_id and entity_eval_id:
                                    st.success("Thank you for your evaluation! Your feedback helps improve our models.")
                                else:
                                    st.error("An error occurred while saving your evaluation. Please try again.")
                        
                        except Exception as e:
                            st.error(f"Error processing URL: {e}")
        
        elif input_method == "Text Input":
            # Text input analysis implementation
            text_input = st.text_area("Enter Text for Analysis", height=200)
            
            col1, col2 = st.columns(2)
            with col1:
                selected_sentiment_model = st.selectbox(
                    "Select Sentiment Analysis Model", 
                    list(SENTIMENT_MODEL_OPTIONS.keys())
                )
            with col2:
                selected_entity_model = st.selectbox(
                    "Select Entity Recognition Model", 
                    list(ENTITY_MODEL_OPTIONS.keys())
                )
            
            if st.button("Analyze Text"):
                if not text_input:
                    st.error("Please enter some text to analyze!")
                else:
                    with st.spinner("Analyzing text..."):
                        try:
                            # Determine language
                            detected_language = detect_language(text_input)
                            
                            # Create article data
                            article_data = {
                                'title': "Manual Text Input",
                                'content': text_input,
                                'source': "Text Input",
                                'url': "manual_input",
                                'language': detected_language
                            }
                            
                            # Load models
                            model_name, tokenizer_name = SENTIMENT_MODEL_OPTIONS[selected_sentiment_model]
                            sentiment_pipe = load_sentiment_pipeline(model_name, tokenizer_name)
                            
                            ner_model = ENTITY_MODEL_OPTIONS[selected_entity_model]
                            ner_pipe = load_ner_pipeline(ner_model)
                            
                            # Perform analysis
                            sentence_sentiments = analyze_sentiment_by_sentence(text_input, sentiment_pipe)
                            overall_sentiment = calculate_overall_sentiment_from_sentences(sentence_sentiments)
                            entities = extract_entities(text_input, ner_pipe)
                            
                            # Prepare results
                            analysis_results = {
                                'overall_sentiment': overall_sentiment,
                                'sentence_sentiments': sentence_sentiments,
                                'entities': entities,
                                'keywords': [],
                                'model_used': selected_sentiment_model
                            }
                            
                            # Save to database
                            article_id = save_to_db(conn, article_data, analysis_results)
                            
                            # Display results
                            st.success(f"Analysis complete! Article ID: {article_id}")
                            
                            # Display dashboard
                            st.subheader("Analysis Dashboard")
                            
                            # Top-level metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Language", detected_language.upper())
                            with col2:
                                st.metric("Overall Sentiment", overall_sentiment['label'])
                            with col3:
                                st.metric("Confidence", f"{overall_sentiment['score']:.2f}")
                            
                            # Sentiment analysis
                            st.subheader("Sentiment Analysis")
                            
                            # Sentiment distribution
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                sentiment_card(overall_sentiment)
                            with col2:
                                sentiment_fig = plot_sentiment_distribution(sentence_sentiments)
                                st.plotly_chart(sentiment_fig, use_container_width=True)
                            
                            # Sentiment Statistics
                            st.subheader("Sentiment Statistics")
                            sentiment_counts = {
                                'POSITIVE': sum(1 for item in sentence_sentiments if item['sentiment'] == 'POSITIVE'),
                                'NEUTRAL': sum(1 for item in sentence_sentiments if item['sentiment'] == 'NEUTRAL'),
                                'NEGATIVE': sum(1 for item in sentence_sentiments if item['sentiment'] == 'NEGATIVE')
                            }
                            total_sentences = len(sentence_sentiments)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Positive", f"{sentiment_counts['POSITIVE']} ({sentiment_counts['POSITIVE']*100/total_sentences if total_sentences else 0:.1f}%)")
                            with col2:
                                st.metric("Neutral", f"{sentiment_counts['NEUTRAL']} ({sentiment_counts['NEUTRAL']*100/total_sentences if total_sentences else 0:.1f}%)")
                            with col3:
                                st.metric("Negative", f"{sentiment_counts['NEGATIVE']} ({sentiment_counts['NEGATIVE']*100/total_sentences if total_sentences else 0:.1f}%)")
                            
                            # Sentence-level sentiment
                            st.subheader("Sentence-level Sentiment")
                            df_sentences = pd.DataFrame(sentence_sentiments)
                            if not df_sentences.empty:
                                for _, row in df_sentences.iterrows():
                                    sentiment_color = "green" if row['sentiment'] == "POSITIVE" else "red" if row['sentiment'] == "NEGATIVE" else "gray"
                                    st.markdown(f"""
                                    <div style="padding: 0.5rem; border-radius: 0.25rem; background-color: {sentiment_color}15; margin-bottom: 0.5rem;">
                                        <p style="margin-bottom: 0.25rem;">{row['sentence']}</p>
                                        <p style="margin: 0; font-size: 0.8rem; color: {sentiment_color};">{row['sentiment']} ({row['score']:.2f})</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Entity extraction
                            st.subheader("Named Entity Recognition")
                            if entities:
                                entity_fig = plot_entities(entities)
                                st.plotly_chart(entity_fig, use_container_width=True)
                                
                                # Display entity list
                                st.markdown("### Detected Entities")
                                for entity_type in set(entity['entity_type'] for entity in entities):
                                    with st.expander(f"{entity_type}"):
                                        filtered_entities = [e for e in entities if e['entity_type'] == entity_type]
                                        filtered_entities.sort(key=lambda x: x['count'], reverse=True)
                                        
                                        for entity in filtered_entities[:10]:  # Show top 10
                                            st.markdown(f"- **{entity['word']}** (Score: {entity['score']:.2f}, Count: {entity['count']})")
                            else:
                                st.info("No entities detected in this content.")
                        
                        except Exception as e:
                            st.error(f"Error processing text: {e}")
            
        elif input_method == "Bulk Analysis":
            st.info("Bulk analysis allows you to analyze multiple URLs at once.")
            
            urls = st.text_area("Enter URLs (one per line)", height=150)
            
            col1, col2 = st.columns(2)
            with col1:
                selected_sentiment_model = st.selectbox(
                    "Select Sentiment Analysis Model", 
                    list(SENTIMENT_MODEL_OPTIONS.keys())
                )
            with col2:
                selected_entity_model = st.selectbox(
                    "Select Entity Recognition Model", 
                    list(ENTITY_MODEL_OPTIONS.keys())
                )
            
            if st.button("Start Bulk Analysis"):
                if not urls:
                    st.error("Please enter at least one URL!")
                else:
                    url_list = [url.strip() for url in urls.split('\n') if url.strip()]
                    
                    if not url_list:
                        st.error("No valid URLs found!")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        
                        # Load models
                        model_name, tokenizer_name = SENTIMENT_MODEL_OPTIONS[selected_sentiment_model]
                        sentiment_pipe = load_sentiment_pipeline(model_name, tokenizer_name)
                        
                        ner_model = ENTITY_MODEL_OPTIONS[selected_entity_model]
                        ner_pipe = load_ner_pipeline(ner_model)
                        
                        for i, url in enumerate(url_list):
                            status_text.text(f"Processing {i+1}/{len(url_list)}: {url}")
                            
                            try:
                                # Extract text from URL
                                article_data = extract_text_from_url(url)
                                
                                # Determine language
                                detected_language = detect_language(article_data['content'])
                                
                                article_data['url'] = url
                                article_data['language'] = detected_language
                                
                                # Perform analysis
                                sentence_sentiments = analyze_sentiment_by_sentence(article_data['content'], sentiment_pipe)
                                overall_sentiment = calculate_overall_sentiment_from_sentences(sentence_sentiments)
                                entities = extract_entities(article_data['content'], ner_pipe)
                                
                                # Prepare results
                                analysis_results = {
                                    'overall_sentiment': overall_sentiment,
                                    'sentence_sentiments': sentence_sentiments,
                                    'entities': entities,
                                    'keywords': [],
                                    'model_used': selected_sentiment_model
                                }
                                
                                # Save to database
                                article_id = save_to_db(conn, article_data, analysis_results)
                                
                                results.append({
                                    'id': article_id,
                                    'url': url,
                                    'title': article_data['title'],
                                    'language': detected_language,
                                    'sentiment': overall_sentiment['label'],
                                    'score': overall_sentiment['score'],
                                    'status': 'Success'
                                })
                                
                            except Exception as e:
                                results.append({
                                    'id': None,
                                    'url': url,
                                    'title': None,
                                    'language': None,
                                    'sentiment': None,
                                    'score': None,
                                    'status': f'Error: {str(e)}'
                                })
                            
                            progress_bar.progress((i + 1) / len(url_list))
                        
                        status_text.text("Analysis complete!")
                        
                        # Display results
                        st.subheader("Bulk Analysis Results")
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)
                        
                        # Export options
                        if st.button("Export Results"):
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"bulk_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
    
    elif page == "History":
        st.title("📜 Analysis History")
        
        # Search options
        st.subheader("Search History")
        search_term = st.text_input("Search by URL or Title")
        
        try:
            if search_term:
                if DB_TYPE == "postgres":
                    search_query = f"""
                        SELECT id, url, title, language, sentiment_score, sentiment_label, source, fetch_date
                        FROM articles
                        WHERE url LIKE %s OR title LIKE %s
                        ORDER BY fetch_date DESC
                    """
                    history_df = pd.read_sql_query(search_query, conn, params=(f"%{search_term}%", f"%{search_term}%"))
                else:
                    search_query = f"""
                        SELECT id, url, title, language, sentiment_score, sentiment_label, source, fetch_date
                        FROM articles
                        WHERE url LIKE ? OR title LIKE ?
                        ORDER BY fetch_date DESC
                    """
                    history_df = pd.read_sql_query(search_query, conn, params=(f"%{search_term}%", f"%{search_term}%"))
            else:
                history_df = get_analysis_history(conn, limit=50)
            
            if not history_df.empty:
                st.dataframe(history_df)
                
                # Allow selection of an article to view full analysis
                selected_article_id = st.selectbox("Select an article to view full analysis", 
                                                  history_df['id'].tolist(),
                                                  format_func=lambda x: history_df[history_df['id']==x]['title'].iloc[0])
                
                if st.button("View Full Analysis"):
                    # Fetch the complete article data
                    article = get_article_by_id(conn, selected_article_id)
                    
                    if article:
                        # Display full analysis dashboard - similar to the Analysis page
                        st.subheader("Analysis Dashboard")
                        
                        # Top-level metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Language", article['language'].upper())
                        with col2:
                            st.metric("Overall Sentiment", article['sentiment_label'])
                        with col3:
                            st.metric("Confidence", f"{article['sentiment_score']:.2f}")
                        
                        # Article preview
                        st.subheader("Article Content")
                        st.markdown(f"**Title:** {article['title']}")
                        st.markdown(f"**Source:** {article['source']}")
                        with st.expander("Show full content"):
                            st.write(article['content'])
                        
                        # Sentiment analysis
                        st.subheader("Sentiment Analysis")
                        
                        # Sentiment distribution
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            sentiment_card({'label': article['sentiment_label'], 'score': article['sentiment_score']})
                        
                        with col2:
                            # Fetch the detailed sentiment analysis history
                            result = get_analysis_result(conn, selected_article_id, 'sentiment')
                            sentence_sentiments = result if result else []
                            sentiment_fig = plot_sentiment_distribution(sentence_sentiments)
                            st.plotly_chart(sentiment_fig, use_container_width=True)
                        
                        # Sentence-level sentiment
                        if sentence_sentiments:
                            # Sentiment Statistics
                            st.subheader("Sentiment Statistics")
                            sentiment_counts = {
                                'POSITIVE': sum(1 for item in sentence_sentiments if item['sentiment'] == 'POSITIVE'),
                                'NEUTRAL': sum(1 for item in sentence_sentiments if item['sentiment'] == 'NEUTRAL'),
                                'NEGATIVE': sum(1 for item in sentence_sentiments if item['sentiment'] == 'NEGATIVE')
                            }
                            total_sentences = len(sentence_sentiments)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Positive", f"{sentiment_counts['POSITIVE']} ({sentiment_counts['POSITIVE']*100/total_sentences if total_sentences else 0:.1f}%)")
                            with col2:
                                st.metric("Neutral", f"{sentiment_counts['NEUTRAL']} ({sentiment_counts['NEUTRAL']*100/total_sentences if total_sentences else 0:.1f}%)")
                            with col3:
                                st.metric("Negative", f"{sentiment_counts['NEGATIVE']} ({sentiment_counts['NEGATIVE']*100/total_sentences if total_sentences else 0:.1f}%)")
                            
                            st.subheader("Sentence-level Sentiment")
                            for item in sentence_sentiments:
                                sentiment_color = "green" if item['sentiment'] == "POSITIVE" else "red" if item['sentiment'] == "NEGATIVE" else "gray"
                                st.markdown(f"""
                                <div style="padding: 0.5rem; border-radius: 0.25rem; background-color: {sentiment_color}15; margin-bottom: 0.5rem;">
                                    <p style="margin-bottom: 0.25rem;">{item['sentence']}</p>
                                    <p style="margin: 0; font-size: 0.8rem; color: {sentiment_color};">{item['sentiment']} ({item['score']:.2f})</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Entity extraction
                        st.subheader("Named Entity Recognition")
                        if article['entities']:
                            try:
                                # Nếu là chuỗi JSON
                                entities = json.loads(article['entities'])
                            except (TypeError, json.JSONDecodeError):
                                # Nếu đã là object Python
                                entities = article['entities']
                        else:
                            entities = []
                        
                        if entities:
                            entity_fig = plot_entities(entities)
                            st.plotly_chart(entity_fig, use_container_width=True)
                            
                            # Display entity list
                            st.markdown("### Detected Entities")
                            for entity_type in set(entity['entity_type'] for entity in entities):
                                with st.expander(f"{entity_type}"):
                                    filtered_entities = [e for e in entities if e['entity_type'] == entity_type]
                                    filtered_entities.sort(key=lambda x: x['count'], reverse=True)
                                    
                                    for entity in filtered_entities[:10]:  # Show top 10
                                        st.markdown(f"- **{entity['word']}** (Score: {entity['score']:.2f}, Count: {entity['count']})")
                        else:
                            st.info("No entities detected in this content.")
            else:
                st.info("No analysis history found.")
                
            # Export full history
            if st.button("Export Full History"):
                full_history = pd.read_sql_query("SELECT * FROM articles", conn)
                csv = full_history.to_csv(index=False)
                st.download_button(
                    label="Download Full History CSV",
                    data=csv,
                    file_name=f"analysis_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error retrieving history: {e}")
    
    elif page == "Settings":
        st.title("⚙️ Settings")
        
        # Database status
        st.subheader("Database Status")
        
        # Lấy thông tin database
        db_info = get_database_info(conn)
        
        st.json(db_info)
        
        # Tùy chọn sao lưu (chỉ cho SQLite)
        if DB_TYPE == "sqlite":
            if st.button("Backup Database"):
                if backup_database():
                    st.success("Database backup created successfully!")
                else:
                    st.error("Failed to create database backup.")
        
        # Database management
        st.subheader("Database Management")
        with st.expander("Danger Zone"):
            st.warning("The following actions are destructive and cannot be undone.")
            confirm = st.checkbox("I understand that this action cannot be undone")
            
            if st.button("Clear All Data", disabled=not confirm):
                if clear_database(conn, confirm=confirm):
                    st.success("All data has been cleared from the database.")
                else:
                    st.error("Failed to clear database data.")
        
        # Tabs cho Settings
        db_tab, models_tab, advanced_tab = st.tabs(["Database", "Models", "Advanced"])
        
        # Tab Database Settings
        with db_tab:
            st.subheader("Database Management")
            
            # Hiển thị thông tin kết nối
            st.write("Database connection information:")
            db_info = {
                "Type": "SQLite",
                "Path": str(SQLITE_PATH).replace("\\", "\\\\")  # Escape backslashes
            }
            st.json(db_info)
            
            # Lấy thông tin về kích thước database
            try:
                # SQLite cursor không hỗ trợ context manager (with)
                cur = conn.cursor()
                
                # Số lượng bài viết
                cur.execute("SELECT COUNT(*) FROM articles")
                article_count = cur.fetchone()[0]  # Với SQLite, đây sẽ là phần tử đầu tiên của tuple
                
                # Số lượng phân tích
                cur.execute("SELECT COUNT(*) FROM analysis_history")
                analysis_count = cur.fetchone()[0]
                
                # Database stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Articles", article_count)
                with col2:
                    st.metric("Analysis Records", analysis_count)
            except Exception as e:
                st.error(f"Error fetching database stats: {e}")
        
        # Tab Model Settings
        with models_tab:
            st.subheader("Model Settings")
            
            # Tabs cho settings và evaluations
            model_settings_tab, model_eval_tab = st.tabs(["Model Settings", "Model Evaluations"])
            
            with model_settings_tab:
                # Settings cài đặt mô hình mặc định
                default_sentiment = st.selectbox(
                    "Default Sentiment Analysis Model",
                    list(SENTIMENT_MODEL_OPTIONS.keys()),
                    index=0
                )
                
                default_entity = st.selectbox(
                    "Default Entity Recognition Model",
                    list(ENTITY_MODEL_OPTIONS.keys()),
                    index=0
                )
                
                # Lưu cài đặt
                if st.button("Save Model Settings"):
                    # Trong ứng dụng thực tế, bạn sẽ lưu cài đặt này vào database hoặc file cấu hình
                    st.success("Settings saved successfully!")
            
            with model_eval_tab:
                st.subheader("Model Evaluation History")
                
                # Filter options
                model_type_filter = st.radio("Model Type", ["All", "Sentiment", "Entity"], horizontal=True)
                
                # Get evaluation data
                if model_type_filter == "All":
                    evals_df = get_model_evaluations(conn)
                elif model_type_filter == "Sentiment":
                    evals_df = get_model_evaluations(conn, model_type="sentiment")
                else:
                    evals_df = get_model_evaluations(conn, model_type="entity")
                
                if evals_df is not None and not evals_df.empty:
                    # Display metrics chart
                    metrics_fig = plot_model_metrics(evals_df)
                    if metrics_fig:
                        st.plotly_chart(metrics_fig, use_container_width=True)
                    
                    # Display user ratings chart
                    ratings_fig = plot_user_ratings(evals_df)
                    if ratings_fig:
                        st.plotly_chart(ratings_fig, use_container_width=True)
                    
                    # Show detailed evaluations in a table
                    st.subheader("Detailed Evaluations")
                    st.dataframe(
                        evals_df[['model_name', 'model_type', 'accuracy', 'precision', 'recall', 'f1_score', 'user_rating', 'evaluation_date']], 
                        use_container_width=True
                    )
                    
                    # Export evaluations
                    if st.button("Export Evaluations"):
                        csv = evals_df.to_csv(index=False)
                        st.download_button(
                            label="Download Evaluations CSV",
                            data=csv,
                            file_name=f"model_evaluations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("No model evaluations available. Evaluate models after analysis to see data here.")
        
        # Tab Advanced Settings
        with advanced_tab:
            st.subheader("Advanced Configuration")
            
            with st.expander("API Configuration"):
                st.text_input("API Key (if needed)", type="password")
                st.number_input("Request Timeout (seconds)", min_value=5, max_value=60, value=15)
            
            with st.expander("Crawler Settings"):
                st.checkbox("Enable JavaScript rendering", value=False)
                st.number_input("Crawl depth", min_value=1, max_value=5, value=1)
                st.text_input("User Agent", value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            with st.expander("Performance Settings"):
                st.slider("Cache TTL (minutes)", min_value=5, max_value=120, value=30)
                st.checkbox("Enable batch processing", value=True)
    
    # Close database connection when app closes
    st.session_state.on_exit = lambda: conn.close()

if __name__ == "__main__":
    main() 