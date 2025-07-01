import sys
import types
import torch
sys.modules["torch.classes"] = types.SimpleNamespace()
import json

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from collections import defaultdict
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import os
import base64

# Safe kaleido initialization
try:
    import kaleido
    pio.kaleido.scope.default_format = "png"
    pio.kaleido.scope.default_engine = "kaleido"
except:
    pass  # Will use alternative method

from emotion_detector import classify_emotions
from topic_extractor import extract_topics
from score_calculator import calculate_adorescore
from wordcloud_generator import show_wordcloud

st.set_page_config(page_title="Sentilytics", layout="wide")
st.title("Sentilytics: AI-powered engine for analyzing customer feedback")

# Initialize session state with proper defaults
def initialize_session_state():
    """Initialize all session state variables with proper defaults"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'result_df' not in st.session_state:
        st.session_state.result_df = None
    if 'figures' not in st.session_state:
        st.session_state.figures = {}
    if 'emotion_phrases' not in st.session_state:
        st.session_state.emotion_phrases = defaultdict(list)
    if 'topic_phrases' not in st.session_state:
        st.session_state.topic_phrases = defaultdict(list)
    if 'display_cols' not in st.session_state:
        st.session_state.display_cols = []
    if 'emotion_wordclouds' not in st.session_state:
        st.session_state.emotion_wordclouds = {}
    if 'topic_wordclouds' not in st.session_state:
        st.session_state.topic_wordclouds = {}

# Initialize session state
initialize_session_state()

uploaded_file = st.file_uploader("📂 Upload a CSV with `comment` column (optional: `timestamp`)", type=["csv"])
use_timestamp = st.toggle("🕒 Use Timestamp for Trend Analysis")

def save_plotly_fig(fig, filename):
    """Save plotly figure as PNG image with multiple fallback methods"""
    try:
        # Method 1: Try kaleido
        fig.write_image(filename, width=800, height=500, engine="kaleido")
        return True
    except Exception as e1:
        try:
            # Method 2: Try orca (if available)
            fig.write_image(filename, width=800, height=500, engine="orca")
            return True
        except Exception as e2:
            try:
                # Method 3: Try plotly's to_image
                img_bytes = fig.to_image(format="png", width=800, height=500)
                with open(filename, "wb") as f:
                    f.write(img_bytes)
                return True
            except Exception as e3:
                try:
                    # Method 4: Use html to image conversion
                    html_str = fig.to_html(include_plotlyjs='cdn')
                    # This is a fallback - you might need to install additional packages
                    st.warning(f"Using fallback method for chart: {filename}")
                    return False
                except Exception as e4:
                    st.error(f"All image export methods failed: {e4}")
                    return False

def save_wordcloud_image(phrases, title, filename):
    """Generate and save wordcloud as image"""
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        if not phrases:
            return False
            
        # Combine all phrases
        text = ' '.join(phrases)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(text)
        
        # Create matplotlib figure
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(filename, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return True
    except Exception as e:
        st.warning(f"Could not generate wordcloud for {title}: {e}")
        return False

def create_summary_stats(result_df):
    """Create summary statistics for the report"""
    stats = {
        "Total Comments": len(result_df),
        "Average Adorescore": f"{result_df['adorescore'].mean():.2f}",
        "Highest Adorescore": f"{result_df['adorescore'].max():.2f}",
        "Lowest Adorescore": f"{result_df['adorescore'].min():.2f}",
    }
    
    # Most common emotion
    emotion_counter = defaultdict(float)
    for emo_list in result_df["emotions"]:
        for emo in emo_list:
            emotion_counter[emo["emotion"]] += emo["intensity"]
    
    if emotion_counter:
        most_common_emotion = max(emotion_counter, key=emotion_counter.get)
        stats["Most Common Emotion"] = most_common_emotion
    
    # Most common topic
    topic_counter = defaultdict(int)
    for topics in result_df["main_topics"]:
        for topic in topics:
            topic_counter[topic] += 1
    
    if topic_counter:
        most_common_topic = max(topic_counter, key=topic_counter.get)
        stats["Most Common Topic"] = most_common_topic
    
    return stats

def generate_pdf_report(charts_paths, table_data, summary_stats, filename="feedback_report.pdf"):
    """Generate comprehensive PDF report"""
    try:
        doc = SimpleDocTemplate(filename, pagesize=A4, 
                              rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        story.append(Paragraph("📊 Customer Feedback Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Summary Statistics
        story.append(Paragraph("📈 Executive Summary", styles["Heading2"]))
        story.append(Spacer(1, 12))
        
        # Create summary table
        summary_data = [["Metric", "Value"]]
        for key, value in summary_stats.items():
            summary_data.append([key, str(value)])
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Charts
        story.append(Paragraph("📊 Visual Analysis", styles["Heading2"]))
        story.append(Spacer(1, 12))
        
        for chart_title, img_path in charts_paths:
            if os.path.exists(img_path):
                story.append(Paragraph(chart_title, styles["Heading3"]))
                story.append(Spacer(1, 6))
                
                # Add image with proper sizing
                img = RLImage(img_path, width=6*inch, height=3.75*inch)
                story.append(img)
                story.append(Spacer(1, 15))
            else:
                story.append(Paragraph(f"Chart not available: {chart_title}", styles["Normal"]))
                story.append(Spacer(1, 12))
        
        # Data Table
        story.append(Paragraph("🧾 Detailed Feedback Data (First 10 Rows)", styles["Heading2"]))
        story.append(Spacer(1, 12))
        
        # Prepare table data
        headers = list(table_data.columns)
        sample_data = table_data.head(10)
        
        # Convert data to strings and handle long text
        table_rows = [headers]
        for _, row in sample_data.iterrows():
            formatted_row = []
            for col in headers:
                cell_value = str(row[col])
                # Truncate long text
                if len(cell_value) > 50:
                    cell_value = cell_value[:47] + "..."
                formatted_row.append(cell_value)
            table_rows.append(formatted_row)
        
        # Create table with dynamic column widths
        col_width = 7*inch / len(headers)
        col_widths = [col_width] * len(headers)
        
        data_table = Table(table_rows, colWidths=col_widths, repeatRows=1)
        data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(data_table)
        story.append(Spacer(1, 12))
        story.append(Paragraph("📌 Note: Only first 10 rows shown. Complete data available in source file.", 
                              styles["Normal"]))
        
        # Build PDF
        doc.build(story)
        return True
    except Exception as e:
        st.error(f"Error building PDF: {e}")
        return False

def perform_analysis(df, use_timestamp):
    """Perform the emotion analysis and store results in session state"""
    try:
        all_results = []
        emotion_phrases = defaultdict(list)
        topic_phrases = defaultdict(list)

        progress_bar = st.progress(0)
        status_text = st.empty()
        total_comments = len(df["comment"])

        for i, comment in df["comment"].items():
            status_text.text(f'Processing comment {i+1} of {total_comments}...')
            
            emotions = classify_emotions(comment)
            topics, subtopics = extract_topics(comment)
            adorescore = calculate_adorescore(emotions, topics)
            # print(emotions,topics,subtopics,adorescore)

            for emo in emotions:
                emotion_phrases[emo["emotion"]].append(comment)
            for topic in topics:
                topic_phrases[topic].append(comment)

            all_results.append({
                "timestamp": df["timestamp"][i] if use_timestamp else None,
                "comment": comment,
                "emotions": emotions,
                "main_topics": topics,
                "subtopics": subtopics,
                "adorescore": adorescore["overall"],
                "breakdown": adorescore["breakdown"]
            })
            
            # Update progress
            progress_bar.progress((len(all_results)) / total_comments)

        progress_bar.empty()
        status_text.empty()
        
        # Store in session state
        st.session_state.result_df = pd.DataFrame(all_results).fillna("")
        st.session_state.emotion_phrases = emotion_phrases
        st.session_state.topic_phrases = topic_phrases
        st.session_state.display_cols = ["comment", "main_topics", "subtopics", "adorescore"]
        if use_timestamp:
            st.session_state.display_cols.insert(0, "timestamp")
        
        st.session_state.analysis_complete = True
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.session_state.analysis_complete = False

def create_and_display_charts(result_df, use_timestamp):
    """Create and display all charts, storing them in session state"""
    
    # Ensure figures dictionary exists
    if 'figures' not in st.session_state or st.session_state.figures is None:
        st.session_state.figures = {}
    
    try:
        # Emotion Distribution Chart
        st.subheader("🎭 Emotion Distribution")
        emotion_counter = defaultdict(float)
        for emo_list in result_df["emotions"]:
            for emo in emo_list:
                emotion_counter[emo["emotion"]] += emo["intensity"]

        if emotion_counter:
            emo_df = pd.DataFrame({
                "emotion": list(emotion_counter.keys()),
                "intensity": list(emotion_counter.values())
            })

            # Only keep emotions with intensity > 0
            emo_df = emo_df[emo_df["intensity"] > 0].sort_values(by="intensity", ascending=False)

            if not emo_df.empty:
                fig1 = px.bar(
                    emo_df, x="emotion", y="intensity", color="emotion",
                    title="Emotion Intensity Distribution (Only > 0)",
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig1.update_layout(showlegend=False, title_x=0.5)
                fig1.update_traces(texttemplate='%{y:.1f}', textposition='outside')

                st.plotly_chart(fig1, use_container_width=True)
                st.session_state.figures["fig1"] = fig1
            else:
                st.warning("No emotions with intensity > 0 to display.")


        # Topic Frequency Chart
        st.subheader("🧩 Topic Frequency")
        topic_counter = defaultdict(int)
        for topics in result_df["main_topics"]:
            for topic in topics:
                topic_counter[topic] += 1

        if topic_counter:
            topic_df = pd.DataFrame({
                "topic": list(topic_counter.keys()), 
                "count": list(topic_counter.values())
            }).sort_values(by="count", ascending=False)
            
            fig2 = px.bar(topic_df, x="topic", y="count", color="topic", 
                         title="Topic Frequency Analysis", 
                         template="plotly_white",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(showlegend=False, title_x=0.5)
            
            # Add value labels on top of bars
            fig2.update_traces(texttemplate='%{y}', textposition='outside')
            
            st.plotly_chart(fig2, use_container_width=True)
            st.session_state.figures["fig2"] = fig2

        # Adorescore Distribution Chart
        st.subheader("📊 Adorescore Distribution")
        fig3 = px.histogram(result_df, x="adorescore", nbins=20,
                           title="Adorescore Distribution", 
                           template="plotly_white",
                           color_discrete_sequence=["#FF6B6B"])
        fig3.update_layout(title_x=0.5)
        st.plotly_chart(fig3, use_container_width=True)
        st.session_state.figures["fig3"] = fig3

        # Trend Chart (if timestamp is used)
        if use_timestamp and "timestamp" in result_df.columns:
            st.subheader("📈 Adorescore Trend Over Time")
            trend_df = result_df.copy()
            trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"])
            trend_df = trend_df.sort_values("timestamp")
            fig4 = px.line(trend_df, x="timestamp", y="adorescore", 
                          title="Adorescore Trend Over Time", 
                          template="plotly_white",
                          line_shape="spline")
            fig4.update_layout(title_x=0.5)
            fig4.update_traces(line_color="#4CAF50", line_width=3)
            st.plotly_chart(fig4, use_container_width=True)
            st.session_state.figures["fig4"] = fig4
            
    except Exception as e:
        st.error(f"Error creating charts: {str(e)}")

def generate_and_download_pdf():
    """Generate PDF and provide download link"""
    if not st.session_state.result_df is not None:
        st.error("No analysis data available. Please run the analysis first.")
        return False
        
    result_df = st.session_state.result_df
    display_cols = st.session_state.display_cols
    figures = st.session_state.figures or {}
    emotion_wordclouds = st.session_state.get('emotion_wordclouds', {})
    topic_wordclouds = st.session_state.get('topic_wordclouds', {})
    
    with st.spinner("🔄 Generating PDF report..."):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                chart_images = []
                
                # Save charts as images
                for fig_key, fig in figures.items():
                    img_path = os.path.join(tmpdir, f"{fig_key}.png")
                    if save_plotly_fig(fig, img_path):
                        if fig_key == "fig1":
                            chart_images.append(("Emotion Intensity Distribution", img_path))
                        elif fig_key == "fig2":
                            chart_images.append(("Topic Frequency Analysis", img_path))
                        elif fig_key == "fig3":
                            chart_images.append(("Adorescore Distribution", img_path))
                        elif fig_key == "fig4":
                            chart_images.append(("Adorescore Trend Over Time", img_path))
                
                # Save emotion wordclouds
                for emotion, phrases in emotion_wordclouds.items():
                    wc_path = os.path.join(tmpdir, f"wordcloud_emotion_{emotion.replace(' ', '_')}.png")
                    if save_wordcloud_image(phrases, f"Word Cloud - {emotion}", wc_path):
                        chart_images.append((f"Word Cloud - {emotion} Emotion", wc_path))
                
                # Save topic wordclouds
                for topic, phrases in topic_wordclouds.items():
                    wc_path = os.path.join(tmpdir, f"wordcloud_topic_{topic.replace(' ', '_')}.png")
                    if save_wordcloud_image(phrases, f"Word Cloud - {topic}", wc_path):
                        chart_images.append((f"Word Cloud - {topic} Topic", wc_path))
                
                # Create summary statistics
                summary_stats = create_summary_stats(result_df)
                
                # Generate PDF
                pdf_path = os.path.join(tmpdir, "customer_feedback_report.pdf")
                
                success = generate_pdf_report(chart_images, result_df[display_cols], 
                                            summary_stats, pdf_path)
                
                if success and os.path.exists(pdf_path):
                    # Read PDF file
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                    
                    st.success("✅ PDF report generated successfully!")
                    st.info(f"📊 Report includes: {len(chart_images)} charts and visualizations")
                    
                    # Auto-download using download button
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_data,
                        file_name=f"customer_feedback_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
                    
                    st.balloons()
                    return True
                else:
                    st.error("❌ Failed to generate PDF report")
                    return False
                    
        except Exception as e:
            st.error(f"❌ Error generating PDF report: {str(e)}")
            st.info("💡 Make sure you have all required packages installed: reportlab, plotly, wordcloud, matplotlib")
            return False

# Main App Logic
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if "comment" not in df.columns:
            st.error("❌ CSV must contain a `comment` column.")
            st.stop()

        if use_timestamp and "timestamp" not in df.columns:
            st.warning("⏱ Timestamp column not found. Proceeding without it.")
            use_timestamp = False

        if use_timestamp and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

        # Analysis Button
        if st.button("🔍 Analyze Comments") or st.session_state.analysis_complete:
            
            # Perform analysis if not already done
            if not st.session_state.analysis_complete:
                st.info("Processing... Please wait ⏳")
                perform_analysis(df, use_timestamp)
            
            # Display results if analysis is complete
            if st.session_state.analysis_complete and st.session_state.result_df is not None:
                result_df = st.session_state.result_df
                display_cols = st.session_state.display_cols
                emotion_phrases = st.session_state.emotion_phrases
                topic_phrases = st.session_state.topic_phrases
                
                # Create and display charts
                create_and_display_charts(result_df, use_timestamp)

                # Display data table
                st.subheader("🧾 Feedback Analysis Results")
                st.dataframe(result_df[display_cols], use_container_width=True)

                # Word clouds
                st.subheader("🎨 Word Clouds by Emotion")
                emotion_wordclouds = {}
                for emo, phrases in emotion_phrases.items():
                    if phrases:
                        show_wordcloud(phrases, f"Common Phrases for Emotion: {emo}")
                        emotion_wordclouds[emo] = phrases

                st.subheader("🎨 Word Clouds by Topic")
                topic_wordclouds = {}
                for topic, phrases in topic_phrases.items():
                    if phrases:
                        show_wordcloud(phrases, f"Common Phrases for Topic: {topic}")
                        topic_wordclouds[topic] = phrases
                
                # Store wordcloud data in session state
                st.session_state.emotion_wordclouds = emotion_wordclouds
                st.session_state.topic_wordclouds = topic_wordclouds

                # PDF Generation Section
                st.subheader("📄 Generate PDF Report")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("🖨️ Create PDF Report", type="primary"):
                        generate_and_download_pdf()
                with col2:
                    st.info("📋 Report will include: Summary stats, all charts, and data table")
                

                st.subheader("📥 Download Raw Analysis Data as JSON")

                if st.session_state.result_df is not None:
                    try:
                        # Convert DataFrame rows to JSON serializable structure
                        raw_json_data = st.session_state.result_df.to_dict(orient="records")

                        # Clean/convert any non-serializable objects (e.g. torch.Tensor) in "emotions"
                        def serialize(item):
                            if isinstance(item, (float, int, str, list, dict)) or item is None:
                                return item
                            return str(item)  # fallback for any custom objects
                        
                        cleaned_data = json.loads(json.dumps(raw_json_data, default=serialize, indent=2))

                        raw_json_str = json.dumps(cleaned_data, indent=2)

                        st.download_button(
                            label="📤 Download as JSON",
                            data=raw_json_str,
                            file_name=f"feedback_analysis_raw_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        st.info("Includes all fields: emotions, topics, subtopics, and Adorescore breakdown.")
                    except Exception as e:
                        st.error(f"⚠️ Failed to prepare JSON file: {e}")

                    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    # Reset session state when no file is uploaded
    st.session_state.analysis_complete = False
    st.session_state.result_df = None
    st.session_state.figures = {}
    st.session_state.emotion_phrases = defaultdict(list)
    st.session_state.topic_phrases = defaultdict(list)
    st.session_state.display_cols = []
    st.session_state.emotion_wordclouds = {}
    st.session_state.topic_wordclouds = {}
    
    st.markdown("""
    ### 🚀 Welcome to Customer Feedback Analyzer!
    
    **To get started:**
    1. 📂 Upload a CSV file with a `comment` column
    2. 🕒 Optionally include a `timestamp` column for trend analysis
    3. 🔍 Click "Analyze Comments" to process your data
    4. 📄 Generate and download a comprehensive PDF report
    
    **Features:**
    - 🎭 Emotion analysis and visualization
    - 🧩 Topic extraction and frequency analysis  
    - 📊 Adorescore calculation and distribution
    - 🎨 Word clouds for emotions and topics
    - 📈 Trend analysis (with timestamps)
    - 📄 Professional PDF report generation
    """)