# wordcloud_generator.py

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

def show_wordcloud(phrases, title):
    """
    Generate and display a word cloud from a list of phrases.
    """
    text = " ".join(phrases)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=set(stopwords.words('english')),
        colormap='Dark2'
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.markdown(f"### {title}")
    st.pyplot(fig)
