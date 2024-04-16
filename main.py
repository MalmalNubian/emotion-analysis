import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


st.set_page_config(layout="wide", page_title="Emotion Analysis in Tweets", page_icon=":bird:")
st.title("Emotion Analysis in Tweets")

st.markdown("""
<style>
.main-title {text-align: center; font-size: 2em; margin-bottom: 0.5em;}
.emotion-dataset-description {font-size: 1.1em;}
.sidebar-header {font-size: 1.25em; font-weight: bold; margin-top: 1em;}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### About This App")
st.sidebar.info("""
This Streamlit app is designed to analyze emotions in tweets. It offers features like emotion prediction, 
visualization of emotion distribution, and insights into the most common words used in different emotional contexts.
Designed by Yasmin.
""")

@st.cache_data
def load_data():
    df = pd.read_csv("./tweets.csv")
    emotion_mapping = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
    df['emotions'] = df['emotions'].map(emotion_mapping)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    return df

df = load_data()

with st.container():
    gap_size = "large"
    col1, col2 = st.columns([2, 1], gap=gap_size)

    with col1:
        st.markdown('<p style="font-size: 24px">Welcome to the \'Emotions\' dataset – a collection of English Twitter messages meticulously annotated with six fundamental emotions: anger, fear, joy, love, sadness, and surprise.</p>', unsafe_allow_html=True)

    with col2:
        st.image("tweeters.png", width=300)

with st.expander("See full data table"):
    st.write(df)

st.markdown("<br>", unsafe_allow_html=True)

stop_words = set(stopwords.words('english'))

custom_stop_words = [
    'feel', 'feeling', 'like', 'im', 'just', 'really', 'know', 'ive', 
    'bit', 'make', 'dont', 'time', 'little', 'people', 'want', 'think', 
    'life', 'things', 'way', 'will', 'still', 'one'
]

stop_words_list = list(stop_words)

st.sidebar.markdown('<div class="sidebar-header">Text Analysis Options</div>', unsafe_allow_html=True)
with st.container():
    emotion_counts = df['emotions'].value_counts().reset_index(name='Frequency')
    emotion_counts.rename(columns={'index': 'Emotion'}, inplace=True)
    if st.sidebar.checkbox('Show Emotion Counts'):
        st.write("## Emotion Counts")
    
        col_count = 3 
        cols = st.columns(col_count)
        for index, (emotion, count) in enumerate(emotion_counts.set_index('Emotion')['Frequency'].items()):
            cols[index % col_count].metric(label=emotion, value=f"{count:,}")
        st.markdown("<br>", unsafe_allow_html=True)

    
    if st.sidebar.checkbox('Show Distribution of Emotions'):
        st.markdown("## Distribution of Emotions")
        
        fig = px.pie(
        emotion_counts, 
        names='Emotion', 
        values='Frequency', 
        color='Emotion',
        color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_traces (
            textposition='inside', 
            textinfo='percent+label'
        )
        st.plotly_chart(fig)

@st.cache_data
def generate_wordclouds(df):
    custom_stop_words = ['feel', 'feeling', 'like', 'im', 'just', 'really', 'know', 'ive', 
                         'bit', 'make', 'dont', 'time', 'little', 'people', 'want', 'think', 
                         'life', 'things', 'way', 'will', 'still', 'one']
    all_stop_words = STOPWORDS.union(set(custom_stop_words))
    emotions = df['emotions'].unique()
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs = axs.flatten()
    for i, emotion in enumerate(emotions):
        words = ' '.join(df[df['emotions'] == emotion]['tweets'])
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=all_stop_words).generate(words)
        axs[i].imshow(wordcloud, interpolation='bilinear')
        axs[i].axis('off')
        axs[i].set_title(f'Word Cloud for {emotion}')
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def display_common_words(df):
    count_vect = CountVectorizer(stop_words='english', max_features=100)
    X_counts = count_vect.fit_transform(df['tweets'])
    sum_words = X_counts.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in count_vect.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    common_words_df = pd.DataFrame(words_freq, columns=['Word', 'Frequency']).head(20)
    return common_words_df

st.sidebar.markdown('<div class="sidebar-header">Advanced Text Analysis</div>', unsafe_allow_html=True)

if st.sidebar.checkbox('Show 20 Most Common Words'):
    st.markdown("## Most Common Words")
    common_words_df = display_common_words(df)
    st.write(common_words_df)


if st.sidebar.checkbox('Generate Emotion Word Clouds'):
    st.markdown("## Emotion Word Clouds")
    st.info('This is a visual representation of the most common words for each emotion. The size of each word indicates its frequency or importance. The stopwords have been removed.', icon="ℹ️")
    generate_wordclouds(df)


@st.cache_data
def train_and_save_model(df):
    vectorizer = TfidfVectorizer(stop_words=stop_words_list)
    X = vectorizer.fit_transform(df['tweets'])
    y = df['emotions']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")  
    
    return model, vectorizer

model, vectorizer = train_and_save_model(df)

st.title("Predicting emotions")
user_input = st.text_area("Enter text here to predict its emotion:")

if st.button('Predict'):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    st.write(f'Predicted Emotion: {prediction}')



