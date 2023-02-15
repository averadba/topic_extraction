import streamlit as st
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.models import CoherenceModel

# Define a function to tag comments with their corresponding topic
def tag_comments(data, topics, num_comments=1000):
    stop_words = set(stopwords.words('english'))
    tagged_comments = []
    texts = [word_tokenize(comment.lower()) for comment in data[:num_comments]]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=len(topics), id2word=dictionary, random_state=42)
    for i, comment in enumerate(data[:num_comments]):
        bow_vector = dictionary.doc2bow(word_tokenize(comment.lower()))
        topic = max(lda_model[bow_vector], key=lambda x: x[1])[0]
        tagged_comments.append((comment, topic))
    return tagged_comments

# Define a function to extract the topics using LDA and the most relevant phrase keyword for each topic
def extract_topics(data, num_topics=5, num_words=5, num_passes=100, min_prob=0.01):
    stop_words = set(stopwords.words('english'))
    texts = [word_tokenize(comment.lower()) for comment in data]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    print("Fitting LDA model...")
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42, iterations=num_passes, alpha='auto', eta='auto', minimum_probability=min_prob)

    # Find the most relevant phrase keyword for each topic
    phrase_keywords = []
    for topic in lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False):
        topic_words = [word[0] for word in topic[1]]
        bigram = models.phrases.Phraser(models.phrases.Phrases(texts))
        topic_bigrams = bigram[topic_words]
        topic_trigrams = models.phrases.Phraser(models.phrases.Phrases(topic_bigrams))
        topic_phrases = topic_trigrams[topic_bigrams]
        topic_phrases = [phrase for phrase in topic_phrases if phrase not in stop_words and not all(char in string.punctuation for char in phrase) and not phrase.isnumeric() and len(phrase) > 1]
        coherence_model_lda = CoherenceModel(model=lda_model, texts=[topic_phrases], dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        top_topic_words = lda_model.show_topic(topic[0], topn=5)
        phrase_keyword = None
        if (phrase_keyword, 0) in top_topic_words:
            top_topic_words.remove((phrase_keyword, 0))
        if len(top_topic_words) > 0:
            max_val = max(top_topic_words, key=lambda x: x[1])
            while max_val[0] in stop_words:
                top_topic_words.remove(max_val)
                if len(top_topic_words) > 0:
                    max_val = max(top_topic_words, key=lambda x: x[1])
                else:
                    break
            phrase_keyword = max_val[0]
        if phrase_keyword:
            phrase_keywords.append(phrase_keyword)
    
    topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
    
    return topics, phrase_keywords

# Define the main function for the app
def main():
    st.title("Comment Topic Extraction App")
    
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if file is not None:
        data = pd.read_csv(file)
        comments = data.iloc[:, 0].tolist()
        if st.checkbox("Show comments"):
            st.write(comments)
        
        column = st.selectbox("Select the column that contains the comments", data.columns)
        comments = data[column].tolist()
        
        if st.button("Extract Topics"):
            st.write("Extracting topics...")
            topics, phrase_keywords = extract_topics(comments)
            tagged_comments = tag_comments(comments, topics)
            topics_df = pd.DataFrame({"Topic": range(len(topics)), "Top Words": [", ".join([word[0] for word in topic[1]]) for topic in topics], "Phrase Keyword": phrase_keywords})
            topics_df = topics_df[topics_df["Phrase Keyword"].apply(lambda x: x not in stopwords.words('english'))]
            topics_df = topics_df[topics_df["Phrase Keyword"].apply(lambda x: not all(char in string.punctuation for char in x))]
            topics_df = topics_df[topics_df["Phrase Keyword"].apply(lambda x: not x.isnumeric() and len(x) > 1)]
            topics_df = topics_df.sort_values("Topic", ascending=True)
            st.write(topics_df)
            if st.checkbox("Show tagged comments"):
                for i, comment in enumerate(tagged_comments):
                    st.write(f"{i+1}. Comment: {comment[0]}")
                    st.write(f"   Topic: {comment[1]}")
    else:
        st.write("Please upload a CSV file.")

if __name__ == "__main__":
    main()

