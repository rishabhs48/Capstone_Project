import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Set page title and icon
st.set_page_config(
    page_title="Review Classifier",
    page_icon=":compression:"
)

# Page title and description
st.title("Review Classifier")
st.write("Enter a review and click the 'Classify' button to analyze its sentiment.")

# Input text box
input_review = st.text_input("Enter the Review")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Function to transform text
def transform_text(text):
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = nltk.word_tokenize(text)
    stopwords_set = set(stopwords.words('english'))
    ps = PorterStemmer()
    transformed_text = [ps.stem(word) for word in text if word not in stopwords_set]
    return " ".join(transformed_text)

# Classify button
if st.button("Classify"):
    transformed_text = transform_text(input_review)

    # Check if transformed_text is not empty before proceeding with prediction
    if transformed_text:
        # Vectorize the input text
        input_vectorized = vectorizer.transform([transformed_text])

        # Load the model
        model = pickle.load(open('model_pip_VG.pkl', 'rb'))

        # Predict using the model
        pipeline_result = model.predict(input_vectorized)[0]

        # Display the result
        if pipeline_result == 'Positive':
            st.success("Review is Positive 😃")
        elif pipeline_result == 'Negative':
            st.error("Review is Negative 😞")
        else:
            st.info("Review is Neutral 😐")
    else:
        st.warning("Please enter a review to classify.")
# About section
st.markdown("### About")
st.write(
    "This is a simple review classifier app built using Streamlit and NLTK. It predicts whether a review is positive, negative, or neutral based on its text content.")

# GitHub link
st.markdown("### Source Code")
st.write(
    "You can find the source code for this app on [GitHub](https://github.com/yourusername/review-classifier-app).")

# Footer
st.markdown("---")
st.write("Made with 🧠 by our team.")