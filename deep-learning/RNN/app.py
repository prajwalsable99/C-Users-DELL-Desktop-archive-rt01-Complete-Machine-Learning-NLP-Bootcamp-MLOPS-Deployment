import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the IMDB dataset's word index
word_index = imdb.get_word_index()

# Model parameters (these should match your model's configuration)
max_len = 500

# Load the trained model (make sure the model file path is correct)
model = load_model('rnnmodel.h5')

# Function to preprocess and predict sentiment
def predict_sentiment(review, model):
    # Tokenize the input review using the predefined IMDB word index
    words = review.lower().split()  # Simple tokenization by space
    review_sequence = []
  
    for word in words:
        if word in word_index:  # Only include words that are in the word index

            if word_index.get(word)>=10000:
                review_sequence.append(2)
            else:    
                review_sequence.append(word_index.get(word))
        else:
            review_sequence.append(2)
              # Offset by 3 to match Keras' preprocessing
    
    # Pad the sequence to match the model's input length (max_len)
    review_padded = pad_sequences([review_sequence], maxlen=max_len)

    # Predict the sentiment
    prediction = model.predict(review_padded)
    
    # Convert the prediction to a human-readable form
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return sentiment, prediction[0][0]

# Streamlit interface
st.title("IMDB Movie Review Sentiment Analysis")

st.write("Enter a movie review below, and the model will predict whether the review is positive or negative.")

# Text area for user input
user_review = st.text_area("Enter your review here:", height=150)

# Button to trigger sentiment prediction
if st.button("Predict Sentiment"):
    if user_review.strip():
        # Predict sentiment
        sentiment, confidence = predict_sentiment(user_review, model)
        
        # Display results
        st.write(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {confidence:.4f}")
    else:
        st.write("Please enter a review before clicking the button.")


