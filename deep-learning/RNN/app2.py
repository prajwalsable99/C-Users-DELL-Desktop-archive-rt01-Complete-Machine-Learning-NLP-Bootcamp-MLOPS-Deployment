import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
import numpy as np

model=load_model('hamlet-model.h5')

tokenizer=joblib.load('hamlet-tokenizer.pkl')

# Step 6: Function to generate next word
def generate_text(model,tokenizer,text,max_seq_len):
    
   
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    token_list = pad_sequences([token_list], maxlen= max_seq_len-1 , truncating='pre')
    # print(token_list)
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)
        
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "not working at moment"
            

# Streamlit interface
st.title("Next word generator using LSTM")


# Text area for user input
inp_text = st.text_area("Enter here:", height=150)

# Button to trigger sentiment prediction
if st.button("Predict next word"):
    
    st.markdown('---------')

    op=generate_text(model,tokenizer,inp_text,13)
    st.markdown(op)
    
        

