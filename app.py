import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle

with open("model.pkl", "rb") as file:
    clf = pickle.load(file)
    
tokenizer = pickle.load(open('tokenizer.pkl','rb'))
tokenizer = Tokenizer(num_words=43666)

def main():

    st.title("QMatch - LSTM Question Pair Similarity")

    q1_test = st.text_input("Enter Question 1:")
    q2_test = st.text_input("Enter Question 2:")

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        return text

    q1_test = clean_text(q1_test)
    q2_test = clean_text(q2_test)

    q1_seq = tokenizer.texts_to_sequences([q1_test])
    q2_seq = tokenizer.texts_to_sequences([q2_test])

    q1_pad = pad_sequences(q1_seq, maxlen=284)
    q2_pad = pad_sequences(q2_seq, maxlen=284)

    pred = clf.predict([q1_pad, q2_pad])[0][0]

    if st.button("Check Similarity"):
        st.write(f"Similarity Score: {pred:.2f}")
        st.write("Duplicate" if pred > 0.5 else "Not Duplicate")
        
        # print("Similarity Score:", pred)
        # print("Duplicate" if pred > 0.5 else "Not Duplicate")

if __name__ == "__main__":
    main()