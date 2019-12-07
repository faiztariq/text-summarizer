"""
Author : Faiz Tariq
Date : 11/16/2019
Desc : Abstractive Text Summarization using NLP
"""

import numpy as np
import pandas as pd
import re
import sys
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from text_summarization.attention_layer.attention import AttentionLayer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K
from tensorflow.keras.models import load_model
import warnings
import nltk
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

# Global Config

max_text_len = 30
max_summary_len = 8
latent_dim = 300
embedding_dim = 100

# Short words Dictionary

contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                       "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                       "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                       "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                       "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                       "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                       "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                       "you're": "you are", "you've": "you have"}

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Removing short words

def text_cleaner(text, num):

    """This is a method for cleaning the text from short words"""

    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"', '', newString)
    newString = ' '.join(
        [contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r"'s\b", "", newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num == 0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens = newString.split()
    long_words = []
    for i in tokens:
        if len(i) > 1:  # removing short word
            long_words.append(i)
    return (" ".join(long_words)).strip()


def decode_sequence(encoder_model, decoder_model, target_word_index, reverse_target_word_index, input_seq):

    """This is a method for decoding the provided input from sequence to Text"""

    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict(
            [target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token != 'eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def predict_summary(article):

    """This is a method for preparing the dictionary and using the trained model for prediction"""

    data = pd.read_csv(".\\text_summarization\data\Reviews.csv", nrows=60000)
    data.drop_duplicates(subset=['Text'], inplace=True)  # dropping duplicates

    data.dropna(axis=0, inplace=True)  # dropping na

    data.info()  
    

    # call the function
    cleaned_text = []
    for t in data['Text']:
        cleaned_text.append(text_cleaner(t, 0))

    cleaned_text[:5]

    # call the function
    cleaned_summary = []
    for t in data['Summary']:
        cleaned_summary.append(text_cleaner(t, 1))

    cleaned_summary[:10]

    data['cleaned_text'] = cleaned_text
    data['cleaned_summary'] = cleaned_summary

    data.replace('', np.nan, inplace=True)
    data.dropna(axis=0, inplace=True)

    text_word_count = []
    summary_word_count = []

    # populate the lists with sentence lengths
    for i in data['cleaned_text']:
        text_word_count.append(len(i.split()))

    for i in data['cleaned_summary']:
        summary_word_count.append(len(i.split()))

    length_df = pd.DataFrame(
        {'text': text_word_count, 'summary': summary_word_count})

    length_df.hist(bins=30)
    # plt.show()

    cnt = 0
    for i in data['cleaned_summary']:
        if(len(i.split()) <= 8):
            cnt = cnt+1

    cleaned_text = np.array(data['cleaned_text'])
    cleaned_summary = np.array(data['cleaned_summary'])

    short_text = []
    short_summary = []

    for i in range(len(cleaned_text)):
        if(len(cleaned_summary[i].split()) <= max_summary_len and len(cleaned_text[i].split()) <= max_text_len):
            short_text.append(cleaned_text[i])
            short_summary.append(cleaned_summary[i])

    df = pd.DataFrame({'text': short_text, 'summary': short_summary})

    df['summary'] = df['summary'].apply(lambda x: 'sostok ' + x + ' eostok')

    x_tr, x_val, y_tr, y_val = train_test_split(np.array(df['text']), np.array(
        df['summary']), test_size=0.1, random_state=0, shuffle=True)

    # prepare a tokenizer for reviews on training data
    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(list(x_tr))

    thresh = 4

    cnt = 0
    tot_cnt = 0
    freq = 0
    tot_freq = 0

    for key, value in x_tokenizer.word_counts.items():
        tot_cnt = tot_cnt+1
        tot_freq = tot_freq+value
        if(value < thresh):
            cnt = cnt+1
            freq = freq+value

    # prepare a tokenizer for reviews on training data
    x_tokenizer = Tokenizer(num_words=tot_cnt-cnt)
    x_tokenizer.fit_on_texts(list(x_tr))

    # convert text sequences into integer sequences
    x_tr_seq = x_tokenizer.texts_to_sequences(x_tr)
    x_val_seq = x_tokenizer.texts_to_sequences(x_val)

    # padding zero upto maximum length
    x_tr = pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
    x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

    # size of vocabulary ( +1 for padding token)
    x_voc = x_tokenizer.num_words + 1

    x_voc

    # prepare a tokenizer for reviews on training data
    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(list(y_tr))

    thresh = 6

    cnt = 0
    tot_cnt = 0
    freq = 0
    tot_freq = 0

    for key, value in y_tokenizer.word_counts.items():
        tot_cnt = tot_cnt+1
        tot_freq = tot_freq+value
        if(value < thresh):
            cnt = cnt+1
            freq = freq+value

    # prepare a tokenizer for reviews on training data
    y_tokenizer = Tokenizer(num_words=tot_cnt-cnt)
    y_tokenizer.fit_on_texts(list(y_tr))

    # convert text sequences into integer sequences
    y_tr_seq = y_tokenizer.texts_to_sequences(y_tr)
    y_val_seq = y_tokenizer.texts_to_sequences(y_val)

    # padding zero upto maximum length
    y_tr = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
    y_val = pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

    # size of vocabulary
    y_voc = y_tokenizer.num_words + 1

    y_tokenizer.word_counts['sostok'], len(y_tr)

    ind = []
    for i in range(len(y_tr)):
        cnt = 0
        for j in y_tr[i]:
            if j != 0:
                cnt = cnt+1
        if(cnt == 2):
            ind.append(i)

    y_tr = np.delete(y_tr, ind, axis=0)
    x_tr = np.delete(x_tr, ind, axis=0)

    ind = []
    for i in range(len(y_val)):
        cnt = 0
        for j in y_val[i]:
            if j != 0:
                cnt = cnt+1
        if(cnt == 2):
            ind.append(i)

    y_val = np.delete(y_val, ind, axis=0)
    x_val = np.delete(x_val, ind, axis=0)

    K.clear_session()
    
    #loading model
    loaded_model = load_model('.\\text_summarization\model\\text_generator.h5', custom_objects={
                          'AttentionLayer': AttentionLayer})

    # Encoder
    encoder_inputs = loaded_model.input[0]

    # embedding layer
    enc_emb = loaded_model.layers[1].output

    # encoder lstm 1
    encoder_output1, state_h1, state_c1 = loaded_model.layers[2].output

    # encoder lstm 2
    encoder_output2, state_h2, state_c2 = loaded_model.layers[4].output

    # encoder lstm 3
    encoder_outputs, state_h, state_c = loaded_model.layers[6].output

    encoder_model = Model(inputs=encoder_inputs, outputs=[
                      encoder_outputs, state_h, state_c])

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = loaded_model.input[1]

    # embedding layer
    dec_emb = loaded_model.layers[5].output

    decoder_lstm = loaded_model.layers[7]
    decoder_outputs, decoder_fwd_state, decoder_back_state = loaded_model.layers[7].output

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = loaded_model.layers[8].output

    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(
        axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    # dense layer
    decoder_dense = loaded_model.layers[10]
    decoder_outputs = loaded_model.layers[10].output

    decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
    decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
    decoder_hidden_state_input = Input(
        shape=(max_text_len, latent_dim), name='input_5')

    # Get the embeddings of the decoder sequence
    dec_emb2 = loaded_model.layers[5].output
    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(
        dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    # attention inference
    attn_out_inf, attn_states_inf = attn_layer(
        [decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(
        axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = decoder_dense(decoder_inf_concat)

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input,
                        decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])

    # model.summary()

    reverse_target_word_index = y_tokenizer.index_word
    reverse_source_word_index = x_tokenizer.index_word
    target_word_index = y_tokenizer.word_index

    original_text_article = article
    article_to_seq_raw = x_tokenizer.texts_to_sequences([original_text_article])
    article_to_seq = pad_sequences(
    article_to_seq_raw,  maxlen=max_text_len, padding='post')
    print(decode_sequence.__doc__)
    return decode_sequence(encoder_model, decoder_model, target_word_index, reverse_target_word_index, article_to_seq[0].reshape(1, max_text_len))
