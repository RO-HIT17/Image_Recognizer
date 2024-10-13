#!/usr/bin/env python
# coding: utf-8

# In[49]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
import numpy as np
import os
from PIL import Image


# In[50]:


def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize image to a fixed size
    img = np.array(img) / 255.0   # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

image_path = "C:\\Rohit\\Projects\\Image Describer\\sample_data\\36979.jpg"
image = load_image(image_path)
print("Image Shape :",image.shape) 
if len(image.shape) == 4 and image.shape[0] == 1:
                    image = np.squeeze(image, axis=0) 
print("Image Shape :",image.shape) 


# In[51]:


captions = ["Several men play cards while around a green table ."]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

caption_sequences = tokenizer.texts_to_sequences(captions)
max_caption_length = max([len(caption) for caption in caption_sequences])

caption_sequences = pad_sequences(caption_sequences, maxlen=max_caption_length, padding='post')

print(f"Max Caption Legth: {max_caption_length}")


# In[52]:


def create_model(vocab_size, max_caption_length):
    # Input for the image
    image_input = Input(shape=(224, 224, 3))

    # Image feature extraction
    cnn = tf.keras.applications.VGG16(include_top=False, input_tensor=image_input)
    cnn.trainable = False
    features = cnn.output
    features = tf.keras.layers.Flatten()(features)
    features = Dense(256, activation='relu')(features)
    
    # Input for the caption
    caption_input = Input(shape=(max_caption_length,))
    x = Embedding(vocab_size, 256)(caption_input)
    x = LSTM(256, return_sequences=False)(x)

    # Combine image and caption
    combined = tf.keras.layers.add([features, x])
    combined = Dense(256, activation='relu')(combined)
    output = Dense(vocab_size, activation='softmax')(combined)

    # Define model
    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Create the model
model = create_model(vocab_size, max_caption_length)
model.summary()


# In[53]:


image_data = np.array([image])  # Example image batch
caption_data = np.array(caption_sequences)  # Example caption batch

# Placeholder labels (one-hot encoded)
labels = np.zeros((caption_data.shape[0], vocab_size))  # Modify this as per your dataset

# Train the model
model.fit([image_data, caption_data], labels, epochs=10, batch_size=32)


# In[54]:


# Save the model
model.save('model/image_captioning_model1.h5')

# Save the tokenizer
import pickle
with open('model/tokenizer1.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

