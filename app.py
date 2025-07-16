import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pickle
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import io # For handling image bytes

# --- 1. Define your Model Classes (MUST be present for tf.keras.models.load_model) ---
# These classes must be identical to how they were defined during training.
# Even though we are using TFSMLayer, having these definitions helps TensorFlow
# understand the custom layers if they were part of the SavedModel's graph.
# However, for pure TFSMLayer usage, these class definitions are technically not
# strictly required at load time IF the SavedModel's serving signature is self-contained.
# But it's good practice to keep them for clarity and potential debugging.

class Attention_model(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape: (batch_size, 64, embedding_dim)
        # hidden shape: (batch_size, hidden_size)

        # expand hidden state to shape: (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # Score shape: (batch_size, 64, units)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # Attention_weights shape: (batch_size, 64, 1)
        # We get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # Context_vector shape: (batch_size, embedding_dim)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Encoder(tf.keras.Model):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        # This Dense layer projects the features extracted by InceptionV3
        self.dense = tf.keras.layers.Dense(embed_dim)

    def call(self, features):
        # features here are assumed to be the output from InceptionV3
        # after global average pooling or similar, then reshaped to (batch, 64, features_dim)
        features = self.dense(features)
        features = tf.keras.activations.relu(features)
        return features

class Decoder(tf.keras.Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.attention = Attention_model(self.units) # Initialize Attention model
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim) # Embedding layer
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units) # First Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size) # Output Dense layer for vocabulary

    def call(self, x, features, hidden):
        # x: input sequence (batch_size, 1) - current word token
        # features: encoder output (batch_size, 64, embed_dim)
        # hidden: previous hidden state of GRU (batch_size, units)

        context_vector, attention_weights = self.attention(features, hidden)
        embed = self.embed(x) # Embed input word. Shape: (batch_size, 1, embedding_dim)

        # Concatenate embedded input with context vector.
        # context_vector needs to be expanded to (batch_size, 1, embed_dim) for concat
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1)

        output, state = self.gru(embed) # GRU output and new hidden state
        # Output shape : (batch_size, 1, units) because return_sequences=True
        # State shape : (batch_size, units)

        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2])) # Reshape for final dense layer
        output = self.d2(output) # Final prediction over vocabulary

        return output, state, attention_weights

    def init_state(self, batch_size): # This method is not directly callable on TFSMLayer
        return tf.zeros((batch_size, self.units))

# --- End of Model Class Definitions ---


# --- 2. Configuration for Saved Models ---
# Define the directories where your models and tokenizer are saved
# Assuming models were saved directly into these directories:
# tf.keras.models.save_model(encoder, './encoder')
# tf.keras.models.save_model(decoder, './decoder')
# pickle.dump(tokenizer, ...) into './tokenizer/tokenizer.pkl'
SAVE_DIRe = './encoder' # Path to encoder SavedModel root directory
SAVE_DIRd = './decoder' # Path to decoder SavedModel root directory
SAVE_DIRt = './tokenizer' # Path to tokenizer.pkl file directory

# --- 3. Load Models and Tokenizer (Cached for Performance) ---
@st.cache_resource
def load_all_assets():
    try:
        # Load Encoder using TFSMLayer for Keras 3 compatibility
        # The path should point directly to the SavedModel root directory (e.g., './encoder')
        encoder = tf.keras.layers.TFSMLayer(SAVE_DIRe, call_endpoint='serving_default')
        st.success("Encoder loaded successfully!")

        # Load Decoder using TFSMLayer for Keras 3 compatibility
        # The path should point directly to the SavedModel root directory (e.g., './decoder')
        decoder = tf.keras.layers.TFSMLayer(SAVE_DIRd, call_endpoint='serving_default')
        st.success("Decoder loaded successfully!")

        # Load Tokenizer
        tokenizer_path = os.path.join(SAVE_DIRt, 'tokenizer.pkl')
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        st.success("Tokenizer loaded successfully!")

        return encoder, decoder, tokenizer
    except Exception as e:
        st.error(f"Error loading models or tokenizer: {e}")
        st.info(f"Please ensure '{SAVE_DIRe}', '{SAVE_DIRd}', and '{SAVE_DIRt}' exist and contain the necessary files. "
                f"Also, verify that the SavedModel contents (e.g., 'saved_model.pb') are directly in '{SAVE_DIRe}' and '{SAVE_DIRd}'.")
        st.stop() # Stop the app if models can't be loaded

encoder, decoder, tokenizer = load_all_assets()

# --- 4. Image Preprocessing and Caption Generation Function ---

# Load InceptionV3 for feature extraction (pre-trained on ImageNet)
# This will be the first part of your encoder pipeline
@st.cache_resource
def load_inception_model():
    inception_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = inception_model.input
    hidden_layer = inception_model.layers[-1].output # Get output of the last conv block
    # Reshape the output to (batch_size, 64, 2048) if it's (batch_size, 8, 8, 2048)
    # This matches the expected input shape for your custom Encoder's Dense layer
    feature_extractor = tf.keras.Model(new_input, hidden_layer)
    return feature_extractor

image_feature_extractor = load_inception_model()


def preprocess_image(image_bytes):
    """Decodes, resizes, and preprocesses an image for InceptionV3."""
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, (299, 299)) # InceptionV3 input size
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def generate_caption(image_bytes, encoder_model, decoder_model, tokenizer_obj, feature_extractor_model, max_len=50):
    """Generates a caption for a given image."""
    # Preprocess the image and get features
    processed_img = preprocess_image(image_bytes)
    img_tensor = tf.expand_dims(processed_img, 0) # Add batch dimension

    # Extract features using InceptionV3
    features = feature_extractor_model(img_tensor)
    # Reshape features to (batch_size, 64, features_dim)
    # Assuming InceptionV3 output is (1, 8, 8, 2048)
    features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))

    # Pass features through your custom Encoder's Dense layer
    # The error message indicates encoder_model returns a dictionary with 'output_1' as the key for features
    encoded_features_output = encoder_model(features)
    encoded_features = encoded_features_output['output_1'] # Extract the tensor from the dictionary


    # --- IMPORTANT: Set this to the actual `units` value used in your Decoder's GRU layer during training ---
    # Common values are 512, 1024, etc.
    DECODER_UNITS = 512 # <--- YOU MUST REPLACE THIS WITH YOUR ACTUAL DECODER UNITS
    hidden = tf.zeros((1, DECODER_UNITS)) # batch_size=1 for single image inference

    dec_input = tf.expand_dims([tokenizer_obj.word_index['<start>']], 0) # Start with <start> token

    result = []
    for i in range(max_len):
        # Call the decoder model by passing a dictionary to the 'inputs' argument.
        # The keys 'args_0', 'args_1', 'args_2' are derived from the SavedModel's signature
        # and correspond to x, features, hidden respectively.
        decoder_outputs = decoder_model(inputs={'args_0': dec_input, 'args_1': encoded_features, 'args_2': hidden})
        
        # Accessing outputs from the dictionary returned by TFSMLayer
        # Based on the error message's output signature:
        # -> Dict[['output_2', TensorSpec(shape=(None, 512), dtype=tf.float32, name='output_2')],
        #         ['output_1', TensorSpec(shape=(None, 5001), dtype=tf.float32, name='output_1')],
        #         ['output_3', TensorSpec(shape=(None, 64, 1), dtype=tf.float32, name='output_3')]]
        # So, predictions is 'output_1', hidden is 'output_2'
        predictions = decoder_outputs['output_1'] # Corresponds to 'output' from your Decoder's call
        hidden = decoder_outputs['output_2']     # Corresponds to 'state' from your Decoder's call
        # attention_weights = decoder_outputs['output_3'] # Corresponds to 'attention_weights', not needed for inference loop

        predicted_id = tf.argmax(predictions[0]).numpy()
        
        # Handle cases where predicted_id might not be in index_word (rare, but good practice)
        if predicted_id not in tokenizer_obj.index_word:
            predicted_word = '<unk>' # Unknown word token
        else:
            predicted_word = tokenizer_obj.index_word[predicted_id]

        result.append(predicted_word)

        if predicted_word == '<end>':
            break
        
        dec_input = tf.expand_dims([predicted_id], 0) # Feed predicted word back as input

    # Clean up the caption
    caption = ' '.join(result).replace('<start>', '').replace('<end>', '').strip()
    return caption

# --- 5. BLEU Score Calculation Function ---
def calculate_bleu(reference_caption_str, predicted_caption_str, weights=(0.25, 0.25, 0.25, 0.25)):
    """Calculates BLEU score."""
    smooth_fn = SmoothingFunction().method1
    
    # Tokenize captions
    # BLEU expects a list of reference sentences, where each sentence is a list of tokens.
    # If you have multiple reference captions, pass them as [[ref1_tokens], [ref2_tokens]]
    # For a single reference, it's [[ref_tokens]]
    reference_tokens = [nltk.word_tokenize(reference_caption_str.lower())] # Ensure lowercasing
    candidate_tokens = nltk.word_tokenize(predicted_caption_str.lower()) # Ensure lowercasing

    score = sentence_bleu(reference_tokens, candidate_tokens, weights=weights, smoothing_function=smooth_fn)
    return score

# --- 6. Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Image Captioning App")

st.title("📸 AI Image Caption Generator")
st.markdown("Upload an image and let the model generate a descriptive caption!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        st.subheader("Generated Caption:")
        with st.spinner("Generating caption..."):
            predicted_caption = generate_caption(image_bytes, encoder, decoder, tokenizer, image_feature_extractor)
        st.success(f"**{predicted_caption}**")

        st.markdown("---")
        st.subheader("Evaluate Performance (Optional)")
        real_caption_input = st.text_input("Enter the actual (real) caption for this image:", "")

        if st.button("Calculate BLEU Score") and real_caption_input:
            bleu_score = calculate_bleu(real_caption_input, predicted_caption)
            st.info(f"**BLEU Score:** {bleu_score * 100:.2f}%")
            st.markdown(f"""
            - **Real Caption:** `{real_caption_input}`
            - **Predicted Caption:** `{predicted_caption}`
            """)
            if bleu_score * 100 >= 70:
                st.balloons()
            elif bleu_score * 100 >= 40:
                st.write("Good match! Keep up the good work.")
            else:
                st.write("The captions differ significantly. Consider retraining or fine-tuning your model.")
