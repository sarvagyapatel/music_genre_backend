from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load pre-trained model
model = tf.keras.models.load_model("./MusicGenre_CNN_79.73.h5")

# Genre dictionary
genre_dict = {
    0: "disco", 1: "pop", 2: "classical", 3: "metal", 4: "rock", 
    5: "blues", 6: "hiphop", 7: "reggae", 8: "country", 9: "jazz"
}

# Function to extract MFCC features
def extract_mfcc(audio_file, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    SAMPLE_RATE = 22050
    signal, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    duration = librosa.get_duration(y=signal, sr=sr)
    samples_per_segment = int((SAMPLE_RATE * duration) / num_segments)
    mfccs = []

    for s in range(num_segments):
        start = samples_per_segment * s
        finish = start + samples_per_segment
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        mfccs.append(mfcc)
    
    return np.array(mfccs)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the file temporarily
        filepath = os.path.join(os.getcwd(), file.filename)
        file.save(filepath)

        # Extract MFCC features
        mfccs = extract_mfcc(filepath)

        # Clean up the temporary file
        os.remove(filepath)

        # Average the segments along the first axis (num_segments)
        X_to_predict = np.mean(mfccs, axis=0)

        # Add the necessary dimensions
        X_to_predict = X_to_predict[..., np.newaxis]  # Add the channel dimension
        X_to_predict = np.expand_dims(X_to_predict, axis=0)  # Add the batch dimension

        # Predict the genre
        prediction = model.predict(X_to_predict)
        predicted_index = np.argmax(prediction, axis=1)[0]

        # Return the predicted genre
        return jsonify({'predicted_genre': genre_dict[predicted_index]})

if __name__ == '__main__':
    app.run(debug=True)
