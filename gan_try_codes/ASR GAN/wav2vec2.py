# Import necessary library

# For managing audio file
import librosa

#Importing Pytorch
import torch

#Importing Wav2Vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor

import numpy as np
from scipy.io.wavfile import write

# Parameters for the noise
duration = 30  # in seconds
sampling_rate = 16000  # in Hz (now set to 16kHz)
amplitude = 0.5  # scale if you want it quieter

# Generate random noise
noise = np.random.normal(0, amplitude, int(sampling_rate * duration))

# Convert noise to 16-bit PCM format for WAV
noise_int16 = np.int16(noise * 32767)

# Write noise to WAV file
write("noise.wav", sampling_rate, noise_int16)

# Loading the audio file

audio, rate = librosa.load("noise.wav", sr = 16000)

# Importing Wav2Vec pretrained model

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Taking an input value

input_values = tokenizer(audio, return_tensors = "pt").input_values

# Storing logits (non-normalized prediction values)
logits = model(input_values).logits

# Storing predicted ids
prediction = torch.argmax(logits, dim = -1)

# Passing the prediction to the tokenzer decode to get the transcription
transcription = tokenizer.batch_decode(prediction)[0]

# Printing the transcription
print(transcription)


# using audio that is not noise

audio1, rate1 = librosa.load("taken_clip.wav", sr = 16000)

input_values1 = tokenizer(audio1, return_tensors = "pt").input_values

logits1 = model(input_values1).logits

prediction1 = torch.argmax(logits1, dim = -1)

transcription1 = tokenizer.batch_decode(prediction1)[0]
print(transcription1)




