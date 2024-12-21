import os
import base64
import uuid
import json
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from io import BytesIO
import requests

# For the transcription API
TRANSCRIPTION_API_URL = 'http://127.0.0.1:5000'  # Replace with actual API URL


def home(request):
    return render(request, 'home.html')


@csrf_exempt
def save_audio(request):
    if request.method == 'POST':
        audio_data = request.POST.get('audio')
        if not audio_data:
            return JsonResponse({'status': 'error', 'message': 'No audio data provided.'})

        audio_binary = base64.b64decode(audio_data.split(',')[1])
        filename = f"recordings/{uuid.uuid4()}.wav"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            f.write(audio_binary)

        return JsonResponse({'status': 'success', 'message': 'Audio saved successfully.', 'filename': filename})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})


def generate_audio_histogram(file_path):
    # Load the audio file using pydub
    audio = AudioSegment.from_wav(file_path)

    # Convert audio to numpy array for FFT
    samples = np.array(audio.get_array_of_samples())

    # Perform FFT and get frequencies
    freqs = np.fft.fftfreq(len(samples), 1.0 / audio.frame_rate)
    fft_vals = np.abs(np.fft.fft(samples))

    # Plot the histogram (frequency spectrum)
    plt.figure(figsize=(10, 6))
    plt.plot(freqs[:len(freqs) // 2], fft_vals[:len(fft_vals) // 2])  # Only positive frequencies
    plt.title("Audio Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    # Save the histogram as an image in memory
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    return img_buf.read()


@csrf_exempt
def transcribe_audio(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        audio_filename = data.get('audio_filename')

        if not audio_filename:
            return JsonResponse({'status': 'error', 'message': 'No audio filename provided.'})

        # Open the saved audio file
        file_path = os.path.join(audio_filename)
        if not os.path.exists(file_path):
            return JsonResponse({'status': 'error', 'message': 'Audio file not found.'})

        with open(file_path, 'rb') as f:
            audio_file = f.read()

        # Send audio to transcription API
        response = requests.post(TRANSCRIPTION_API_URL, files={'audio': audio_file})

        if response.status_code == 200:
            data = response.json()

            # Group segments by speaker
            speaker_segments = []
            for segment in data.get("segments", []):
                speaker = segment.get("speaker", "Unknown")
                text = segment.get("text", "")
                speaker_segments.append({
                    'speaker': speaker,
                    'text': text
                })

            # Encode the histogram image to base64 to send to the frontend
            histogram_image = generate_audio_histogram(file_path)
            histogram_base64 = base64.b64encode(histogram_image).decode('utf-8')

            return JsonResponse(
                {'status': 'success', 'speaker_segments': speaker_segments, 'histogram': histogram_base64})
        else:
            return JsonResponse({'status': 'error', 'message': 'Transcription failed.'})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})
