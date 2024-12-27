import os
import base64
import uuid
import requests
from io import BytesIO
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from app.models import SpeakerEmbeddings
from django.conf import settings

import torch
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.spatial.distance import cdist
from scipy.io import wavfile


TRANSCRIPTION_API_URL = 'https://0121-34-142-235-206.ngrok-free.app/'

class Speaker(object):
    def __init__(self, name):
        self.name = name
        self.segment_number = 0
        self.segments = []
        self.most_mathing_recorded_speaker = None
        self.score = None

def home(request):
    return render(request, 'home.html')


@csrf_exempt
def transcribe_audio(request):
    if request.method == 'POST':
        audio_data = request.POST.get('audio')
        if not audio_data:
            return JsonResponse({'status': 'error', 'message': 'No audio data provided.'})

        # save audio
        save_func_response = save_audio(audio_data)
        if save_func_response.get('error'):
            return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})
        else:
            audio_filename = save_func_response.get('file_path')

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

            # get speaker segments
            speakers = get_speaker_segments(data)

            # concatenate speaker segments
            for speaker in speakers:
                combined_data = concatenate_speaker_segments(speaker, file_path)

                # save temporary concatenate file
                temp_save_func_response = save_audio(combined_data)
                if temp_save_func_response.get('error'):
                    return JsonResponse({'status': 'error', 'message': 'combined_data geçici olarak kaydedilirken bir hata oluştu'})
                else:
                    combined_audio_file_path = temp_save_func_response.get('file_path')
                    most_matching_speaker, score = speaker_is_recorded_check(combined_audio_file_path)

                speaker.most_mathing_recorded_speaker = most_matching_speaker
                speaker.score = score

            # Encode the histogram image to base64 to send to the frontend
            histogram_image = generate_audio_histogram(file_path)
            histogram_base64 = base64.b64encode(histogram_image).decode('utf-8')

            return JsonResponse(
                {'status': 'success', 'speaker_segments': None, 'histogram': None})
        else:
            return JsonResponse({'status': 'error', 'message': 'Transcription failed.'})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})


def get_speaker_segments(data):
    segments = data.get("segments")
    speakers = []
    speaker_names = []
    current_speaker = None

    for segment in segments:
        speaker_name = segment.get("speaker")
        start = segment.get("start")
        end = segment.get("end")
        if speaker_name not in speaker_names:
            # save speaker obj
            speaker = Speaker(speaker_name)
            speaker.segments.append({"start": start, "end": end})
            speaker.segment_number += 1
            speakers.append(speaker)

            # set local variables
            speaker_names.append(speaker_name)
            current_speaker = speaker_name

        elif current_speaker == speaker_name:
            for speaker in speakers:
                if speaker.name == speaker_name:
                    # change end value
                    speaker.segments[speaker.segment_number - 1]["end"] = end

        else:
            for speaker in speakers:
                if speaker.name == speaker_name:
                    # add new segment for speaker
                    speaker.segments.append({"start": start, "end": end})

                    # set local vari
                    current_speaker = speaker_name
    return speakers

def concatenate_speaker_segments(speaker, file_path):
    time_intervals = []
    for speaker_segment in speaker.segments:
        start_time = speaker_segment["start"]
        end_time = speaker_segment["end"]
        time_intervals.append((start_time, end_time))

    selected_wav_data = []
    sample_rate, wav_data = wavfile.read(file_path)

    for start_time, end_time in time_intervals:
        # Başlangıç ve bitiş örneklerini hesapla
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Belirtilen zaman aralığındaki verileri al
        selected_wav_data.append(wav_data[start_sample:end_sample])

    combined_data = np.concatenate(selected_wav_data)

    return combined_data

def speaker_is_recorded_check(file_path):
    distances = []

    # Get the voice embedding values
    inference = settings.INFERENCE
    inference.to(torch.device("cuda"))
    request_audio_embedding = inference(file_path)

    speakers = SpeakerEmbeddings.objects.all()
    for speaker in speakers:
        requested_audio = np.array(request_audio_embedding).reshape(1, -1)
        saved_audio = np.array(speaker.embedding).reshape(1, -1)
        distance = cdist(requested_audio, saved_audio, metric="cosine")[0, 0]
        distances.append({"speaker":speaker, "distance":distance})

    distances.sort(key=lambda x: x["distance"])
    return distances[0]["speaker"], distances[0]["distance"]


def save_audio(audio_data):
    try:
        audio_binary = base64.b64decode(audio_data.split(',')[1])
        file_path = f"recordings/{uuid.uuid4()}.wav"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(audio_binary)
        result = {'error': 'false', 'file_path': file_path}
    except Exception as e:
        result = {'error': 'true', 'message': str(e)}

    return result


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
def person_labeling(request):
    if request.method == 'POST':
        try:
            name = request.POST.get('speaker_name')
            names = SpeakerEmbeddings.objects.all().values_list('name', flat=True)
            if name in names:
                return JsonResponse({'status': 'error', 'message': 'Name already exists.'})

            audio_data = request.POST.get('audio')
            if not audio_data:
                return JsonResponse({'status': 'error', 'message': 'No audio data provided.'})

            # save audio
            save_func_response = save_audio(audio_data)
            if not save_func_response.get('error'):
                return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})
            else:
                audio_filename = save_func_response.get('file_path')

            # Open the saved audio file
            file_path = os.path.join(audio_filename)
            if not os.path.exists(file_path):
                return JsonResponse({'status': 'error', 'message': 'Audio file not found.'})

            # Get the voice embedding values
            inference = settings.INFERENCE
            inference.to(torch.device("cuda"))
            embeded_audio = inference(file_path)

            speaker_embedded_obj = SpeakerEmbeddings.objects.create(name=name, embedding=embeded_audio.tolist())
            speaker_embedded_obj.save()

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

        return JsonResponse({'status': 'success'})

    else:
        return render(request, 'labeling.html')