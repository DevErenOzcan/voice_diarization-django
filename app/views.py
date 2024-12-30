import base64
import os
from io import BytesIO

import librosa
import requests
from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from app.models import Speaker, EmbeddedSpeakers, Segment, Word
from django.conf import settings

import torch
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.spatial.distance import cdist


TRANSCRIPTION_API_URL = 'https://2479-34-16-202-65.ngrok-free.app/'

def home(request):
    return render(request, 'home.html')


@csrf_exempt
def transcribe_audio(request):
    if request.method == 'POST':
        # get audio file
        audio_file = request.FILES.get('audio_file')
        file_save_success, file_save_message = save_audio_file(audio_file)
        if file_save_success:
            file_path = file_save_message

            with open(file_path, 'rb') as f:
                audio_file = f.read()

            # Send audio to transcription API
            response = requests.post(TRANSCRIPTION_API_URL, files={'audio': audio_file})

            if response.status_code == 200:
                data = response.json()
                Speaker.objects.all().delete()
                Segment.objects.all().delete()
                Word.objects.all().delete()

                # save speaker segments
                segment_save_succes = save_speaker_segments(data)
                if segment_save_succes == True:
                    parsed_audios = parse_audio_file(file_path)
                    # parsed_audio[0].get("segment").id
                    speakers = Speaker.objects.all()
                    # embedding check
                    for speaker in speakers:
                        # get speaker audios
                        speaker_audios = get_speaker_audios(speaker, parsed_audios)
                        # concatenate speaker segments
                        concatenated_speaker_audio = concatenate_speaker_audio(speaker, speaker_audios)
                        # speaker is recorded??
                        most_matching_speaker, score = speaker_is_recorded_check(concatenated_speaker_audio)
                        # add the embedding cheack values to Speaker object
                        speaker.most_matching_recorded_speaker = most_matching_speaker
                        speaker.score = score
                        speaker.save()

                    speaker_segments = []
                    for segment in Segment.objects.all():
                        sentiment_analyze_result = None
                        for parsed_audio in parsed_audios:
                            if segment == parsed_audio['segment']:
                                sentiment_analyze_result = voice_sentiment_analyze(parsed_audio['audio'])

                        speaker_segments.append({
                            'speaker': segment.speaker.most_matching_recorded_speaker.name,
                            'score': segment.speaker.score,
                            'text': segment.text,
                            'happy': sentiment_analyze_result["mutlu"],
                            'angry': sentiment_analyze_result["sinirli"],
                            'sad': sentiment_analyze_result["uzgun"],
                        })
                    # Encode the histogram image to base64 to send to the frontend
                    histogram_image = generate_audio_histogram(file_path)
                    histogram_base64 = base64.b64encode(histogram_image).decode('utf-8')
                    return JsonResponse(
                        {'status': 'success', 'speaker_segments': speaker_segments, 'histogram': histogram_base64})
                else:
                    return JsonResponse({'error': True, 'message': 'An error occurred while saving segments. Error: '})
            else:
                return JsonResponse({'error': True, 'message': "response not 200"})
        else:
            return JsonResponse({'error': True, 'message': file_save_message})
    else:
        return JsonResponse({'error': True, 'message': 'Invalid request'})


def save_audio_file(audio_file):
    file_name = f'recordings/transcribe/transcribe_'
    temp_file_path = default_storage.save(file_name, audio_file)
    # Dosya yolunu belirleyin
    input_file_path = default_storage.path(temp_file_path)
    output_file_path = os.path.splitext(input_file_path)[0] + ".wav"
    try:
        # Pydub ile dosyayı wav formatına dönüştürün
        audio = AudioSegment.from_file(input_file_path)
        audio.export(output_file_path, format="wav")
        default_storage.delete(temp_file_path)
        return True, output_file_path
    except Exception as e:
        return False, e


def save_speaker_segments(segments):
    try:
        for segment_data in segments["segments"]:
            speaker, _ = Speaker.objects.get_or_create(name=segment_data.get("speaker"))

            # hata varsa kaldırabilirsiniz
            pipe = settings.PIPE
            word_sentiment_analyze = pipe(segment_data.get("text"))

            segment = Segment.objects.create(
                start=segment_data.get("start"),
                end=segment_data.get("end"),
                text=segment_data.get("text"),
                speaker=speaker,
                sentiment = word_sentiment_analyze[0].get("label"),
                sentiment_score = word_sentiment_analyze[0].get("score"),
            )

            for word_data in segment_data["words"]:
                Word.objects.create(
                    segment=segment,
                    word=word_data.get("word"),
                    start=word_data.get("start"),
                    end=word_data.get("end"),
                    score=word_data.get("score"),
                    speaker=speaker
                )
        return True
    except Exception as e:
        return e

def parse_audio_file(file_path):
    audio = AudioSegment.from_file(file_path, format="wav")
    parsed_audio = []
    segments = Segment.objects.all()
    for segment in segments:
        # pydub a ms cinsinden göndermek gerekiyor o yüzden * 1000
        start_time = segment.start * 1000
        end_time = segment.end * 1000
        segment_audio = audio[start_time:end_time]
        parsed_audio.append({
            "segment": segment,
            "audio": segment_audio,
        })
    return parsed_audio


def get_speaker_audios(speaker, parsed_audios):
    speaker_audios = []
    speaker_segments = Segment.objects.filter(speaker=speaker)
    for segment in speaker_segments:
        for parsed_audio in parsed_audios:
            if parsed_audio.get("segment") == segment:
                speaker_audios.append(parsed_audio)
    return speaker_audios

def concatenate_speaker_audio(speaker, speaker_audios):
    concatenated_speaker_audio = AudioSegment.empty()
    for speaker_audio in speaker_audios:
        audio_of_segment = speaker_audio.get("audio")
        concatenated_speaker_audio += audio_of_segment
    return concatenated_speaker_audio


def speaker_is_recorded_check(combined_data):
    distances = []

    # Get the voice embedding values
    inference = settings.INFERENCE
    #inference.to(torch.device("cuda"))

    # embedding yapılacak olan ses pytorch tensor formatında olmalı bu yüzden combined data değişkenini önce
    # numpy array formatına sonra tensor formatına dönüştürüyoruz
    waveform = np.array(combined_data.get_array_of_samples())
    waveform = waveform.reshape((-1, combined_data.channels))
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
    waveform_tensor = waveform_tensor.permute(1, 0)  # (channels, time)

    # !embedding!
    embedding = inference({'waveform': waveform_tensor, 'sample_rate': combined_data.frame_rate})
    embedding = embedding.reshape(1, -1)

    speakers = EmbeddedSpeakers.objects.all()
    for speaker in speakers:
        saved_embedding = np.array(speaker.embedding).reshape(1, -1)
        distance = cdist(embedding, saved_embedding, metric="cosine")[0, 0]
        distances.append({"speaker": speaker, "distance": distance})

    distances.sort(key=lambda x: x["distance"])
    return distances[0]["speaker"], distances[0]["distance"]


def voice_sentiment_analyze(segment_of_audio):
    # Load the pre-trained model
    model = settings.EMOTION_RECOGNITION

    # Get labels from the model
    labels = model.classes_

    # Convert audio segment to a NumPy array and normalize to floating-point
    waveform = np.array(segment_of_audio.get_array_of_samples(), dtype=np.float32)
    waveform /= np.iinfo(segment_of_audio.sample_width * 8).max  # Normalize to [-1.0, 1.0]

    # Reshape for multi-channel support
    waveform = waveform.reshape((-1, segment_of_audio.channels))

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=waveform[:, 0], sr=segment_of_audio.frame_rate, n_mfcc=40)  # Use one channel
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Predict probabilities
    probabilities = model.predict_proba([mfccs_mean])

    # Prepare results
    results = dict(zip(labels, probabilities[0]))
    return results


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
            # get speaker name
            name = request.POST.get('speaker_name')
            names = EmbeddedSpeakers.objects.all().values_list('name', flat=True)
            if name in names:
                return JsonResponse({'error': True, 'message': 'Name already exists.'})

            # get audio file
            audio_file = request.FILES.get('audio_file')
            if audio_file.size == 0:
                return JsonResponse({'error': True, 'message': 'No audio data provided.'})

            # save audio
            if audio_file and name:
                file_name = f'recordings/labeled/{name}_{audio_file.name}'
                file_path = default_storage.save(file_name, audio_file)

                # get embedding values
                inference = settings.INFERENCE
                #inference.to(torch.device("cuda"))
                embeded_audio = inference(file_path)

                # save mebedding values
                speaker_embedded_obj = EmbeddedSpeakers.objects.create(name=name, embedding=embeded_audio.tolist())
                speaker_embedded_obj.save()
            return JsonResponse({'error': False, 'message': 'Success'})
        except Exception as e:
            return JsonResponse({'error': True, 'message': str(e)})
    else:
        return render(request, 'labeling.html')