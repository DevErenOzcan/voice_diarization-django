import base64
import os
from io import BytesIO

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


TRANSCRIPTION_API_URL = 'https://fbb5-34-143-172-38.ngrok-free.app/'

def home(request):
    return render(request, 'home.html')


@csrf_exempt
def transcribe_audio(request):
    if request.method == 'POST':
        # get audio file
        audio_file = request.FILES.get('audio_file')
        save_audio_file_resp = save_audio_file(audio_file)
        if not save_audio_file_resp.get("error"):
            if not False:
                file_path = save_audio_file_resp.get("message")

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
                    save_speaker_segment_resp = save_speaker_segments(data.get('segments'))
                    if not save_speaker_segment_resp["error"]:
                        speakers = Speaker.objects.all()
                        # embedding check
                        for speaker in speakers:
                            # concatenate speaker segments
                            concatenated_segments = concatenate_speaker_segments(speaker, file_path)
                            # speaker is recorded??
                            most_matching_speaker, score = speaker_is_recorded_check(concatenated_segments)
                            # add the embedding cheack values to Speaker object
                            speaker.most_matching_recorded_speaker = most_matching_speaker
                            speaker.score = score
                            speaker.save()

                        speaker_segments = []
                        for segment in Segment.objects.all():
                            speaker_segments.append({
                                'speaker': segment.speaker.most_matching_recorded_speaker.name,
                                'text': segment.text
                            })
                        # Encode the histogram image to base64 to send to the frontend
                        histogram_image = generate_audio_histogram(file_path)
                        histogram_base64 = base64.b64encode(histogram_image).decode('utf-8')
                        return JsonResponse(
                            {'status': 'success', 'speaker_segments': speaker_segments, 'histogram': histogram_base64})
                    else:
                        return JsonResponse({'error': True, 'message': f'Failed to save segments. Error: {save_speaker_segment_resp["message"]}'})
                else:
                    return JsonResponse({'error': True, 'message': "response not 200"})
            else:
                return JsonResponse({'error': True, 'message': f'An error occurred while saving the audio file.Error:{save_audio_file_resp.get("message")}'})
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
        return {'error': False, 'message': output_file_path}
    except Exception as e:
        return {'error': True, 'message': str(e)}


def save_speaker_segments(segments):
    try:
        for segment_data in segments:
            speaker, _ = Speaker.objects.get_or_create(name=segment_data.get("speaker"))

            # hata varsa kaldırabilirsiniz
            pipe = settings.PIPE
            sentiment_analyze = pipe(segment_data.get("text"))

            segment = Segment.objects.create(
                start=segment_data.get("start"),
                end=segment_data.get("end"),
                text=segment_data.get("text"),
                speaker=speaker,
                sentiment = sentiment_analyze[0].get("label"),
                sentiment_score = sentiment_analyze[0].get("score"),
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
        return {'error': False, 'message': "Segments retrieved successfully."}
    except Exception as e:
        return {'error': True, 'message': str(e)}



def concatenate_speaker_segments(speaker, file_path):
    audio = AudioSegment.from_file(file_path, format="wav")
    concatenated_audio = AudioSegment.empty()

    segments = Segment.objects.filter(speaker=speaker)
    for segment in segments:
        # pydub a ms cinsinden göndermek gerekiyor o yüzden * 1000
        start_time = segment.start * 1000
        end_time = segment.end * 1000
        segment_audio = audio[start_time:end_time]
        concatenated_audio += segment_audio
    return concatenated_audio




def speaker_is_recorded_check(combined_data):
    distances = []
    sample_rate = combined_data.frame_rate

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
    embedding = inference({'waveform': waveform_tensor, 'sample_rate': sample_rate})
    embedding = embedding.reshape(1, -1)

    speakers = EmbeddedSpeakers.objects.all()
    for speaker in speakers:
        saved_embedding = np.array(speaker.embedding).reshape(1, -1)
        distance = cdist(embedding, saved_embedding, metric="cosine")[0, 0]
        distances.append({"speaker": speaker, "distance": distance})

    distances.sort(key=lambda x: x["distance"])
    return distances[0]["speaker"], distances[0]["distance"]


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

        except Exception as e:
            return JsonResponse({'error': True, 'message': str(e)})

        return JsonResponse({'error': False, 'message':'Success'})

    else:
        return render(request, 'labeling.html')