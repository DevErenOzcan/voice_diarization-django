import base64
import os
from io import BytesIO

from groq import Groq
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


TRANSCRIPTION_API_URL = 'https://f83f-34-142-182-249.ngrok-free.app/'
GROK_API_URL = "https://api.grok.com/topic-analysis"
GROK_TOKEN = "<GROK_TOKEN>"

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
                    # topic analysis
                    texts = ""
                    segments = Segment.objects.all()
                    for segment in segments:
                        texts += segment.text + "\n"
                    topic_analysis = get_topic_analysis(texts)
                    parsed_save_success = parse_and_save_speaker_audios(file_path)
                    if parsed_save_success == True:
                        speakers = Speaker.objects.all()
                        # embedding check
                        for speaker in speakers:
                            # concatenate speaker segments
                            concatenated_speaker_audio = concatenate_speaker_audio(speaker)
                            # speaker is recorded??
                            most_matching_speaker, score = speaker_is_recorded_check(concatenated_speaker_audio)
                            # add the embedding cheack values to Speaker object
                            speaker.most_matching_recorded_speaker = most_matching_speaker
                            speaker.score = score
                            speaker.save()

                        segments = []
                        for segment in Segment.objects.all():
                            sentiment_analyze_result = voice_sentiment_analyze(segment.audio)
                            segments.append({
                                'speaker': segment.speaker.most_matching_recorded_speaker.name,
                                'score': segment.speaker.score,
                                'sentiment': segment.sentiment,
                                'sentiment_score': segment.sentiment_score,
                                'text': segment.text,
                                'happy': sentiment_analyze_result["mutlu"],
                                'angry': sentiment_analyze_result["sinirli"],
                                'sad': sentiment_analyze_result["uzgun"],
                            })
                        # Encode the histogram image to base64 to send to the frontend
                        histogram_image = generate_audio_histogram(file_path)
                        histogram_base64 = base64.b64encode(histogram_image).decode('utf-8')
                        return JsonResponse(
                        {'status': 'success', 'speaker_segments': segments, 'topic': topic_analysis, 'histogram': histogram_base64})
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


def parse_and_save_speaker_audios(file_path):
    try:
        audio = AudioSegment.from_file(file_path, format="wav")
        segments = Segment.objects.all()
        for segment in segments:
            start_time = segment.start * 1000
            end_time = segment.end * 1000
            segment_audio = audio[start_time:end_time]
            output_dir = "recordings/segment"
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
            output_file_path = os.path.join(output_dir, f"segment_{segment.id}.wav")

            # Export the audio segment
            segment_audio.export(output_file_path, format="wav")

            # Save the file path to the segment instance
            segment.audio = output_file_path
            segment.save()
        return True
    except Exception as e:
        return e

def concatenate_speaker_audio(speaker):
    concatenated_speaker_audio = AudioSegment.empty()
    speaker_segmnets = Segment.objects.filter(speaker=speaker)
    for speaker_segment in speaker_segmnets:
        audio = AudioSegment.from_file(speaker_segment.audio, format="wav")
        audio_of_segment = audio
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


def voice_sentiment_analyze(audio_path):
    # Load the pre-trained model
    model = settings.EMOTION_RECOGNITION

    # Get labels from the model
    labels = model.classes_

    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Predict probabilities
    probabilities = model.predict_proba([mfccs_mean])

    # Prepare results
    result = {
        labels[0]: probabilities[0][0],
        labels[1]: probabilities[0][1],
        labels[2]: probabilities[0][2],
    }
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


def get_topic_analysis(text):
    client = Groq(api_key=GROK_TOKEN)
    chat_completion = client.chat.completions.create(
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": "Şuan da benim asistanımsın."
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": f"Bu text'in konusunu belirler misin? : {text}",
            }
        ],
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )
    return chat_completion.choices[0].message.content