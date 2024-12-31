import base64
import os
from io import BytesIO

from django.utils.lorem_ipsum import words
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
from scipy.fft import fft


TRANSCRIPTION_API_URL = 'https://69b4-34-91-196-117.ngrok-free.app/'
GROK_API_URL = "https://api.grok.com/topic-analysis"
GROK_TOKEN = "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"

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

                        response = []
                        segments = Segment.objects.all()
                        for segment in segments:
                            sentiment_analyze_result = voice_sentiment_analyze(segment.audio)
                            response.append({
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
                        histogram_image = generate_audio_histogram(file_path, speakers)
                        histogram_base64 = base64.b64encode(histogram_image).decode('utf-8')
                        return JsonResponse(
                        {'status': 'success', 'speaker_segments': response, 'topic': topic_analysis, 'histogram': histogram_base64})
                else:
                    return JsonResponse({'error': True, 'message': 'An error occurred while saving segments. Error: '})
            elif response.status_code == 404:
                return JsonResponse({'error': True, 'message': "Page not found error from api"})
            elif response.status_code == 500:
                return JsonResponse({'error': True, 'message': "Internal server error from api"})
            else:
                return JsonResponse({'error': True, 'message': "Unknown error from api"})
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


def generate_audio_histogram(file_path, speakers):
    # Load the audio file using pydub
    audio = AudioSegment.from_wav(file_path)

    # Convert audio to numpy array
    samples = np.array(audio.get_array_of_samples())

    # Create a figure with customized size
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Waveform Histogram (Amplitude vs Time)
    axes[0].plot(samples, color='blue')  # Plot all the samples
    axes[0].set_title("Audio Waveform")
    axes[0].set_xlabel("Samples")
    axes[0].set_ylabel("Amplitude")

    # Spectrogram
    axes[1].specgram(samples, NFFT=1028, Fs=audio.frame_rate, cmap='viridis')
    axes[1].set_title("Spectrogram")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_ylim(0, 8000)

    # Frequency Spectrum as a Pie Chart

    labels = []
    word_counts = []
    for speaker in speakers:
        labels.append(EmbeddedSpeakers.objects.get(id=speaker.most_matching_recorded_speaker.id).name)
        word_counts.append(Word.objects.filter(speaker=speaker).count())


    # Increase the size of the pie chart by adjusting the radius
    axes[2].pie(word_counts, labels=labels, autopct="%1.1f%%", textprops={'fontsize': 10}, radius=1.3)
    axes[2].set_title("Pie Chart")

    # Adjust layout
    plt.tight_layout()

    # Save the plot as an image in memory
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close(fig)
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
                "content": "Sen benim sadece türkçe konuşan asistanımsın ."
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": f"Bu metni analiz et ve konuşmanın genel olarak ne hakkında olduğunu kısaca açıkla. Yanıtın yalnızca Türkçe olmalı. İşte metin: {text}"
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