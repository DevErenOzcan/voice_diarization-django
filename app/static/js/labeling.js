let mediaRecorder;
let audioChunks = [];
let isRecording = false;

document.getElementById('record-toggle').addEventListener('click', async () => {
    const button = document.getElementById('record-toggle');

    if (isRecording) {
        mediaRecorder.stop();
        button.innerText = 'Start Recording';
        button.style.backgroundColor = '#4CAF50';  // Yeşil renk
        document.getElementById('status-message').innerText = 'Processing...';
    } else {
        const stream = await navigator.mediaDevices.getUserMedia({audio: true});
        mediaRecorder = new MediaRecorder(stream, {mimeType: 'audio/webm'});  // WebM formatında kaydetme

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };
        mediaRecorder.onstop = async () => {
            const blob = new Blob(audioChunks, {type: 'audio/webm'});

            // WAV formatına dönüştürme işlemi
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await blob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

            const wavBuffer = audioBufferToWav(audioBuffer);

            const reader = new FileReader();
            reader.onloadend = async () => {
                const base64Audio = reader.result; // Base64 ses verisini al

                document.getElementById('status-message').innerText = 'Embedding...';
                speaker_name = document.getElementById('speaker_name').value;

                const transcriptionResponse = await fetch('/person_labeling/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `audio=${encodeURIComponent(base64Audio)}&speaker_name=${encodeURIComponent(speaker_name)}`,
                });

                const transcriptionResult = await transcriptionResponse.json();

                if (transcriptionResult.status === 'success') {
                    document.getElementById('status-message').innerText = 'Speaker Saved';
                    let transcriptionText = '';

                    transcriptionResult.speaker_segments.forEach(segment => {
                        transcriptionText += `<h3>${segment.speaker}:</h3><p>${segment.text}</p>`;
                    });

                    document.getElementById('status-message').innerHTML += transcriptionText;

                    // Histogram görselini göster
                    const histogramImage = `data:image/png;base64,${transcriptionResult.histogram}`;
                    const histogramImgElement = document.createElement('img');
                    histogramImgElement.src = histogramImage;
                    document.getElementById('status-message').appendChild(histogramImgElement);
                } else {
                    document.getElementById('status-message').innerText = transcriptionResult.message;
                }
            };
            reader.readAsDataURL(new Blob([wavBuffer], {type: 'audio/wav'}));
        };

        mediaRecorder.start();
        button.innerText = 'Stop Recording';
        button.style.backgroundColor = '#f44336';  // Kırmızı renk
        document.getElementById('status-message').innerText = 'Recording...';
    }

    isRecording = !isRecording;
});

// WAV dönüştürme fonksiyonu
function audioBufferToWav(audioBuffer) {
    const numOfChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const length = audioBuffer.length * numOfChannels;
    const buffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(buffer);
    let offset = 0;

    const writeString = (str) => {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset++, str.charCodeAt(i));
        }
    };

    // RIFF header
    writeString('RIFF');
    view.setUint32(offset, 36 + length * 2, true);
    offset += 4;
    writeString('WAVE');

    // fmt chunk
    writeString('fmt ');
    view.setUint32(offset, 16, true);
    offset += 4; // Subchunk1Size
    view.setUint16(offset, 1, true);
    offset += 2;  // AudioFormat (PCM)
    view.setUint16(offset, numOfChannels, true);
    offset += 2;
    view.setUint32(offset, sampleRate, true);
    offset += 4;
    view.setUint32(offset, sampleRate * numOfChannels * 2, true);
    offset += 4;  // ByteRate
    view.setUint16(offset, numOfChannels * 2, true);
    offset += 2;  // BlockAlign
    view.setUint16(offset, 16, true);
    offset += 2;  // BitsPerSample

    // data chunk
    writeString('data');
    view.setUint32(offset, length * 2, true);
    offset += 4;

    // Writing audio data (16-bit signed PCM)
    for (let i = 0; i < audioBuffer.length; i++) {
        for (let channel = 0; channel < numOfChannels; channel++) {
            const sample = audioBuffer.getChannelData(channel)[i];
            view.setInt16(offset, sample * 0x7FFF, true); // 16-bit PCM
            offset += 2;
        }
    }

    return new Uint8Array(buffer);
}
