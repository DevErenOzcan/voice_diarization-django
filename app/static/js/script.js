document.addEventListener("DOMContentLoaded", function () {
    const recordToggle = document.getElementById("record-toggle");
    const audioPlayback = document.getElementById("audio-playback");
    const speakerNameInput = document.getElementById("speaker_name");
    const statusMessage = document.getElementById("status-message");

    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;

    // Toggle recording
    recordToggle.addEventListener("click", function () {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    // Start recording function
    function startRecording() {
        statusMessage.textContent = "Recording...";

        navigator.mediaDevices.getUserMedia({audio: true})
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                isRecording = true;
                recordToggle.textContent = "Stop Recording";
                recordToggle.style.backgroundColor = '#f44336';

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
            })
            .catch(error => {
                console.error("Error accessing microphone:", error);
                statusMessage.textContent = "Error accessing microphone. Please check your permissions.";
            });
    }

    // Stop recording function
    function stopRecording() {
        if (!mediaRecorder) return;

        mediaRecorder.stop();
        isRecording = false;
        recordToggle.textContent = "Start Recording";
        recordToggle.style.backgroundColor = '#4CAF50';

        mediaRecorder.onstop = () => {
            statusMessage.textContent = "Recording stopped. Processing audio...";

            const audioBlob = new Blob(audioChunks, {type: 'audio/wav'});
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayback.src = audioUrl;
            audioPlayback.hidden = false;

            // Send audio file to the server
            uploadAudio(audioBlob);
        };
    }

    // Function to upload audio to the server
    function uploadAudio(audioBlob) {
        const formData = new FormData();
        formData.append('audio_file', audioBlob);

        fetch('/transcribe_audio/', {
            method: 'POST',
            body: formData,
        })
            .then(response => {
                if (response.ok) {
                    return response.json();
                } else {
                    throw new Error("Failed to upload audio.");
                }
            })
            .then(data => {
                // Update the status message
                document.getElementById('status-message').innerText = 'Transcription Complete';

                // Parse transcription result
                let transcriptionText = '';
                const transcriptionResult = data; // Assuming server returns transcriptionResult in response
                transcriptionResult.speaker_segments.forEach(segment => {
                    transcriptionText += `<h3>${segment.speaker}: score: ${segment.score} ${segment.sentiment}: score: ${segment.sentiment_score} </h3>
                      <p>${segment.text}</p>
                      <p>Sentiments: Happy: ${segment.happy}, 
                                     Angry: ${segment.angry}, 
                                     Sad: ${segment.sad}</p>`;
                });

                document.getElementById('status-message').innerHTML += transcriptionText;

                document.getElementById('status-message').innerHTML += transcriptionResult.topic


                // Display the histogram image
                const histogramImage = `data:image/png;base64,${transcriptionResult.histogram}`;
                const histogramImgElement = document.createElement('img');
                histogramImgElement.src = histogramImage;
                document.getElementById('status-message').appendChild(histogramImgElement);
            })
            .catch(error => {
                console.error("Error uploading audio:", error);
                statusMessage.textContent = "Failed to upload audio. Please try again.";
            });
    }
});
