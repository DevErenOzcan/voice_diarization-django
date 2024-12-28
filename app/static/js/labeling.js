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
        const speakerName = speakerNameInput.value.trim();
        if (!speakerName) {
            statusMessage.textContent = "Please enter a speaker name before recording.";
            return;
        }
        statusMessage.textContent = "Recording...";

        navigator.mediaDevices.getUserMedia({ audio: true })
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

            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayback.src = audioUrl;
            audioPlayback.hidden = false;

            // Send audio file to the server
            uploadAudio(audioBlob);
        };
    }

    // Function to upload audio to the server
    function uploadAudio(audioBlob) {
        const speakerName = speakerNameInput.value.trim();
        const formData = new FormData();
        formData.append('audio_file', audioBlob, `${speakerName}.wav`);
        formData.append('speaker_name', speakerName);

        fetch('/person_labeling/', {
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
                statusMessage.textContent = "Audio uploaded successfully!";
                console.log("Server response:", data);
            })
            .catch(error => {
                console.error("Error uploading audio:", error);
                statusMessage.textContent = "Failed to upload audio. Please try again.";
            });
    }
});
