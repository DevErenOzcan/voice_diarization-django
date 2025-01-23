document.addEventListener("DOMContentLoaded", function () {
    const recordToggle = document.getElementById("record-toggle");
    const audioPlayback = document.getElementById("audio-playback");
    const statusMessage = document.getElementById("status-message");
    const loadingSpinner = document.getElementById("loading-spinner"); // Yükleniyor simgesi için div


    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;

    // Toggle recording
    recordToggle.addEventListener("click", function () {
        isRecording ? stopRecording() : startRecording();
        toggleRecordingState();
    });

    // Start recording function
    function startRecording() {
        audioChunks = [];
        statusMessage.textContent = "Recording...";

        navigator.mediaDevices.getUserMedia({audio: true})
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                isRecording = true;

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
        if (!mediaRecorder) {
            console.warn("MediaRecorder is not initialized.");
            return;
        }

        mediaRecorder.stop();
        isRecording = false;
        statusMessage.textContent = "Recording stopped. Processing audio...";

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, {type: "audio/wav"});
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayback.src = audioUrl;
            audioPlayback.hidden = !audioPlayback.src;

            // Yükleniyor simgesini göster
            showLoadingSpinner();
            uploadAudio(audioBlob);
        };
    }

    // Toggle recording state
    function toggleRecordingState() {
        recordToggle.classList.toggle("recording");
        recordToggle.textContent = isRecording ? "Start Recording" : "Stop Recording";
    }

    // Function to upload audio to the server
    function uploadAudio(audioBlob) {
        const formData = new FormData();
        formData.append("audio_file", audioBlob);

        fetch("/transcribe_audio/", {
            method: "POST",
            body: formData,
        })
            .then(response =>
                response.ok ? response.json() : Promise.reject("Failed to upload audio.")
            )
            .then(data => {
                displayTranscriptionResults(data);
                hideLoadingSpinner(); // Yükleniyor simgesini gizle
                recordToggle.textContent = "Transcribe New Audio";
            })
            .catch(error => {
                console.error("Error uploading audio:", error);
                recordToggle.textContent = "Transcribe New Audio";
                statusMessage.textContent = "Failed to upload audio. Please try again.";
                hideLoadingSpinner(); // Hata durumunda da yükleniyor simgesini gizle
            });
    }

    // Function to show loading spinner
    function showLoadingSpinner() {
        recordToggle.style.display = "none"; // Butonu gizle
        loadingSpinner.style.display = "block"; // Yükleniyor simgesini göster
    }

    // Function to hide loading spinner
    function hideLoadingSpinner() {
        loadingSpinner.style.display = "none"; // Yükleniyor simgesini gizle
        recordToggle.style.display = "block"; // Butonu geri getir
    }

    // Function to display transcription results
    function displayTranscriptionResults(data) {
        statusMessage.textContent = "Transcription Complete";

        if (data.speaker_segments) {
            data.speaker_segments.forEach(segment => {
                // Segment container
                const segmentCard = document.createElement("div");
                segmentCard.classList.add("segment-card");

                // Speaker information
                const speakerInfo = document.createElement("h3");
                speakerInfo.classList.add("segment-speaker");
                speakerInfo.textContent = `${segment.speaker}: ${segment.score}`;
                segmentCard.appendChild(speakerInfo);

                // Segment text
                const textParagraph = document.createElement("p");
                textParagraph.classList.add("segment-text");
                textParagraph.textContent = segment.text;
                segmentCard.appendChild(textParagraph);

                // Sentiments section
                const sentimentsContainer = document.createElement("div");
                sentimentsContainer.classList.add("segment-sentiments");

                const happySentiment = document.createElement("p");
                happySentiment.textContent = `Voice-Happy: ${segment.happy}`;
                sentimentsContainer.appendChild(happySentiment);

                const angrySentiment = document.createElement("p");
                angrySentiment.textContent = `Voice-Angry: ${segment.angry}`;
                sentimentsContainer.appendChild(angrySentiment);

                const sadSentiment = document.createElement("p");
                sadSentiment.textContent = `Voice-Sad: ${segment.sad}`;
                sentimentsContainer.appendChild(sadSentiment);

                segmentCard.appendChild(sentimentsContainer);

                // text Sentiments section
                const textsentimentsContainer = document.createElement("div");
                textsentimentsContainer.classList.add("segment-sentiments");

                const texthappySentiment = document.createElement("p");
                texthappySentiment.textContent = `Text-Happy: ${segment.positive_score}`;
                textsentimentsContainer.appendChild(texthappySentiment);

                const textsadSentiment = document.createElement("p");
                textsadSentiment.textContent = `Text-Sad: ${segment.negative_score}`;
                textsentimentsContainer.appendChild(textsadSentiment);

                segmentCard.appendChild(textsentimentsContainer);

                // average Sentiments section
                const averagesentimentsContainer = document.createElement("div");
                averagesentimentsContainer.classList.add("segment-sentiments");

                const averagehappySentiment = document.createElement("p");
                averagehappySentiment.textContent = `Average-Happy: ${segment.average_happy}`;
                averagesentimentsContainer.appendChild(averagehappySentiment);

                const averagesadSentiment = document.createElement("p");
                averagesadSentiment.textContent = `Average-Sad: ${segment.average_sad}`;
                averagesentimentsContainer.appendChild(averagesadSentiment);

                segmentCard.appendChild(averagesentimentsContainer);

                // Append segment card to status message
                statusMessage.appendChild(segmentCard);
            });
        }

        if (data.topic) {
            const topicContainer = document.createElement("div");
            topicContainer.classList.add("topic-container");

            const topicTitle = document.createElement("h3");
            topicTitle.classList.add("topic-title");
            topicTitle.textContent = "Topic";

            const topicDescription = document.createElement("p");
            topicDescription.classList.add("topic-description");
            topicDescription.textContent = data.topic;

            topicContainer.appendChild(topicTitle);
            topicContainer.appendChild(topicDescription);

            statusMessage.appendChild(topicContainer);
        }

        if (data.histogram) {
            const histogramImg = document.createElement("img");
            histogramImg.src = `data:image/png;base64,${data.histogram}`;
            histogramImg.style.maxWidth = "100%";
            histogramImg.style.marginTop = "20px";
            statusMessage.appendChild(histogramImg);
        }
    }
});
