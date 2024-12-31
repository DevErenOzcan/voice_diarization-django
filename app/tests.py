from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch, MagicMock


class AudioFileTestCase(TestCase):
    def setUp(self):
        # Test için farklı dosya formatları oluşturuyorum
        self.audio_files = [
            ("test_audio.mp3", b"Fake MP3 content", "audio/mpeg"),
            ("test_audio.wav", b"Fake WAV content", "audio/wav"),
            ("test_audio.ogg", b"Fake OGG content", "audio/ogg"),
            ("test_audio.flac", b"Fake FLAC content", "audio/flac"),
            ("test_audio.aac", b"Fake AAC content", "audio/aac"),
            ("test_audio.m4a", b"Fake M4A content", "audio/mp4"),
            ("test_audio.wma", b"Fake WMA content", "audio/x-ms-wma"),
            ("test_audio.amr", b"Fake AMR content", "audio/amr"),
            ("test_audio.aiff", b"Fake AIFF content", "audio/aiff"),
            ("test_audio.opus", b"Fake OPUS content", "audio/opus"),
        ]

        self.test_audio_files = [
            SimpleUploadedFile(name, content, content_type=ctype)
            for name, content, ctype in self.audio_files
        ]

    @patch('app.views.default_storage.save')
    @patch('app.views.default_storage.path')
    @patch('app.views.AudioSegment.from_file')
    @patch('app.views.default_storage.delete')
    def test_save_audio_file_success(self, mock_delete, mock_from_file, mock_path, mock_save):
        # Mocking for file saving
        mock_save.return_value = "mock_path/test_audio"
        mock_path.return_value = "/mock_path/test_audio"

        # Mocking for audio conversion
        mock_audio = MagicMock()
        mock_from_file.return_value = mock_audio
        mock_audio.export.return_value = True

        from app.views import save_audio_file

        for i, audio_file in enumerate(self.test_audio_files):
            with self.subTest(file_format=self.audio_files[i][0]):
                success, message = save_audio_file(audio_file)

                # Fonksiyonun True dönemsini bekliyoruz
                self.assertTrue(success)
                self.assertTrue(message.endswith(".wav"))
                mock_save.assert_called_once()
                mock_from_file.assert_called_once_with("/mock_path/test_audio")
                mock_audio.export.assert_called_once_with("/mock_path/test_audio.wav", format="wav")
                mock_delete.assert_called_once_with("mock_path/test_audio")

                # Mock'ları sıfırlıyoruz
                mock_save.reset_mock()
                mock_from_file.reset_mock()
                mock_audio.export.reset_mock()
                mock_delete.reset_mock()

    @patch('app.views.default_storage.save')
    @patch('app.views.default_storage.path')
    @patch('app.views.AudioSegment.from_file')
    def test_save_audio_file_failure(self, mock_from_file, mock_path, mock_save):
        # Mocking for file saving
        mock_save.return_value = "mock_path/test_audio"
        mock_path.return_value = "/mock_path/test_audio"

        # Mocking for audio conversion to raise an exception
        mock_from_file.side_effect = Exception("Conversion error")

        from app.views import save_audio_file

        for i, audio_file in enumerate(self.test_audio_files):
            with self.subTest(file_format=self.audio_files[i][0]):
                success, message = save_audio_file(audio_file)

                # False bekliyoruz
                self.assertFalse(success)
                self.assertEqual(str(message), "Conversion error")

                # Mock'ları sıfırlıyoruz
                mock_save.reset_mock()
                mock_from_file.reset_mock()

import unittest
import os

@patch('pydub.AudioSegment.from_file')  # Doğru yolu kullanıyoruz
@patch('app.views.os.path.join')
@patch('app.views.os.makedirs')
@patch('app.models.Segment.objects.all')  # Doğru modeli mock'layın
def test_parse_and_save_speaker_audios_success(self, mock_segment_all, mock_makedirs, mock_path_join,
                                               mock_audio_from_file):
    # Mock AudioSegment.from_file döndürülen nesneyi ayarla
    mock_audio = MagicMock()
    mock_audio_from_file.return_value = mock_audio

    # Mock audio segment slicing
    mock_segment_audio = MagicMock()
    mock_audio.__getitem__.return_value = mock_segment_audio

    # Mock Segment model'ındaki nesneleri ayarla
    mock_segment1 = MagicMock()
    mock_segment1.start = 1.0  # Başlangıç zamanı (saniye)
    mock_segment1.end = 2.0  # Bitiş zamanı (saniye)
    mock_segment1.id = 101
    mock_segment1.audio = None

    mock_segment2 = MagicMock()
    mock_segment2.start = 3.0
    mock_segment2.end = 4.0
    mock_segment2.id = 102
    mock_segment2.audio = None

    mock_segment_all.return_value = [mock_segment1, mock_segment2]

    # os.path.join davranışını tanımla
    mock_path_join.side_effect = lambda *args: os.path.join(*args)

    # Fonksiyonu çağır
    result = parse_and_save_speaker_audios("dummy_path.wav")

    # Sonucun True olduğunu doğrula
    self.assertTrue(result)

    # AudioSegment.from_file'ın doğru şekilde çağrıldığını doğrula
    mock_audio_from_file.assert_called_once_with("dummy_path.wav", format="wav")

    # os.makedirs'ın doğru şekilde çağrıldığını doğrula
    mock_makedirs.assert_called_once_with("recordings/segment", exist_ok=True)

    # Audio'nun doğru zaman diliminde kesildiğini doğrula
    expected_slices = [
        unittest.mock.call(slice(1000.0, 2000.0, None)),  # segment1 için
        unittest.mock.call(slice(3000.0, 4000.0, None))  # segment2 için
    ]
    mock_audio.__getitem__.assert_has_calls(expected_slices, any_order=False)

    # segment_audio.export'un doğru parametrelerle çağrıldığını doğrula
    expected_exports = [
        unittest.mock.call(os.path.join("recordings/segment", "segment_101.wav"), format="wav"),
        unittest.mock.call(os.path.join("recordings/segment", "segment_102.wav"), format="wav")
    ]
    mock_segment_audio.export.assert_has_calls(expected_exports, any_order=False)

    # Segment nesnelerinin audio alanlarının doğru şekilde ayarlandığını ve save metodunun çağrıldığını doğrula
    self.assertEqual(mock_segment1.audio, os.path.join("recordings/segment", "segment_101.wav"))
    self.assertEqual(mock_segment2.audio, os.path.join("recordings/segment", "segment_102.wav"))
    mock_segment1.save.assert_called_once()
    mock_segment2.save.assert_called_once()

    @patch('pydub.AudioSegment.from_file')  # Doğru yolu kullanıyoruz
    def test_parse_and_save_speaker_audios_exception(self, mock_audio_from_file):
        # AudioSegment.from_file metodunun istisna fırlatmasını sağla
        mock_audio_from_file.side_effect = Exception("Dosya yüklenemedi")

        # Fonksiyonu çağır
        result = parse_and_save_speaker_audios("dummy_path.wav")

        # Sonucun bir istisna nesnesi olduğunu doğrula
        self.assertIsInstance(result, Exception)
        self.assertEqual(str(result), "Dosya yüklenemedi")




from app.views import get_topic_analysis, parse_and_save_speaker_audios


class TestTopicAnalysis(TestCase):
    @patch('app.views.Groq')
    def test_get_topic_analysis(self, MockGroq):
        # Arrange
        mock_groq_instance = MagicMock()
        MockGroq.return_value = mock_groq_instance

        # Mocking the API response structure
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This text is about AI and machine learning."
        mock_groq_instance.chat.completions.create.return_value = mock_response

        # Define the text to be analyzed
        text = "Bu text'in konusunu belirler misin?"

        # Act
        result = get_topic_analysis(text)

        # Assert
        self.assertIsInstance(result, str)