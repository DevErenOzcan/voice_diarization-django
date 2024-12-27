from django.db import models

class SpeakerEmbeddings(models.Model):
    name = models.CharField(max_length=100, unique=True)
    embedding = models.JSONField()  # Embedding'i JSON formatında saklamak
    created_at = models.DateTimeField(auto_now_add=True)
