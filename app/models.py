from django.db import models

class EmbeddedSpeakers(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, unique=True)
    embedding = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.name

class Speaker(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=30, unique=True, null=True, blank=True)
    most_matching_recorded_speaker = models.ForeignKey(EmbeddedSpeakers, on_delete=models.CASCADE, null=True, blank=True)
    score = models.FloatField(null=True, blank=True)

    def __str__(self):
        return self.name

class Segment(models.Model):
    id = models.AutoField(primary_key=True)
    start = models.FloatField(null=True, blank=True)
    end = models.FloatField(null=True, blank=True)
    text = models.TextField(null=True, blank=True)
    speaker = models.ForeignKey(Speaker, related_name='segments', on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"Segment ({self.start}-{self.end}): {self.text[:30]}..."

class Word(models.Model):
    id = models.AutoField(primary_key=True)
    segment = models.ForeignKey(Segment, related_name='words', on_delete=models.CASCADE)
    speaker = models.ForeignKey(Speaker, related_name='words', on_delete=models.CASCADE, null=True, blank=True)
    word = models.CharField(max_length=100, null=True, blank=True)
    start = models.FloatField(null=True, blank=True)
    end = models.FloatField(null=True, blank=True)
    score = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Word: {self.word} ({self.start}-{self.end})"

