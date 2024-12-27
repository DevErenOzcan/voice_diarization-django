from django.apps import AppConfig


class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'

    def ready(self):
        from django.conf import settings
        from pyannote.audio import Model, Inference
        settings.INFERENCE = Inference(Model.from_pretrained("pyannote/embedding",
                               use_auth_token="<HF_TOKEN>"), window="whole")