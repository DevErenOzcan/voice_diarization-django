# Projenin kurulumu için yapılması gerekenler:
### Environment oluştur:
    # virtual env i kur (ubuntu):
    python3.12 -m venv .venv
    source .venv/bin/activate

    # virtual env i kur (windows):
    python3.12 -m venv .venv
    .venv\Scripts\activate

### Gerekli paketleri indir:
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

### Migrationsları yap:
    python manage.py makemigrations
    python manage.py migrate

### Django serverı başlat:
    python manage.py runserver

8000 portundan uygulamaya ulaşabilirsiniz
    