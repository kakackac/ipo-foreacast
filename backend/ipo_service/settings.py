import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from django.core.management.utils import get_random_secret_key

BASE_DIR = Path(__file__).resolve().parents[1]
DEVELOPMENT = os.getenv('DJANGO_ENV') == 'development'
DEBUG = DEVELOPMENT
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY')
if not SECRET_KEY:
    if not DEVELOPMENT:
        raise ImproperlyConfigured('DJANGO_SECRET_KEY is required')
    SECRET_KEY = get_random_secret_key()
ALLOWED_HOSTS = os.getenv('DJANGO_ALLOWED_HOSTS', '127.0.0.1,localhost').split(',')
if not DEVELOPMENT and (len(SECRET_KEY) < 50 or not os.getenv('DJANGO_ALLOWED_HOSTS') or '*' in ALLOWED_HOSTS):
    raise ImproperlyConfigured('Production requires a strong secret and explicit allowed hosts')
INSTALLED_APPS = [
    'django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes',
    'django.contrib.sessions', 'django.contrib.messages', 'django.contrib.staticfiles',
    'service_core',
]
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware', 'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware', 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware', 'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
ROOT_URLCONF = 'ipo_service.urls'
WSGI_APPLICATION = 'ipo_service.wsgi.application'
TEMPLATES = [{'BACKEND': 'django.template.backends.django.DjangoTemplates', 'APP_DIRS': True,
              'OPTIONS': {'context_processors': ['django.template.context_processors.request',
                          'django.contrib.auth.context_processors.auth', 'django.contrib.messages.context_processors.messages']}}]
if DEVELOPMENT:
    (BASE_DIR / 'var').mkdir(exist_ok=True)
    DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': BASE_DIR / 'var/db.sqlite3'}}
else:
    DATABASES = {'default': {'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ['POSTGRES_DB'], 'USER': os.environ['POSTGRES_USER'],
        'PASSWORD': os.environ['POSTGRES_PASSWORD'], 'HOST': os.environ['POSTGRES_HOST'],
        'PORT': os.getenv('POSTGRES_PORT', '5432')}}
AUTH_USER_MODEL = 'service_core.User'
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 12}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]
TIME_ZONE = 'Asia/Seoul'
USE_TZ = True
LANGUAGE_CODE = 'ko-kr'
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'var/static'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
DATA_UPLOAD_MAX_MEMORY_SIZE = 65536
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_SECURE = not DEVELOPMENT
CSRF_COOKIE_SECURE = not DEVELOPMENT
SECURE_SSL_REDIRECT = not DEVELOPMENT
SECURE_HSTS_SECONDS = 31536000 if not DEVELOPMENT else 0
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
IPO_PROCESSED_DIR = BASE_DIR.parent / 'ipo_predictor/data/processed'
