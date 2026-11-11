from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    email = models.EmailField(unique=True)
    notification_enabled = models.BooleanField(default=True)


class Offering(models.Model):
    event_id = models.CharField(max_length=255, unique=True)
    company_name = models.CharField(max_length=200)
    ticker = models.CharField(max_length=20, blank=True)
    market = models.CharField(max_length=10, choices=[('KOSPI', 'KOSPI'), ('KOSDAQ', 'KOSDAQ')])
    listing_date = models.DateField()
    offering_type = models.CharField(max_length=40, default='general_ipo')
    data_version = models.CharField(max_length=64)
    prediction_status = models.CharField(max_length=40, default='not_released', editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-listing_date', 'id']


class Watchlist(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    offering = models.ForeignKey(Offering, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['user', 'offering'], name='unique_user_offering')]


class Notification(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    message = models.CharField(max_length=500)
    deduplication_key = models.CharField(max_length=255)
    read_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['user', 'deduplication_key'], name='unique_user_notification')]


class Subscription(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    provider = models.CharField(max_length=30)
    transaction_reference = models.CharField(max_length=255, unique=True)
    product_id = models.CharField(max_length=100)
    status = models.CharField(max_length=20, default='pending', choices=[('pending', 'pending'), ('active', 'active'), ('expired', 'expired'), ('revoked', 'revoked')])
    verified_at = models.DateTimeField(null=True, editable=False)
    expires_at = models.DateTimeField(null=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)


class Post(models.Model):
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    title = models.CharField(max_length=160)
    body = models.TextField(max_length=10000)
    offering = models.ForeignKey(Offering, on_delete=models.SET_NULL, null=True, blank=True)
    moderation_status = models.CharField(max_length=20, default='pending', choices=[('pending', 'pending'), ('published', 'published'), ('hidden', 'hidden')])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at', '-id']
