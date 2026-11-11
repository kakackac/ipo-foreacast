import hashlib
import json
from functools import wraps

from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.password_validation import validate_password
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.db.models import Q
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.utils import timezone
from django.views.decorators.debug import sensitive_post_parameters

from .models import Notification, Offering, Post, Subscription, User, Watchlist


def endpoint(methods, private=False):
    def decorate(function):
        @wraps(function)
        def wrapped(request, *args, **kwargs):
            if request.method not in methods:
                response = JsonResponse({'error': 'method_not_allowed'}, status=405)
                response['Allow'] = ', '.join(methods)
                return response
            if private and not request.user.is_authenticated:
                return JsonResponse({'error': 'authentication_required'}, status=401)
            try:
                request.payload = {}
                if request.body:
                    if request.content_type != 'application/json':
                        return JsonResponse({'error': 'json_required'}, status=415)
                    request.payload = json.loads(request.body)
                    if not isinstance(request.payload, dict):
                        raise ValueError()
                return function(request, *args, **kwargs)
            except (ValueError, TypeError, ValidationError, IntegrityError):
                return JsonResponse({'error': 'invalid_input'}, status=400)
        return wrapped
    return decorate


def user_info(user):
    return {'id': user.pk, 'username': user.username, 'notification_enabled': user.notification_enabled}


def auth_limited(request):
    key = 'auth:' + hashlib.sha256(request.META.get('REMOTE_ADDR', 'unknown').encode()).hexdigest()
    cache.add(key, 0, timeout=60)
    return cache.incr(key) > 20


@endpoint(['GET'])
def health(request):
    return JsonResponse({'status': 'ok', 'prediction_released': False, 'billing_enabled': False})


@endpoint(['GET'])
def csrf(request):
    return JsonResponse({'csrf_token': get_token(request)})


@sensitive_post_parameters()
@endpoint(['POST'])
def register(request):
    if auth_limited(request):
        return JsonResponse({'error': 'rate_limited'}, status=429)
    data = request.payload
    if not all(isinstance(data.get(k), str) for k in ['username', 'email', 'password']):
        raise ValueError()
    user = User(username=data['username'], email=data['email'].strip().lower())
    user.full_clean(exclude=['password'])
    validate_password(data['password'], user)
    user.set_password(data['password'])
    user.save()
    login(request, user)
    return JsonResponse(user_info(user), status=201)


@sensitive_post_parameters()
@endpoint(['POST'])
def sign_in(request):
    if auth_limited(request):
        return JsonResponse({'error': 'rate_limited'}, status=429)
    if not all(isinstance(request.payload.get(k), str) for k in ['username', 'password']):
        raise ValueError()
    user = authenticate(request, username=request.payload['username'], password=request.payload['password'])
    if user is None:
        return JsonResponse({'error': 'invalid_credentials'}, status=401)
    login(request, user)
    return JsonResponse(user_info(user))


@endpoint(['POST'], private=True)
def sign_out(request):
    logout(request)
    return JsonResponse({'signed_out': True})


@endpoint(['GET', 'PATCH'], private=True)
def me(request):
    if request.method == 'PATCH':
        value = request.payload.get('notification_enabled')
        if type(value) is not bool:
            raise ValueError()
        request.user.notification_enabled = value
        request.user.save(update_fields=['notification_enabled'])
    return JsonResponse(user_info(request.user))


@sensitive_post_parameters()
@endpoint(['POST'], private=True)
def change_password(request):
    old, new = request.payload.get('old_password'), request.payload.get('new_password')
    if not isinstance(old, str) or not isinstance(new, str) or not request.user.check_password(old):
        return JsonResponse({'error': 'invalid_credentials'}, status=400)
    validate_password(new, request.user)
    request.user.set_password(new)
    request.user.save(update_fields=['password'])
    update_session_auth_hash(request, request.user)
    return JsonResponse({'password_changed': True})


def page(query, request):
    number = int(request.GET.get('page', 1))
    if not 1 <= number <= 10000:
        raise ValueError()
    return query[(number - 1)*50:number*50]


@endpoint(['GET'])
def offerings(request):
    query = Offering.objects.all()
    if request.GET.get('q'):
        query = query.filter(company_name__icontains=request.GET['q'][:100])
    return JsonResponse({'results': list(page(query, request).values(
        'id', 'event_id', 'company_name', 'ticker', 'market', 'listing_date', 'offering_type', 'prediction_status', 'data_version'))})


@endpoint(['GET', 'POST'], private=True)
def watchlist(request):
    if request.method == 'POST':
        identity = request.payload.get('offering_id')
        if type(identity) is not int or not Offering.objects.filter(pk=identity).exists():
            raise ValueError()
        item, created = Watchlist.objects.get_or_create(user=request.user, offering_id=identity)
        return JsonResponse({'id': item.pk, 'offering_id': item.offering_id}, status=201 if created else 200)
    query = Watchlist.objects.filter(user=request.user).order_by('-created_at', '-id')
    return JsonResponse({'results': list(page(query, request).values('id', 'offering_id', 'created_at'))})


@endpoint(['DELETE'], private=True)
def watchlist_item(request, identity):
    deleted, _ = Watchlist.objects.filter(pk=identity, user=request.user).delete()
    return JsonResponse({'deleted': bool(deleted)}, status=200 if deleted else 404)


@endpoint(['GET'], private=True)
def notifications(request):
    query = Notification.objects.filter(user=request.user).order_by('-created_at', '-id')
    return JsonResponse({'results': list(page(query, request).values('id', 'message', 'read_at', 'created_at'))})


@endpoint(['POST'], private=True)
def mark_read(request, identity):
    updated = Notification.objects.filter(pk=identity, user=request.user, read_at__isnull=True).update(read_at=timezone.now())
    exists = updated or Notification.objects.filter(pk=identity, user=request.user).exists()
    return JsonResponse({'read': bool(exists)}, status=200 if exists else 404)


@endpoint(['GET', 'POST'], private=True)
def subscription(request):
    if request.method == 'POST':
        return JsonResponse({'error': 'server_payment_verification_not_configured'}, status=503)
    active = Subscription.objects.filter(user=request.user, status='active', verified_at__lte=timezone.now(),
                                         expires_at__gt=timezone.now()).exists()
    return JsonResponse({'active': active, 'billing_enabled': False})


def visible_posts(user):
    return Post.objects.filter(Q(moderation_status='published') | Q(author=user)) if user.is_authenticated else Post.objects.filter(moderation_status='published')


@endpoint(['GET', 'POST'])
def posts(request):
    if request.method == 'POST':
        if not request.user.is_authenticated:
            return JsonResponse({'error': 'authentication_required'}, status=401)
        title, body = request.payload.get('title'), request.payload.get('body')
        if not isinstance(title, str) or not isinstance(body, str) or not title.strip() or not body.strip():
            raise ValueError()
        post = Post(author=request.user, title=title.strip(), body=body.strip())
        post.full_clean()
        post.save()
        return JsonResponse({'id': post.pk, 'moderation_status': post.moderation_status}, status=201)
    query = visible_posts(request.user)
    return JsonResponse({'results': list(page(query, request).values('id', 'title', 'author_id', 'moderation_status', 'created_at'))})


@endpoint(['GET', 'PATCH', 'DELETE'])
def post_detail(request, identity):
    post = visible_posts(request.user).filter(pk=identity).first()
    if post is None:
        return JsonResponse({'error': 'not_found'}, status=404)
    if request.method != 'GET':
        if not request.user.is_authenticated or post.author_id != request.user.pk:
            return JsonResponse({'error': 'forbidden'}, status=403)
        if request.method == 'DELETE':
            post.delete()
            return JsonResponse({'deleted': True})
        for field in ['title', 'body']:
            if field in request.payload:
                value = request.payload[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError()
                setattr(post, field, value.strip())
        post.moderation_status = 'pending'
        post.full_clean()
        post.save()
    return JsonResponse({'id': post.pk, 'title': post.title, 'body': post.body,
                         'author_id': post.author_id, 'moderation_status': post.moderation_status})
