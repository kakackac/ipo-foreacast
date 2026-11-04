import json
import secrets

from django.core.cache import cache
from django.test import Client, TestCase
from django.utils import timezone

from .models import Notification, Offering, Post, Subscription, User, Watchlist


class ServiceTests(TestCase):
    def setUp(self):
        cache.clear()
        self.password = secrets.token_urlsafe(24)
        self.user = User.objects.create_user('owner', 'owner@example.test', self.password)
        self.other = User.objects.create_user('other', 'other@example.test', secrets.token_urlsafe(24))
        self.offering = Offering.objects.create(event_id='event', company_name='Sample', market='KOSDAQ',
                                                listing_date='2026-01-02', data_version='test')

    def send(self, path, data=None, method='post', client=None):
        return getattr(client or self.client, method)(f'/api/v1/{path}', data=json.dumps(data or {}), content_type='application/json')

    def test_register_hashes_password_and_ignores_client_privileges(self):
        password = secrets.token_urlsafe(24)
        result = self.send('auth/register/', {'username': 'new', 'email': 'NEW@example.test',
                                              'password': password, 'is_staff': True})
        self.assertEqual(result.status_code, 201)
        user = User.objects.get(username='new')
        self.assertTrue(user.check_password(password))
        self.assertFalse(user.is_staff)
        self.assertNotIn(password, result.content.decode())
        self.assertEqual(user.email, 'new@example.test')

    def test_csrf_required_even_for_login(self):
        client = Client(enforce_csrf_checks=True)
        data = {'username': 'owner', 'password': self.password}
        self.assertEqual(self.send('auth/login/', data, client=client).status_code, 403)
        token = client.get('/api/v1/auth/csrf/').json()['csrf_token']
        response = client.post('/api/v1/auth/login/', data=json.dumps(data), content_type='application/json', HTTP_X_CSRFTOKEN=token)
        self.assertEqual(response.status_code, 200)

    def test_login_and_logout_control_private_access(self):
        self.assertEqual(self.client.get('/api/v1/me/').status_code, 401)
        self.assertEqual(self.send('auth/login/', {'username': 'owner', 'password': self.password}).status_code, 200)
        self.assertEqual(self.client.get('/api/v1/me/').json()['id'], self.user.pk)
        self.send('auth/logout/')
        self.assertEqual(self.client.get('/api/v1/watchlist/').status_code, 401)

    def test_watchlist_is_idempotent_and_private(self):
        self.client.force_login(self.user)
        payload = {'offering_id': self.offering.pk, 'user_id': self.other.pk}
        first = self.send('watchlist/', payload)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(self.send('watchlist/', payload).status_code, 200)
        self.assertEqual(Watchlist.objects.count(), 1)
        self.client.force_login(self.other)
        self.assertEqual(self.client.get('/api/v1/watchlist/').json()['results'], [])
        self.assertEqual(self.send(f'watchlist/{first.json()["id"]}/', method='delete').status_code, 404)
        self.assertEqual(Watchlist.objects.count(), 1)

    def test_notifications_cannot_be_read_by_another_account(self):
        notification = Notification.objects.create(user=self.user, message='Changed', deduplication_key='key')
        self.client.force_login(self.other)
        self.assertEqual(self.send(f'notifications/{notification.pk}/read/').status_code, 404)
        self.client.force_login(self.user)
        self.assertEqual(self.send(f'notifications/{notification.pk}/read/').status_code, 200)
        self.assertEqual(self.send(f'notifications/{notification.pk}/read/').status_code, 200)

    def test_client_cannot_activate_a_subscription(self):
        self.client.force_login(self.user)
        self.assertEqual(self.send('subscription/', {'status': 'active', 'verified': True}).status_code, 503)
        self.assertEqual(Subscription.objects.count(), 0)
        Subscription.objects.create(user=self.user, provider='test', transaction_reference='fixture',
                                    product_id='fixture', status='active', expires_at=timezone.now() + timezone.timedelta(days=1))
        self.assertFalse(self.client.get('/api/v1/subscription/').json()['active'])

    def test_post_moderation_and_authorization(self):
        self.client.force_login(self.user)
        result = self.send('posts/', {'title': 'Title', 'body': 'Body', 'moderation_status': 'published'})
        self.assertEqual(result.status_code, 201)
        identity = result.json()['id']
        self.assertEqual(result.json()['moderation_status'], 'pending')
        self.client.logout()
        self.assertEqual(self.client.get('/api/v1/posts/').json()['results'], [])
        self.assertEqual(self.client.get(f'/api/v1/posts/{identity}/').status_code, 404)
        Post.objects.filter(pk=identity).update(moderation_status='published')
        self.assertEqual(len(self.client.get('/api/v1/posts/').json()['results']), 1)
        self.client.force_login(self.other)
        self.assertEqual(self.send(f'posts/{identity}/', {'title': 'Hijack'}, method='patch').status_code, 403)
        self.client.force_login(self.user)
        self.assertEqual(self.send(f'posts/{identity}/', {'body': 'Revised'}, method='patch').json()['moderation_status'], 'pending')

    def test_notification_preference_is_strict_boolean(self):
        self.client.force_login(self.user)
        self.assertEqual(self.send('me/', {'notification_enabled': 'false'}, method='patch').status_code, 400)
        self.assertFalse(self.send('me/', {'notification_enabled': False}, method='patch').json()['notification_enabled'])

    def test_invalid_json_and_pagination_are_rejected(self):
        self.assertEqual(self.client.post('/api/v1/auth/login/', data='broken', content_type='application/json').status_code, 400)
        self.assertEqual(self.client.get('/api/v1/offerings/?page=0').status_code, 400)
        self.assertEqual(self.send('offerings/').status_code, 405)

    def test_password_change_does_not_accept_wrong_current_password(self):
        self.client.force_login(self.user)
        new = secrets.token_urlsafe(24)
        self.assertEqual(self.send('auth/password/', {'old_password': 'incorrect', 'new_password': new}).status_code, 400)
        self.assertEqual(self.send('auth/password/', {'old_password': self.password, 'new_password': new}).status_code, 200)
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password(new))

    def test_throttle_does_not_reveal_credentials(self):
        for _ in range(20):
            self.send('auth/login/', {'username': 'unknown', 'password': 'invalid'})
        response = self.send('auth/login/', {'username': 'unknown', 'password': 'invalid'})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json(), {'error': 'rate_limited'})
