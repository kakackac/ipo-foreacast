from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import Notification, Offering, Post, Subscription, User, Watchlist

admin.site.register(User, UserAdmin)
admin.site.register(Offering)
admin.site.register(Watchlist)
admin.site.register(Notification)


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'moderation_status', 'created_at']
    list_filter = ['moderation_status']
    search_fields = ['title', 'body']


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ['user', 'provider', 'product_id', 'status', 'expires_at']

    def get_readonly_fields(self, request, obj=None):
        return [f.name for f in self.model._meta.fields]

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
