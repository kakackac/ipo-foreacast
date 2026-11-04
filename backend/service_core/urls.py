from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health), path('auth/csrf/', views.csrf),
    path('auth/register/', views.register), path('auth/login/', views.sign_in),
    path('auth/logout/', views.sign_out), path('auth/password/', views.change_password), path('me/', views.me),
    path('offerings/', views.offerings), path('watchlist/', views.watchlist),
    path('watchlist/<int:identity>/', views.watchlist_item), path('notifications/', views.notifications),
    path('notifications/<int:identity>/read/', views.mark_read), path('subscription/', views.subscription),
    path('posts/', views.posts), path('posts/<int:identity>/', views.post_detail),
]
