from django.urls import path
from .views import (
    SubCategoryByCategoryView, register_user, login_user, get_user_profile, update_user_profile,
    upload_clothing, get_wardrobe,get_wardrobe_by_user,get_outfits_by_user,delete_planned_outfit,
    create_outfit, get_outfits,delete_outfit, delete_clothing, ai_generate_outfit,
    plan_outfit, get_planned_outfits,delete_post,get_user_profile_by_id,get_outfit_by_id,
    create_post, get_all_posts, toggle_like_post, toggle_follow, get_following_feed
)
from rest_framework.routers import DefaultRouter
from .views import CategoryViewSet, SubCategoryViewSet
from django.conf import settings
from django.conf.urls.static import static

router = DefaultRouter()
router.register(r'categories', CategoryViewSet, basename='category')
router.register(r'subcategories', SubCategoryViewSet, basename='subcategory')
urlpatterns = [
    path('register/', register_user, name='register_user'),
    path('login/', login_user, name='login_user'),
    path('profile/', get_user_profile, name='get_user_profile'),
    path('profile/update/', update_user_profile, name='update_user_profile'),
    path('users/<int:user_id>/profile/', get_user_profile_by_id, name='user_profile_by_id'),
    # New routes to get another user's wardrobe and outfits
    path('users/<int:user_id>/wardrobe/', get_wardrobe_by_user, name='get_wardrobe_by_user'),
    path('users/<int:user_id>/outfits/', get_outfits_by_user, name='get_outfits_by_user'),

    # Wardrobe APIs
    path('wardrobe/upload/', upload_clothing, name='upload_clothing'),
    path('wardrobe/', get_wardrobe, name='get_wardrobe'),
    path('wardrobe/<int:item_id>/', delete_clothing, name='delete_clothing'),

    # Outfit APIs
    path('outfits/create/', create_outfit, name='create_outfit'),
    path('outfits/', get_outfits, name='get_outfits'),
    path('outfits/ai-generate/', ai_generate_outfit, name='ai_generate_outfit'),
    path('outfits/<int:pk>/', delete_outfit, name='delete_outfit'),
    path('outfits/<int:pk>/', get_outfit_by_id, name='get_outfit_by_id'),

    # Outfit Planner APIs
    path('planner/', get_planned_outfits, name='get_planned_outfits'),
    path('planner/plan/', plan_outfit, name='plan_outfit'),
    path('planner/<int:pk>/delete/', delete_planned_outfit,name='delete_planned_outfit'),

    # Feed APIs
    path('feed/posts/create/', create_post, name='create_post'),
    path('feed/posts/', get_all_posts, name='get_all_posts'),
    path('feed/posts/<int:post_id>/like/', toggle_like_post, name='toggle_like_post'),
    path('feed/posts/<int:post_id>/delete/', delete_post, name='delete_post'),

    path('feed/follow/<int:user_id>/', toggle_follow, name='toggle_follow'),
    path('feed/following/', get_following_feed, name='get_following_feed'),
    path('categories/<int:category_id>/subcategories/', SubCategoryByCategoryView.as_view(), name='subcategories_by_category'),

]

urlpatterns += router.urls  # ✅ Important
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
