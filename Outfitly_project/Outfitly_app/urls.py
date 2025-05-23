from django.urls import path
from .views import (
    SubCategoryByCategoryView, register_user, login_user, get_user_profile, update_user_profile,
    upload_clothing, get_wardrobe, update_clothing,
    create_outfit, get_outfits,delete_outfit, delete_clothing, ai_generate_outfit,
    plan_outfit, get_planned_outfits,
    create_post, get_all_posts, toggle_like_post, toggle_follow, get_following_feed
)
from rest_framework.routers import DefaultRouter
from .views import CategoryViewSet, SubCategoryViewSet

router = DefaultRouter()
router.register(r'categories', CategoryViewSet, basename='category')
router.register(r'subcategories', SubCategoryViewSet, basename='subcategory')
urlpatterns = [
    path('register/', register_user, name='register_user'),
    path('login/', login_user, name='login_user'),
    path('profile/', get_user_profile, name='get_user_profile'),
    path('profile/update/', update_user_profile, name='update_user_profile'),

    # Wardrobe APIs
    path('wardrobe/upload/', upload_clothing, name='upload_clothing'),
    path('wardrobe/', get_wardrobe, name='get_wardrobe'),
    path('wardrobe/update/<int:item_id>/', update_clothing, name='update_clothing'),
    path('wardrobe/<int:item_id>/', delete_clothing, name='delete_clothing'),

    # Outfit APIs
    path('outfits/create/', create_outfit, name='create_outfit'),
    path('outfits/', get_outfits, name='get_outfits'),
    path('outfits/ai-generate/', ai_generate_outfit, name='ai_generate_outfit'),
    path('outfits/<int:pk>/', delete_outfit, name='delete_outfit'),

    # Outfit Planner APIs
    path('planner/', get_planned_outfits, name='get_planned_outfits'),
    path('planner/plan/', plan_outfit, name='plan_outfit'),

    # Feed APIs
    path('feed/posts/create/', create_post, name='create_post'),
    path('feed/posts/', get_all_posts, name='get_all_posts'),
    path('feed/posts/<int:post_id>/like/', toggle_like_post, name='toggle_like_post'),
    path('feed/follow/<int:user_id>/', toggle_follow, name='toggle_follow'),
    path('feed/following/', get_following_feed, name='get_following_feed'),
    path('categories/<int:category_id>/subcategories/', SubCategoryByCategoryView.as_view(), name='subcategories_by_category'),

]

urlpatterns += router.urls  # ✅ Important
