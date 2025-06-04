from rest_framework import serializers
from django.contrib.auth.models import User
from .models import (
    UserProfile, Category, SubCategory, Wardrobe, Outfit, 
    OutfitPlanner, Post, Like, Follow
)


class UserProfileSerializer(serializers.ModelSerializer):
    user = serializers.SerializerMethodField()
    followers_count = serializers.SerializerMethodField()
    following_count = serializers.SerializerMethodField()
    is_following = serializers.SerializerMethodField()
    total_likes = serializers.SerializerMethodField()
    
    class Meta:
        model = UserProfile
        fields = [
            'user',  # full user object
            'gender',
            'modesty_preference',
            'profile_picture',
            'bio',
            'location',
            'followers_count',
            'following_count',
            'is_following',
            'total_likes',
        ]
    def get_total_likes(self, obj):
        # Efficient aggregate count
        from .models import Post
        return Post.objects.filter(user=obj.user).aggregate(total=serializers.models.Count('likes'))['total'] or 0
    def get_user(self, obj):
        request = self.context.get('request')
        return {
            'id': obj.user.id,
            'username': obj.user.username,
            'email': obj.user.email,
            'profile_picture': request.build_absolute_uri(obj.profile_picture.url) if (request and obj.profile_picture) else None
        }
    def get_followers_count(self, obj):
        return obj.user.followers.count()

    def get_following_count(self, obj):
        return obj.user.following.count()
    
    def get_is_following(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return Follow.objects.filter(follower=request.user, following=obj.user).exists()
        return False


        # ✅ User Serializer
class UserSerializer(serializers.ModelSerializer):
    profile = UserProfileSerializer(read_only=True, source='userprofile')

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'profile']

# ✅ Category Serializer
class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ['id', 'name']
        

# ✅ SubCategory Serializer
class SubCategorySerializer(serializers.ModelSerializer):
    category = CategorySerializer(read_only=True)
    category_id = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(), source='category', write_only=True
    )

    class Meta:
        model = SubCategory
        fields = ['id', 'name', 'category', 'category_id']
        
# ✅ Updated Category Serializer with subcategories
class CategoryWithSubSerializer(serializers.ModelSerializer):
    subcategories = SubCategorySerializer(many=True, read_only=True)

    class Meta:
        model = Category
        fields = ['id', 'name', 'subcategories']



# ✅ Wardrobe Serializer
class WardrobeSerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(read_only=True)
    category = CategorySerializer(read_only=True)
    category_id = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(), source='category', write_only=True
    )
    subcategory = SubCategorySerializer(read_only=True)
    subcategory_id = serializers.PrimaryKeyRelatedField(
        queryset=SubCategory.objects.all(), source='subcategory', write_only=True, allow_null=True, required=False
    )

    class Meta:
        model = Wardrobe
        fields = [
            'id', 'user', 'category', 'category_id', 'subcategory', 'subcategory_id',
            'color', 'size', 'material', 'season', 'tags', 'photo_path'
        ]


# ✅ Outfit Serializer
class OutfitSerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(read_only=True)
    selected_items = WardrobeSerializer(read_only=True, many=True)
    selected_items_ids = serializers.PrimaryKeyRelatedField(
        queryset=Wardrobe.objects.all(), many=True, write_only=True, source='selected_items'
    )

    class Meta:
        model = Outfit
        fields = [
            'id', 'user', 'type', 'selected_items', 'selected_items_ids',
            'is_hijab_friendly', 'description', 'photo_path' , 'season', 'tags'
        ]


# ✅ OutfitPlanner Serializer
class OutfitPlannerSerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(read_only=True)
    outfit = OutfitSerializer(read_only=True)
    outfit_id = serializers.PrimaryKeyRelatedField(
        queryset=Outfit.objects.all(), source='outfit', write_only=True
    )

    class Meta:
        model = OutfitPlanner
        fields = ['id', 'user', 'outfit', 'outfit_id', 'date']


# ✅ KEEP THIS ONE ONLY
class PostSerializer(serializers.ModelSerializer):
    user = serializers.SerializerMethodField()
    outfit = OutfitSerializer(read_only=True)
    outfit_id = serializers.PrimaryKeyRelatedField(
        queryset=Outfit.objects.all(), source='outfit', write_only=True, required=False
    )
    like_count = serializers.SerializerMethodField()
    is_liked_by_current_user = serializers.SerializerMethodField()
    is_own_post = serializers.SerializerMethodField()

    class Meta:
        model = Post
        fields = [
            'id',
            'user',  # ✅ now will use get_user()
            'outfit',
            'outfit_id',
            'image',
            'caption',
            'created_at',
            'is_own_post','like_count',
            'is_liked_by_current_user'
        ]

    def get_user(self, obj):
        request = self.context.get('request')
        profile = getattr(obj.user, 'profile', None)

        return {
            'id': obj.user.id,
            'username': obj.user.username,
            'email': obj.user.email,
            'profile_picture': (
                request.build_absolute_uri(profile.profile_picture.url)
                if request and profile and profile.profile_picture
                else None
            ),
        }
    def get_is_own_post(self, obj):
        request = self.context.get('request')
        return request.user == obj.user if request else False
    def get_like_count(self, obj):
        return obj.likes.count()

    def get_is_liked_by_current_user(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return obj.likes.filter(user=request.user).exists()
        return False

    
# ✅ Like Serializer
class LikeSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    post = PostSerializer(read_only=True)
    post_id = serializers.PrimaryKeyRelatedField(
        queryset=Post.objects.all(), source='post', write_only=True
    )

    class Meta:
        model = Like
        fields = ['id', 'user', 'post', 'post_id', 'created_at']


# ✅ Follow Serializer
class FollowSerializer(serializers.ModelSerializer):
    follower = UserSerializer(read_only=True)
    following = UserSerializer(read_only=True)
    following_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(), source='following', write_only=True
    )

    class Meta:
        model = Follow
        fields = ['id', 'follower', 'following', 'following_id', 'created_at']


