from urllib import request
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework import status
from .models import Follow, Like, Post, Wardrobe, Outfit, OutfitPlanner, UserProfile , Category, SubCategory
from .serializers import CategoryWithSubSerializer, PostSerializer, WardrobeSerializer, OutfitSerializer, OutfitPlannerSerializer, UserProfileSerializer,CategorySerializer, SubCategorySerializer
from django.contrib.auth import authenticate, login
from django.http import JsonResponse
import json
import re
from rest_framework.authtoken.models import Token
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import api_view, permission_classes, parser_classes
from .Aimodels.ai_outfit_generator import generate_outfit_from_items


def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def is_complex_password(password):
    return (
        len(password) >= 6 and
        re.search(r"[A-Z]", password) and
        re.search(r"\d", password) and
        re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)
    )

# ✅ Register a new user
@api_view(['POST'])
@permission_classes([AllowAny])
def register_user(request):
    username = request.data.get('username', '').strip().lower() 
    email = request.data.get('email', '').strip().lower()
    password = request.data.get('password')

    if not username or not email or not password:
        return Response({'error': 'All fields are required'}, status=status.HTTP_400_BAD_REQUEST)

    if not is_valid_email(email):
        return Response({'error': 'Invalid email format'}, status=status.HTTP_400_BAD_REQUEST)

    if not is_complex_password(password):
        return Response({'error': 'Password must contain uppercase, number, symbol and be at least 6 characters'}, status=status.HTTP_400_BAD_REQUEST)

    if User.objects.filter(username__iexact=username).exists():
        return Response({'error': 'Username already exists'}, status=status.HTTP_400_BAD_REQUEST)
    
    if User.objects.filter(email=email).exists():
        return Response({'error': 'Email already registered'}, status=status.HTTP_400_BAD_REQUEST)

    user = User.objects.create_user(username=username, email=email, password=password)
    UserProfile.objects.create(user=user)  # Create associated profile
    token, _ = Token.objects.get_or_create(user=user)

    return Response({'message': 'User registered successfully', 'token': token.key, 'user_id': user.id}, status=status.HTTP_201_CREATED)


# ✅ Login and return token
@api_view(['POST'])
@permission_classes([AllowAny])
def login_user(request):
    identifier = request.data.get('username')  # can be username or email
    password = request.data.get('password')

    if not identifier or not password:
        return Response({'error': 'Email/Username and password are required'}, status=status.HTTP_400_BAD_REQUEST)

    identifier = identifier.strip()

    try:
        user_obj = User.objects.get(email__iexact=identifier)
        username = user_obj.username
    except User.DoesNotExist:
        try:
            user_obj = User.objects.get(username__iexact=identifier)
            username = user_obj.username
        except User.DoesNotExist:
            return Response({'error': 'No account found with that email or username'}, status=status.HTTP_404_NOT_FOUND)

    user = authenticate(username=username, password=password)
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        return Response({'message': 'Login successful', 'token': token.key, 'user_id': user.id}, status=status.HTTP_200_OK)
    else:
        return Response({'error': 'Incorrect password'}, status=status.HTTP_401_UNAUTHORIZED)


# ✅ Get profile for logged-in user
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_profile(request):
    try:
        profile = request.user.profile
        serializer = UserProfileSerializer(profile)
        return Response(serializer.data)
    except UserProfile.DoesNotExist:
        return Response({'error': 'Profile not found'}, status=status.HTTP_404_NOT_FOUND)


# ✅ Update profile for logged-in user
@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_user_profile(request):
    try:
        profile, _ = UserProfile.objects.get_or_create(user=request.user)

        profile.bio = request.data.get('bio', profile.bio)
        profile.location = request.data.get('location', profile.location)
        profile.gender = request.data.get('gender', profile.gender)
        profile.modesty_preference = request.data.get('modesty_preference', profile.modesty_preference)
        # ✅ Handle profile picture if included
        if 'profile_picture' in request.FILES:
            profile.profile_picture = request.FILES['profile_picture']

        profile.save()
        serializer = UserProfileSerializer(profile)
        return Response(serializer.data)
    except Exception as e:
        print("Error:", str(e))
        return Response({'error': 'Something went wrong'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Category.objects.all()
    serializer_class = CategoryWithSubSerializer  # Updated here

class SubCategoryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = SubCategory.objects.all()
    serializer_class = SubCategorySerializer

class SubCategoryByCategoryView(APIView):
    def get(self, request, category_id):
        subcategories = SubCategory.objects.filter(category_id=category_id)
        serializer = SubCategorySerializer(subcategories, many=True)
        return Response(serializer.data)
    
# ✅ Upload Clothing Item (User can add clothes)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_clothing(request):
    print("DATA:", request.data)
    print("FILES:", request.FILES)
    serializer = WardrobeSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(user=request.user)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    print("ERRORS:", serializer.errors)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# ✅ Get All Wardrobe Items (User's Clothes)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_wardrobe(request):
    print(f"🔍 Authenticated user: {request.user} (ID: {request.user.id})")
    subcategory_id = request.query_params.get('subcategory_id')
    category_id = request.query_params.get('category_id')

    items = Wardrobe.objects.filter(user=request.user)

    if subcategory_id:
        items = items.filter(subcategory_id=subcategory_id)
    elif category_id:
        items = items.filter(category_id=category_id)
    print(f"🧥 Found {items.count()} items for user {request.user.username}")
    for item in items:
        print(f"🧢 Item: {item.id}, Category: {item.category}, Photo: {item.photo_path}")
    serializer = WardrobeSerializer(items, many=True)
    return Response(serializer.data)

# ✅ Update Clothing Item
@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_clothing(request, item_id):
    """Allows users to edit details of a wardrobe item"""
    try:
        item = Wardrobe.objects.get(id=item_id, user=request.user)
        serializer = WardrobeSerializer(instance=item, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    except Wardrobe.DoesNotExist:
        return Response({'error': 'Clothing item not found'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_clothing(request, item_id):
    item = get_object_or_404(Wardrobe, id=item_id, user=request.user)
    item.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)

# ✅ Create an Outfit (User Selects Clothes)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def create_outfit(request):
    """Allows users to create outfits manually by selecting items from their wardrobe."""
    serializer = OutfitSerializer(data=request.data, context={'request': request})

    if serializer.is_valid():
        outfit = serializer.save(user=request.user)  # Automatically associate with logged-in user
        return Response(OutfitSerializer(outfit).data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# ✅ Get All Outfits (User's Saved Outfits)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_outfits(request):
    """Retrieves all outfits for the logged-in user"""
    outfits = Outfit.objects.filter(user=request.user)
    serializer = OutfitSerializer(outfits, many=True)
    return Response(serializer.data)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_outfit(request, pk):
    try:
        outfit = Outfit.objects.get(pk=pk, user=request.user)
        outfit.delete()
        return Response(status=204)
    except Outfit.DoesNotExist:
        return Response({'error': 'Outfit not found'}, status=404)

# ✅ AI Generates an Outfit (Basic API Placeholder)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def ai_generate_outfit(request):
    """Allows AI to generate a full outfit (Placeholder API)"""
    # TODO: Integrate AI model (TensorFlow, OpenCV, or pre-trained API)
    return Response({'message': 'AI outfit generation coming soon!'}, status=status.HTTP_501_NOT_IMPLEMENTED)

# ✅ Plan an Outfit for a Date
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def plan_outfit(request):
    """Assigns an outfit to a calendar date"""
    serializer = OutfitPlannerSerializer(data=request.data)
    if serializer.is_valid():
        plan = serializer.save(user=request.user)
        return Response(OutfitPlannerSerializer(plan).data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# ✅ Get All Planned Outfits (User's Calendar)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_planned_outfits(request):
    """Retrieves all planned outfits for the logged-in user"""
    plans = OutfitPlanner.objects.filter(user=request.user)
    serializer = OutfitPlannerSerializer(plans, many=True)
    return Response(serializer.data)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_planned_outfit(request, plan_id):
    try:
        plan = OutfitPlanner.objects.get(id=plan_id, user=request.user)
        plan.delete()
        return Response({'message': 'Planned outfit deleted successfully'}, status=200)
    except OutfitPlanner.DoesNotExist:
        return Response({'error': 'Planned outfit not found'}, status=404)

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_planned_outfit(request, plan_id):
    try:
        plan = OutfitPlanner.objects.get(id=plan_id, user=request.user)
        serializer = OutfitPlannerSerializer(plan, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    except OutfitPlanner.DoesNotExist:
        return Response({'error': 'Planned outfit not found'}, status=404)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])  # Optional, but helpful if you still allow image upload
def create_post(request):
    """Create a new post with an outfit and optionally an image"""
    outfit_id = request.data.get('outfit_id')
    caption = request.data.get('caption', '')

    try:
        outfit = Outfit.objects.get(id=outfit_id, user=request.user)

        # If image was not uploaded, use the outfit photo
        image = request.FILES.get('image')
        if image is None:
            image = outfit.photo_path

        post = Post.objects.create(
            user=request.user,
            outfit=outfit,
            caption=caption,
            image=image  # Assign outfit's image if image was None
        )

        return Response(PostSerializer(post).data, status=status.HTTP_201_CREATED)

    except Outfit.DoesNotExist:
        return Response({'error': 'Outfit not found or not owned by user'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
def get_all_posts(request):
    """Retrieve all posts from all users (public feed)"""
    posts = Post.objects.all().order_by('-created_at')
    serializer = PostSerializer(posts, many=True)
    return Response(serializer.data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def toggle_like_post(request, post_id):
    """Like or unlike a post"""
    try:
        post = Post.objects.get(id=post_id)
        like, created = Like.objects.get_or_create(user=request.user, post=post)
        if not created:
            like.delete()
            return Response({'message': 'Unliked post'}, status=status.HTTP_200_OK)
        return Response({'message': 'Liked post'}, status=status.HTTP_201_CREATED)
    except Post.DoesNotExist:
        return Response({'error': 'Post not found'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def toggle_follow(request, user_id):
    """Follow or unfollow a user"""
    try:
        to_follow = User.objects.get(id=user_id)
        if request.user == to_follow:
            return Response({'error': 'Cannot follow yourself'}, status=status.HTTP_400_BAD_REQUEST)

        follow, created = Follow.objects.get_or_create(follower=request.user, following=to_follow)
        if not created:
            follow.delete()
            return Response({'message': 'Unfollowed user'}, status=status.HTTP_200_OK)
        return Response({'message': 'Followed user'}, status=status.HTTP_201_CREATED)
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_following_feed(request):
    """Get posts only from followed users"""
    following_users = Follow.objects.filter(follower=request.user).values_list('following_id', flat=True)
    posts = Post.objects.filter(user_id__in=following_users).order_by('-created_at')
    serializer = PostSerializer(posts, many=True)
    return Response(serializer.data)
# ✨ NEW: Get Wardrobe Items by SubCategory ✨
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_wardrobe_by_subcategory(request, subcategory_id):
    """Retrieves wardrobe items for the logged-in user filtered by subcategory"""
    try:
        # Check if subcategory exists (optional, but good practice)
        subcategory = SubCategory.objects.get(id=subcategory_id)
        items = Wardrobe.objects.filter(user=request.user, subcategory_id=subcategory_id)
        serializer = WardrobeSerializer(items, many=True)
        return Response(serializer.data)
    except SubCategory.DoesNotExist:
        return Response({"error": "SubCategory not found"}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    # ✨ NEW: Get SubCategories by Category ✨
@api_view(["GET"])
@permission_classes([IsAuthenticated]) # Or AllowAny if subcategories are public
def get_subcategories_by_category(request, category_id):
    """Retrieves all subcategories belonging to a specific category"""
    try:
        # Ensure the category exists
        category = Category.objects.get(id=category_id)
        subcategories = SubCategory.objects.filter(category_id=category_id)
        serializer = SubCategorySerializer(subcategories, many=True)
        return Response(serializer.data)
    except Category.DoesNotExist:
        return Response({"error": "Category not found"}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        print(f"Error in get_subcategories_by_category: {e}")
        return Response({"error": "An internal error occurred"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_all_posts(request):
    posts = Post.objects.all().order_by('-created_at')
    response_data = []
    for post in posts:
        serialized_post = PostSerializer(post).data
        serialized_post['is_liked_by_current_user'] = Like.objects.filter(post=post, user=request.user).exists()
        serialized_post['like_count'] = post.likes.count()
        serialized_post['user'] = {
            'id': post.user.id,
            'username': post.user.username,
        }
        response_data.append(serialized_post)

    return Response(response_data, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_outfit_by_id(request, pk):
    try:
        outfit = Outfit.objects.get(pk=pk)
        serializer = OutfitSerializer(outfit)
        return Response(serializer.data)
    except Outfit.DoesNotExist:
        return Response({'error': 'Outfit not found'}, status=404)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_post_likes(request, post_id):
    try:
        post = Post.objects.get(id=post_id)
    except Post.DoesNotExist:
        return Response({"error": "Post not found"}, status=404)

    liked_users = post.likes.all().select_related('user__profile')
    data = [
        {
            "id": like.user.id,
            "username": like.user.username,
            "profile_picture": (
                request.build_absolute_uri(like.user.profile.profile_picture.url)
                if like.user.profile.profile_picture else None
            )
        }
        for like in liked_users
    ]
    return Response(data)
