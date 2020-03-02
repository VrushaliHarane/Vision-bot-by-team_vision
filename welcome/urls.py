from django.urls import path
from . import views

urlpatterns=[
    
    path('',views.home,name='home'),
    path('send/',views.chat1,name='chat1'),
    path('chat1/',views.chat1,name='chat1'),
    path('search/',views.search,name='search'),
    path('shop/',views.shop,name='shop'),
    path('blog/',views.blog,name='blog'),
    path('contact/',views.contact,name='contact'),
    path('blog-details/',views.blogd,name='b-d'),
    path('shopping-cart/',views.shoppingcart,name='shopping-cart'),
    path('check-out/',views.checkout,name='checkout'),
    path('faq/',views.faq,name='faq'),
    path('register/',views.register,name='register'),
    path('login/',views.login,name='login'),
]
