from django.urls import path

from . import views


app_name = 'texter'
urlpatterns = [
    # ex: /texter/
    path('', views.index, name='index'),
    path('adder',views.adder, name = 'adder'),
    path('adder/add_person',views.add_person,name ='add_person'),
    path('send_screen',views.send_screen,name= 'send_screen'),
    path('send_screen/message',views.message,name='message'),
    path('deleter',views.deleter,name='deleter'),
    path('deleter/delete_person',views.delete_person,name ='delete_person')
]