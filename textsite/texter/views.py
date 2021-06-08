from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render,get_object_or_404
from django.http import Http404, HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth.decorators import login_required

from .models import Person

from .texting.text_sender import send_messages

@login_required
def index(request):
    template = loader.get_template('texter/index.html')
    context = {}
    return HttpResponse(template.render(context, request))

@login_required
def adder(request):
    template = loader.get_template('texter/adder.html')

    return HttpResponse(template.render({},request)) 

@login_required
def add_person(request):

    first = request.POST['first']
    last = request.POST['last']
    num = request.POST['number']

    new_p = Person(first_name = first, last_name = last, number = num)
    new_p.save()

    return HttpResponseRedirect(reverse('texter:index'))

@login_required
def deleter(request):
    template = loader.get_template('texter/deleter.html')

    return HttpResponse(template.render({},request))   

@login_required
def delete_person(request): 
    
    first = request.POST['first']
    last = request.POST['last']

    tb_removed = Person.objects.filter(first_name = first).filter(last_name = last)

    tb_removed.delete()

    return HttpResponseRedirect(reverse('texter:index'))

@login_required
def send_screen(request):
    person_list = Person.objects.all()
    context = {'person_list': person_list}
    template = loader.get_template('texter/send.html')
    return HttpResponse(template.render(context,request))

@login_required
def message(request):
    
    link_dict = {False:False,"Yes":True}

    name_numbers = dict(request.POST)
    name_numbers.pop('csrfmiddlewaretoken')
    text = name_numbers.pop('message')
    link = name_numbers.pop('link',[False])
    link = link_dict[link[0]]
    link_txt = name_numbers.pop('link_text',[None])
    signature = name_numbers.pop('signature')

    print(name_numbers)
    
    send_messages(name_numbers,text[0],link,link_txt[0],signature[0])

    return HttpResponseRedirect(reverse('texter:index'))