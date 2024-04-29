from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt

@require_POST
@csrf_exempt
def getMessage(request):
    userText = request.POST.get('text')
    return HttpResponse(userText)