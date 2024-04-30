from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt
from api.huggingface.huggingface import Huggingface

@require_POST
@csrf_exempt
def getMessage(request):
    userText = request.POST.get('text')

    llmClassObject = Huggingface('mistralai/Mistral-7B-Instruct-v0.2')
    llmResult = Huggingface.getResultOfLlm(llmClassObject, {"inputs": userText})
    llmResult = llmResult[0]['generated_text']
    return HttpResponse(llmResult)