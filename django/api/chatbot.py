from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt
from api.huggingface.huggingface import Huggingface


@require_POST
@csrf_exempt
# meta-llama/Meta-Llama-3-8B"
def getMessage(request):
    try:
        userText = request.POST.get('text')
        # return userText
        llmClassObject = Huggingface('mistralai/Mistral-7B-Instruct-v0.2')
        llmResult = Huggingface.getResultOfLlm(llmClassObject, userText)
        llmResult = llmResult[0]['generated_text']
        return HttpResponse(llmResult)
    except Exception as e:
        return e
