from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class TranscriptionRequest(BaseModel):
    source_lang: int

class TranslationRequest(BaseModel):
    source_language: int
    content: str
    target_language: int    

class TtsRequest(BaseModel):
    source_language: int
    content: str

languages = {
    "hindi": "hi", #hindi
    "Gom": "gom", #Gom
    "Kannade": "kn", #Kannada
    "Dogri": "doi", #Dogri    
    "Bodo": "brx", #Bodo 
    "Urdu": "ur",  #Urdu
    "Tamil": "ta",  #Tamil
    "Kashmiri": "ks",  #Kashmiri
    "Assamese": "as",  #Assamese
    "Bengali": "bn", #Bengali
    "Marathi": "mr", #Marathi
    "Sindhi": "sd", #Sindhi
    "Maihtili": "mai",#Maithili
    "Punjabi": "pa", #Punjabi
    "Malayalam": "ml", #Malayalam
    "Manipuri": "mni",#Manipuri
    "Telugu": "te", #Telugu
    "Sanskrit": "sa", #Sanskrit
    "Nepali": "ne", #Nepali
    "Santali": "sat",#Santali
    "Gujarati": "gu", #Gujarati
    "Oriya": "or", #Oriya
    "English": "en",#English
}

@app.post('/translate', response_model=dict)
async def translate(request: TranslationRequest):
    source_language = languages[request.source_language]
    content = request.content
    target_language = languages[request.target_language] 

    payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language
                    }
                }
            }
        ],
        "pipelineRequestConfig": {
            "pipelineId" : "64392f96daac500b55c543cd"
        }
    }

    headers = {
        "Content-Type": "application/json",
        "userID": "e832f2d25d21443e8bb90515f1079041",
        "ulcaApiKey": "39e27ce432-f79c-46f8-9c8c-c0856007cb4b"
    }

    response = requests.post('https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline', json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        service_id = response_data["pipelineResponseConfig"][0]["config"][0]["serviceId"]

        compute_payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": source_language,
                            "targetLanguage": target_language
                        },
                        "serviceId": service_id
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": content
                    }
                ],
                "audio": [
                    {
                        "audioContent": None
                    }
                ]
            }
        }

        callback_url = response_data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
        
        headers2 = {
            "Content-Type": "application/json",
            response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]:
                response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
        }

        compute_response = requests.post(callback_url, json=compute_payload, headers=headers2)

        if compute_response.status_code == 200:
            compute_response_data = compute_response.json()
            translated_content = compute_response_data["pipelineResponse"][0]["output"][0]["target"]
            return {
                "status_code": 200,
                "message": "Translation successful",
                "translated_content": translated_content
            }
        else:
            return {
                "status_code": compute_response.status_code,
                "message": "Error in translation",
                "translated_content": None
            }
    else:
        return {
            "status_code": response.status_code,
            "message": "Error in translation request",
            "translated_content": None
        }

@app.post('/transcribe', response_model=dict)
async def transcribe(request: TranscriptionRequest):
    source_language = languages[request.source_lang]
    ans = {
        "audio": {
        "mime": "audio/mpeg",
        "data": """
        A
        """
        }
    }
    content = ans["audio"]["data"]
    
    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {
                        "sourceLanguage": source_language
                    }
                }
            }
        ],
        "pipelineRequestConfig": {
            "pipelineId" : "64392f96daac500b55c543cd"
        }
    }
    headers = {
        "Content-Type": "application/json",
        "userID": "4b1666d332054624a5a171d109d0cf3d",
        "ulcaApiKey": "36a0e06739-84b7-4359-88ec-6712c7979674"
    }
    response = requests.post('https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline', json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        service_id = response_data["pipelineResponseConfig"][0]["config"][0]["serviceId"]

        compute_payload = {
            "pipelineTasks": [
                {
                    "taskType": "asr",
                    "config": {
                        "language": {
                            "sourceLanguage": source_language,
                        },
                        "serviceId": service_id
                    }
                }
            ],
            "inputData": {
                "audio": [
                    {
                        "audioContent": content
                    }
                ]
            }
        }

        callback_url = response_data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
        
        headers2 = {
            response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]:
                response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
        }

        compute_response = requests.post(callback_url, json=compute_payload, headers=headers2)

        if compute_response.status_code == 200:
            compute_response_data = compute_response.json()
            translated_content = compute_response_data["pipelineResponse"][0]["output"][0]["source"]
            return {
                "status_code": 200,
                "message": "Translation successful",
                "translated_content": translated_content
            }
        else:
            return {
                "status_code": compute_response.status_code,
                "message": "Error in translation",
                "translated_content": None
            }
    else:
        return {
            "status_code": response.status_code,
            "message": "Error in translation request",
            "translated_content": None
        }
    

@app.post('/tts', response_model=dict)
async def tts(request: TtsRequest):
    source_language = languages[request.source_language]
    content = request.content

    payload = {
            "pipelineTasks": [
                {
                    "taskType": "tts",
                    "config": {
                        "language": {
                            "sourceLanguage": source_language,
                        }
                    }
                }
            ],
            "pipelineRequestConfig": {
                "pipelineId" : "64392f96daac500b55c543cd"
            }
        }

    headers = {
        "Content-Type": "application/json",
        "userID": "e832f2d25d21443e8bb90515f1079041",
        "ulcaApiKey": "39e27ce432-f79c-46f8-9c8c-c0856007cb4b"
    }

    response = requests.post('https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline', json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        service_id = response_data["pipelineResponseConfig"][0]["config"][0]["serviceId"]

        compute_payload={
            "pipelineTasks": [       
                {
                    "taskType": "tts",
                    "config": {
                        "language": {
                            "sourceLanguage": source_language
                        },
                        "serviceId": service_id,
                        "gender": "female"
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": content
                    }
                ],
                "audio": [
                    {
                        "audioContent": None
                    }
                ]
            }
        }

        callback_url = response_data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
        
        headers2 = {
            "Content-Type": "application/json",
            response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]:
                response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
        }

        compute_response = requests.post(callback_url, json=compute_payload, headers=headers2)

        if compute_response.status_code == 200:
            compute_response_data = compute_response.json()
            translated_content = compute_response_data["pipelineResponse"][0]["audio"][0]["audioContent"]
            return {
                "status_code": 200,
                "message": "Translation successful",
                "translated_content": translated_content
            }
        else:
            return {
                "status_code": compute_response.status_code,
                "message": "Error in translation",
                "translated_content": None
            }
    else:
        return {
            "status_code": response.status_code,
            "message": "Error in translation request",
            "translated_content": None
        }