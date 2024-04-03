from fastapi import FastAPI,File,UploadFile,Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, requests, base64
from io import BytesIO
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel
from langchain.chains import LLMChain
from tempfile import NamedTemporaryFile
import joblib
import warnings
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


#can remove if u want its just to remove warning
warnings.filterwarnings("ignore", message="Trying to unpickle estimator .* from version .* when using version .*")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


languages = {
    "Hindi": "hi", #hindi
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

############################RAG##############################

OPENAI_API_KEY = "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"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
new_db = FAISS.load_local("C:\\Users\\Anand\\Desktop\\Bhashini\\Bhashini-PS-2\\arif\\faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = new_db.as_retriever()

chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY
)

client = OpenAI(openai_api_key=OPENAI_API_KEY)

message = [
    SystemMessage(content="""You are a customer-service chatbot. You have to answer the responses based on the contexts that will be provided to you.'"""),
    HumanMessage(content="Hi Multilingo, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?")
]


async def augment_prompt(query,emotion):
    if len(message)>2:
        results = retriever.invoke(str(message[-2])+'\n'+query)
    else:
        results = retriever.invoke(query)
    source_knowledge = "\n".join([x.page_content for x in results])

    augemented_prompt = f"""Using the contexts below, answer the query. Also you are provided with previous conversations of the 
    system with the user. Refer to this conversation along with the emotion of user and understand what the user is asking for making you a conversational bot.

    Context:
    {source_knowledge}

    Emotion: {emotion}
    
    Query:
    {query}"""
    return augemented_prompt



async def gpt_response(query,emotion):
    prompt = HumanMessage(
        content= await augment_prompt(query, emotion)
    )
    message.append(prompt)
    res = chat.invoke(message[-4:])
    message.append(res)
    return res.content


##########################EMOTION DETECTION#########################

def predict_emotion(text):
    template = """The emotions are: [happy, neutral, unhappy]. Detect Emotion for the sentence below in just one word.\n
    Sentence: {text}"""
    prompt = PromptTemplate.from_template(template)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.invoke(text)
    return answer


##############################Translate##############################


async def translation(source_lang, target_lang, content):
    source_language = languages[source_lang]
    target_language = languages[target_lang]
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


@app.post("/gettext")
async def gettext(text: str = Form(...), language: str = Form(...)):
    try:
        translate_response = await translation(language, "English", text)
        print(translate_response)
        english_text = translate_response["translated_content"]
        print(english_text)
        detected_emotion = predict_emotion(english_text)
        print(detected_emotion)
        gpt_result = await gpt_response(english_text,detected_emotion)
        print(gpt_result)
        result_text = await translation("English", language,gpt_result)
        print(result_text)
        result = result_text["translated_content"]
        return JSONResponse(content={"text": result, "success": True}, status_code=200)
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={ "text": "Error processing text", "success": False}, status_code=500)


#####################################TRANSCRIBE###########################

async def transcribe(source_lang, content):
    source_language = languages[source_lang]
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
            transcribed_content = compute_response_data["pipelineResponse"][0]["output"][0]["source"]
            return {
                "status_code": 200,
                "message": "Translation successful",
                "transcribed_content": transcribed_content
            }
        else:
            return {
                "status_code": compute_response.status_code,
                "message": "Error in translation",
                "transcribed_content": None
            }
    else:
        return {
            "status_code": response.status_code,
            "message": "Error in translation request",
            "transcribed_content": None
        }
    

async def text_to_speech(source_lang,content):
    source_language = languages[source_lang]
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
            tts_b64 = compute_response_data["pipelineResponse"][0]["audio"][0]["audioContent"]
            return {
                "status_code": 200,
                "message": "Translation successful",
                "tts_base64": tts_b64
            }
        else:
            return {
                "status_code": compute_response.status_code,
                "message": "Error in translation",
                "tts_base64": None
            }
    else:
        return {
            "status_code": response.status_code,
            "message": "Error in translation request",
            "tts_base64": None
        }


@app.post("/getaudio")
async def getaudio(language: str = Form(...), audio: UploadFile = File(...)):
    try:
            mp3_data = await audio.read()
            base64_encoded_data = base64.b64encode(mp3_data).decode('utf-8')
            source_text = await transcribe(language,base64_encoded_data)
            text = source_text["transcribed_content"]
            print(text)
            translate_response = await translation(language,"English",text)
            english_text = translate_response["translated_content"]
            print(english_text)
            detected_emotion = predict_emotion(english_text)
            print(detected_emotion)
            gpt_result = await gpt_response(english_text,detected_emotion)
            print(gpt_result)
            result_text = await translation("English", language,gpt_result)
            result = result_text["translated_content"]
            print(result)
            speech = await text_to_speech(language,result)
            if speech["status_code"] == 200:
                tts_b64 = speech["tts_base64"]
                audio_bytes = base64.b64decode(tts_b64)
                with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

            # Return the path to the temporary file as a FileResponse
            # Set media_type explicitly if needed, though it should be inferred from the file extension
                return FileResponse(path=tmp_path, filename="output.mp3", media_type="audio/mpeg")
            else:
                return JSONResponse(content={"text": "Error generating speech", "success": False}, status_code=speech["status_code"])
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={ "text": "Error processing text", "success": False}, status_code=500)    

