from fastapi import FastAPI,File,UploadFile,Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, requests, base64
from io import BytesIO
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel
import joblib
import warnings
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

chat_history=[]

async def gpt_response(query,emotion):
    global chat_history
    chat_history_string = "\n".join([f"User: {entry['query']}\nBot: {entry['response']}" for entry in chat_history])
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    _inputs = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        | StrOutputParser(),
    )
    _context = {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": lambda x: x["standalone_question"],
    }
    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    answer = conversational_qa_chain.invoke(
        {
            "question": query,
            "emotion": emotion,            
            "chat_history": chat_history_string
        }
    )
    new_entry = {"query":query, "response": answer.content}
    chat_history.append(new_entry)
    return answer.content



##########################EMOTION DETECTION#########################

def predict_emotion(text):
    pipe_lr = joblib.load(open("./emotion_classifier_pipe_lr.pkl", "rb"))
    # Make prediction
    prediction = pipe_lr.predict([text])[0]
    return prediction


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
            translate_response = await translation(language,"English",text)
            english_text = translate_response["translated_content"]
            detected_emotion = predict_emotion(english_text)
            gpt_result = await gpt_response(english_text,detected_emotion)
            result_text = await translation("English", language,gpt_result)
            result = result_text["translated_content"]
            speech = await text_to_speech(language,result)
            if speech["status_code"] == 200:
                tts_b64 = speech["tts_base64"]
                audio_bytes = base64.b64decode(tts_b64)
                audio_io = BytesIO(audio_bytes)
                audio_io.seek(0)
                return StreamingResponse(audio_io, media_type="audio/mpeg")
            else:
                return JSONResponse(content={"text": "Error generating speech", "success": False}, status_code=speech["status_code"])
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={ "text": "Error processing text", "success": False}, status_code=500)    

