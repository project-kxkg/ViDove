from openai import OpenAI
from pathlib import Path
from PIL import Image
import cv2
from collections import Counter
from pathlib import Path

import clip
import logging
import torch
import stable_whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def get_transcript(method, src_srt_path, source_lang, audio_path, client, task_logger, pre_load_asr_model = None):

    is_trans = False # is trans flag 

    if not Path.exists(src_srt_path):
        # extract script from audio
        task_logger.info("extract script from audio")
        task_logger.info(f"Module 1: ASR inference method: {method}")
        init_prompt = "Hello, welcome to my lecture." if source_lang == "EN" else ""

        # process the audio by method
        if method == "whisper-api": 
            transcript = get_transcript_whisper_api(audio_path, source_lang, init_prompt, client)
            is_trans = True
        elif method == "whisper-large-v3":
            transcript = get_transcript_whisper_large_v3(audio_path, pre_load_asr_model)
            is_trans = True
        elif method == "whisper_clips":
            # File path
            file_path = "/content/categories_places365.txt"
            category_list = []
            with open(file_path, "r") as file:
                for line in file:
                    category_path = line.strip().split(" ")[0][3:]
                    categories = category_path.split("/")
                    category_list=category_list+categories
            modified_category_list_2=[]
            for words in category_list:
                categories=words.replace("_"," ")
                modified_category_list_2.append(categories)
                category_database= list(set(modified_category_list_2))
        
        elif "stable" in method:
            whisper_model = method.split("-")[2]
            transcript = get_transcript_stable(audio_path, whisper_model, init_prompt, pre_load_asr_model)
            is_trans = True
        else:
            raise RuntimeError(f"unavaliable ASR inference method: {method}")   
    
    # return transcript or None
    if (is_trans == True):
        return transcript    
    else: 
        return None
        
def get_transcript_whisper_api(audio_path, source_lang, init_prompt, client):
    with open(audio_path, 'rb') as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="srt", language=source_lang.lower(), prompt=init_prompt)
    return transcript
    
def get_transcript_stable(audio_path, whisper_model, init_prompt, pre_load_asr_model = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pre_load_asr_model is not None:
        model = pre_load_asr_model
    else:
        model = stable_whisper.load_model(whisper_model, device)
    transcript = model.transcribe(str(audio_path), regroup=False, initial_prompt=init_prompt)
    (
        transcript
        .split_by_punctuation(['.', '。', '?'])
        .merge_by_gap(.15, max_words=3)
        .merge_by_punctuation([' '])
        .split_by_punctuation(['.', '。', '?'])
    )
    transcript = transcript.to_dict()
    transcript = transcript['segments']
    # after get the transcript, release the gpu resource
    torch.cuda.empty_cache()

    return transcript

def get_transcript_whisper_large_v3(audio_path, pre_load_asr_model = None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"

    if pre_load_asr_model is not None:
        model = pre_load_asr_model
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype)
    
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
    )

    transcript_whisper_v3 = pipe(str(audio_path))

    # convert format
    transcript = []
    for i in transcript_whisper_v3['chunks']:
        transcript.append({'start': i['timestamp'][0], 'end': i['timestamp'][1], 'text':i['text']})

    return transcript
