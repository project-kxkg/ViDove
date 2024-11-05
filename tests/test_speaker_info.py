from pathlib import Path
import librosa
import logging
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

def get_speaker_info(method:str, local_dir:str, audio_path:str, task_logger:logging.Logger):
    # Speaker Diarization

    # speaker_info_path = Path.joinpath(local_dir)
    if audio_path is not None:
        # task_logger.info("extract speaker info from audio")
        # task_logger.info(f"Speaker Diarization method: {method}")

        if method == "Qwen/Qwen2-Audio-7B-Instruct":
            speaker_info = get_speaker_info_qwen2_audio(audio_path)
        else:
            speaker_info = None
        
        # TODO: save the speaker_info locally

    return speaker_info

def get_speaker_info_qwen2_audio(audio_path:str):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
    # model.to(device)

    conversation = [
        {'role': 'system', 'content': ''}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path}, 
            {"type": "text", "text": "How many speakers are there? Answer with an integer number."}
        ]}, 
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []

    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(
                            ele["audio_url"], 
                            sr=processor.feature_extractor.sampling_rate)[0]
                    )
    # audio = librosa.load(audio_path, 
    #                      sr=processor.feature_extractor.sampling_rate)[0]
    
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda")

    generate_ids = model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    audio_path = Path("./local_dump").joinpath("task_8b842160-bafe-4778-a5a0-68d41543a0d3", "task_8b842160-bafe-4778-a5a0-68d41543a0d3.mp3")
    speaker_info = get_speaker_info("Qwen/Qwen2-Audio-7B-Instruct", "", audio_path, None)
    print(speaker_info)
