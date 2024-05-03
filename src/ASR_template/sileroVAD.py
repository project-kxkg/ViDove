import torch
import logging
from silero_utils_vad import read_audio, init_jit_model, OnnxWrapper
from silero_utils_vad import get_speech_timestamps as gst


device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')

def get_speech_timestamps (source_audio, method = 'silero_onnx'):

    if method == 'silero_jit':
        return silero_jit (source_audio, 'silero_vad.jit')
    if method == 'silero_onnx' :
        return silero_onnx (source_audio, 'silero_vad.onnx')



def silero_jit (source_audio, jit_model = "silero_vad.jit"):

    audio = read_audio (source_audio, sampling_rate = 16000).to (device)
    model = init_jit_model (jit_model, device = device)

    speech_timestamps = gst (audio = audio, model = model, return_seconds = True)
    
    log.debug (f"Detected speech argument by {jit_model}:")
    for i in speech_timestamps:
        log.debug (f"{i['start']} - {i['end']}")

    return speech_timestamps


def silero_onnx (source_audio, onnx_model = "silero_vad.onnx"): 
    
    audio = read_audio (source_audio, sampling_rate = 16000).to (device)
    model = OnnxWrapper (onnx_model)

    speech_timestamps = gst (audio = audio, model = model, return_seconds = True)

    log.debug (f"Detected speech argument by {onnx_model}:")
    for i in speech_timestamps:
        log.debug (f"{i['start']} - {i['end']}")

    return speech_timestamps
    

