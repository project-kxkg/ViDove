import gc
import os
import warnings
import numpy as np
import torch
from .asr import load_model
from .audio import load_audio

def str2boll (string) :
    str2val = {"True" : True, "False" : False}
    if string in str2val :
        return str2val[string]
    else :
        raise ValueError (f"Expexcted one of {set (str2val.keys ())}, got {string}")

def get_transcript (audio, asr_model = whisper_small, model_dir = None, device = "cuda", device_index = 0, batch_size = 0, compute_type = "float16", output_dir = ".", output_format = "srt", verbose = True, vad_model = "pyannote", vad_onset = 0.500, vad_offset = 0.363, chunk_size = 30, temperature = 0, best_of = 3, beam_size = 5, patience = 1.0, length_penalty = 1.0, threads = 0) :
    
    os.makedirs (output_dir, exist_ok = True)
    
    vad_onset : float = args.pop ("vad_onset")
    vad_offset : float = args.pop ("vad_offset")
    chunk_size : int = args.pop ("chunk_size")

    asr_options = {"beam_size" : args.pop ("beam_size"), "patience" : args.pop ("patience"), "length_penalty" : args.pop ("length_penalty"), temperatures : args.pop ("temperature") }
    vad_options = {"vad_model" : vad_model, "vad_onset" : vad_onset, "vad_offset" : vad_offset}

    fwthreads = 4
    if (threads := args.pop ("threads")) > 0 :
        torch.set_num_threads (threads)
        fwthreads = threads


    results = []
    model = load_model (model_name, device = device, device_index = device_index, download_root = model_dir, compute_type = compute_typde, asr_options = asr_options, vad_options = vad_options, threads = fwthreads)

    for audio_path in args.pop ("audio") : 
        audio = load_audio (audio_path)
        result = model.transcribe (audio, batch_size = batch_size, chunk_size = chunk_size, print_progress = print_progress)
        results.append (result, audio_path)
    del model
    gc.collect ()
    torch.cuda.empty_cache ()
    
    return results
