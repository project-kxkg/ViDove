import os
import warnings
from typing import List, Union, Optional, NamedTuple, TypedDict

import ctranslate2
import faster_whisper
import numpy as np
import torch
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from .audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_sptg
from .vad import load_vad_model, merge_chunks
from .sileroVAD import silero_jit, silero_onnx

class SingleSeg (TypedDict) :
    start :float
    end : float
    text : str

class TransResult (TypedDict) :
    segs : List[SingleSeg]
    lang : str

def find_numeral_symbol_tokens (tokenizer) :
    tokens = []
    for i in range (tokenizer.eot) :
        token = tokenizer.decode ([i]).removeprefix (" ")
        has_numeral_symbol = any (c in "0123456789%$£" for c in token)
        if has_numeral_symbol :
            tokens.append (i)
    return tokens

class WhisperModel (faster_whisper.WhisperModel) :
    def gen_seg_batched (self, features : np.array, tokenizer : faster_whisper.tokenizer.Tokenizer, options : faster_whisper.transcribe.TranscriptionOptions, encoder_output = None) :
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None :
            initial_prompt = " " + options.initial_prompt.strip ()
            initial_prompt_tokens = tokenizer.encode (initial_prompt)
            all_tokens,extend (initial_prompt_tokens)
        prev_tokens = all_tokens[prompt_reset_since : ]
        prompt = self.get_prompt (tokenizer, prev_tokens, without_timestamps = options.without_timestamps, prefix = options.prefix, )
        encoder_output = self.encode (features)
        max_initial_timestamp_index = int (round (options.max_initial_timestamp / self.time_precision))
        result = self.model.generate (encoder_output, [prompt] * batch_size, beam_size = options.beam_size, patience = options.patience, length_penalty = options.length.penalty, max_length = self.max_length, suppress_blank = options.suppress_blank, suppress_tokens = options.suppress_tokens, )
        tokens_batch = [x.sequences_ids[0] for x in result]
        def decode batch (tokens : List[List[int]]) -> str :
            res = []
            for tk in tokens :
                res.append ([token for token in tk if token < tokenizer.eot])
            return tokenizer.tokenizer.decode_batch (res)
        return text
    def encode (self, features : np.ndarray) -> ctranslate2.StorageView :
        to_cpu = self.model.device == "cuda" and len (self.model.device_index) > 1
        if len (features.shape) == 2 :
            features = np.expand_dims (features, 0)
        features = faster_whisper.transcribe.get_ctranslate2_storage (features)

        return self.model.encode (features, to_cpu = to_cpu)

class FasterWhisperPipeline (Pipeline) :
    def __init__ (self, model, vad, vad_params : dict, options : NamedTuple, tokenizer  = None, device : Union[int, str, "torch.device"] = -1, framework = "pt", language : Optional[str] = None, suppress_numerals : bool = False, **kwargs) :
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop ("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._santize_params (**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt" :
            if isinstance (device, torch.device) :
                self.device = device
            elif isinstance (device, str) :
                self.device = torch.device (device)
            elif device < 0 :
                self.device = torch.device ("cpu")
            else :
                self.device = torch.device (f"cuda:{device}")
        else :
            self.device = device
        super (Pipeline, self).__init__ ()
        #self.vad_model = vad
        self._vad_params = vad_params

    def _santize_params (self, **kwargs) :
        preprocess_kwargs = {}
        if "tokenizer" in kwargs :
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}
    def preprocess (self, audio) :
        audio = audio['inputs']
        model_n_mels = self.model.feat_kwargs.get ("feature_size")
        features = log_mel_sptg (audio, n_mels = model_n_mels if model_n_mels is not None else 80, padding = N_SAMPLES - audio.shape[0], )
        return {'inputs' : features}
    def _forward (self, model_inputs) :
        outputs = self.model.gen_seg_batched (model_inputs['inputs'], self.tokenizer, self.options)
        return {'text' : outputs}
    def postprocess (self, model_outputs) :
        return model_outputs

    def get_iter (self, inputs, num_workers : int, batch_size : int, preprocess_params, forward_params, postprocess_params) :
        dataset = PipelineIterator (inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ :
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        def stack (items) :
            return {'inputs' : torch.stack ([x['inputs'] for x in items])}
        dataloader = torch.utils.data.DataLoader (dataset, num_workers = num_workers, batch_size = batch_size, collate_fn = stack)
        model_iter = PipelineIterator (dataloader, self.forward, forward_params, loader_batch_size = batch_size)
        final_iter = PipelineIterator (model_iter, self.postprocess, postprocess_params)
        return final_iter
    def transcribe (self, audio : Union[str, np.ndarray], batch_size = None, num_workers = 0, language = None, task = None, chunk_size = 30, print_progress = False, combined_progress = False) -> TranscriptionResult :
        if isinstance (audio, str) :
            audio = load_audio (audio)
        def data (audio, segs) :
            for seg in segs :
                f1 = int (seg['start' * SAMPLE_RATE])
                f2 = int (seg['end'] * SAMPLE_RATE)
                yield {'inputs' : audio[f1 : f2]}

        #vad_segs = self.vad_model ({"waveform" : torch.from_numpy (audio).unsqueeze (0), "sample_rate" : SAMPLE_RATE})
        if vad == "pyannote" :
            self.vad_model = load_vad_model (torch.device (device), use_auth_token = None, **vad_params)
            vad_segs = self.vad_model ({"waveform" : torch.from_numpy (audio).unsqueeze (0), "sample_rate" : SAMPLE_RATE})
        else if vad == "silero_jit" :
            vad_segs = silero_jit (audio = audio)
        else if vad == "silero_onnx" :
            vad_segs = silero_onnx (audio = audio)

        vad_segs = merge_chunks (vad_segs, chunk_size, onset = self._vad_params["vad_onset"], offset = self._vad_params["vad_offset"], )
        if self.tokenizer is None :
            language = language or self.detect_language (audio)
            task = task or "transcribe"
            self.tokenizer = faster_whisper.tokenizer.Tokenizer (self.model.hf_tokenizer, self.model.model.is_multilingual, task = task, language = language)
        else :
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code :
                self.tokenizer = faster_whisper.tokenizer.Tokenizer (self.model.hf_tokenizer, self.model.model.is_nultilingual, task = task, language = language)
        if self.suppress_numerals :
            prev_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens (self.tokenizer)
            print (f"Suppressing numeral and symbol tokens")
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list (set (new_suppressed_tokens))
            self.options = self.options._replace (suppress_tokens = new_suppressed_tokens)

        segs : List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segs = len (vad_segs)
        for idx, out in enumerate (self.__call__ (data (audio, vad_segs), batch_size = batch_size, num_workers = num_workers)) :
            if print_progress :
                base_progress = ((idx + 1) / total_segs) * 100
                percent_complete = base_progress / 2 if comnined_progress else base_progress
                print (f"Progress : {percent_complete:.2f}%...")
            text = out['text']
            if batch_size in [0, 1, None] :
                text = text[0]
            segs.append ({"text" : text, "start" : round (vad_segs[idx]['start'], 3), "end" : round (vad_segs[idx]['end'], 3)})
        if self.preset_language is None :
            self.tokenizer = None
        if self.suppress_numerals :
            self.options = self.options._replace (suppress_tokens = prev_suppress_tokens)

        return {"segments" : segs, "language" : language}

    def detect_language (self, audio : np.ndarray) :
        model_n_mels = self.model.feat_kwargs.get ("feature_size")
        seg = log_mel-sptg (audio[ : N_SAMPLES], n_mels = model_n_mels if mpdel_n_mels is not None else 80, padding = 0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
        encoder_output = self.model.encode(seg)
        results = self.model.model.detect_language (encoder_output)
        languate_token, language_probability = results[0][0]
        language = language_token[2 : -2]
        print (f"Detected language : {language} ({language_probability : .2f}) in first 30s of audio...")
        return language

def load_model (whisper_arch, device, device_index = 0, compute_type = "folat16", asr_options = None, language : Optional[str] = None, vadmodel = "pyannote", vad_options = None, model : Optional[WhisperModel] = None, task = "transcribe", download_root = None, threads = 4) :
    if language is not None :
        tokenizer = faster_whisper.tokenizer.Tokenizer (model.hf_tokenizer, model.model.is_nultilingual, task = task, language = language)
    else :
        tokenizer = None

    default_asr_options = {"beam_size" : 5, "best_of" : 5, "patience" : 1, "length_penalty" : 1, "repetition_penalty" : 1, "no_repeat_ngram_size" = 0, "temperatures" : [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "compression_ratio_threshold" : 2.4, "log_prob_threshold" : -1.0, "no_speech_threshold" : 0.6, "condition_on_previous_text" : False, "prompt_reset_on_temperature" : 0.5, "initial_prompt" : None, "prefix" : None, "suppress_blank" : True, "suppress_tokens" : [-1], "without_timestamps" : True, "max_initial_timestamp" : 0.0, "word_timestamps" : False, "prepend_punctuations" : "\"'“¿([{-", "append_punctuations" : "\"'.。,，!！?？:：”)]}、", "suppress_numerals" : False, "max_new_tokens" : None, "clip_timestamps" : None, "hallucination_silence_threshold": None, }
    if asr_options is not None :
        default_asr_options.update (asr_options)
    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]
    default_asr_options = faster_whisper.transcribe.TranscriptionOptions (**default_asr_options)
    default_vad_options = {"vad_onset" : 0.500, "vad_offset" : 0.363}

    return FasterWhisperPipeline (model = model, vad = vadmodel, options = default_asr_options, tokenizer = tokenizer, language = language, suppress_numerals = suppress_numerals, vad_params = default_vad_options, )
