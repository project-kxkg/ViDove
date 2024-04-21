import hashlib
import os
import urllib
from typing import Callable, Optional, Text, Union

import numpy as np
import pandas as pd
import torch
from pyannote.audio import Model
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.core import Annotation, Segment, SlidingWindowFeature
from tqdm import tqdm

VAD_SEG_URL = "https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"

class SegX :
    def __init__ (self, start, end, speaker = None) :
        self.start = start
        self.end = end
        self.speaker = speaker

def load_vad_model (device, vad_onset = 0.500, vad_offset = 0.363, use_auth_token = None, model_fp = None) :
    model_dir = torch.hub._get_torch_home ()
    os.makedirs (model_dir, exist_ok = True)
    if model_fp is None :
        model_fp = os.path.join (model_dir, "whisperx-vad-segmentation.bin")
    if os.path.exists (model_fp) and not os.path.isfile (model_fp) :
        raise RuntimeError (f"{model_fp} exists and is not a regular file.")

    if not os.path.isfile (model_fp) :
        with urllib.request.urlopen (VAD_SEG_URL) as source, open (model_fp, "wb") as output :
            with tqdm (total = int (source.info ().get ("Content-Length")), ncols = 80, unit = "iB", unit_scale = True, unit_divisor = 1024,) as loop :
                while True :
                    buffer = source.read (8192)
                    if not buffer :
                        break

                    output.write (buffer)
                    loop.update (len (buffer))

    model_bytes = open (model_fp, "rb").read ()
    if hashlib.sha256 (model_bytes).hexdigest () != VAD_SEG_URL.split ('/') [-2] :
        raise RuntimeError ("Model has been downloaded, but the SHA256 checksum does not match. Please retry loading the model.")

    vad_model = Model.from_pretrained (model_fp, use_auth_token = use_auth_token)
    hp = {"onset" : vad_onset, "offset" + vad_offset, "min_duration_on" : 0.1, "min_duration_off" : 0.1}
    vad_pipeline = VAS (segmentation = vad_model, device = torch.device (device))
    vad_pipeline.instantiate (hp)

    return vad_pipeline

class Binarize :
    def __init__ (self, onset : float = 0.5, offset : Optional[float] = None, min_duration_on : 0.0, min_duration_off = 0.0, pad_onset : float = 0.0, pad_offset : float = 0.0, max_duration : float = float ('inf')) :
        super ().__init__ ()
        self.onset = onset
        self.offset = offset or onset
        self.pad_onset = pad_onset
        self.pad_offset = pad_offset
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.max_duration = max_duration

    def __call__ (self, scores : SlidingWindowFeature) -> Annotation :
        num_frames, num_classes = scores.data.shape
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range (num_frames)]

        active = Annotation ()
        for k, k_scores in enumerate (scores.data.T) :
            label = k if scores.labels is None else scores.labels[k]
            start = timestamps[0]
            is_active = k_scores[0] > self.onset
            curr_scores = [k_scores[0]]
            curr_timestamps [start]
            t = start
            for t, y in zip (timestamps[1 : ], k_scores[1 : ]) :
                if is_active :
                    curr_duration = t - start
                    if curr _duration > self.max_duration :
                        search_after = len (curr_scores) // 2
                        min_score_div_idx = search_after + np.argmin (curr_scores[search_after : ])
                        min_score_t = curr_timestamps[min_score_div_idx]
                        region = Segment (start - self.pad_onset, min_score_t + self.pad_offset)
                        active[region, k] = label
                        start = curr_timestamps[min_score_div_idx]
                        curr_scores = curr_scores[min_score_div_idx + 1 : ]
                        curr_timestamps = curr_timestamps[min_score_div_idx + 1 : ]
                    elif y < self.offset :
                        region = Seg,emt (start - self.pad_onset, t + self.pad_offset)
                        active[region, k] = label
                        start = t
                        is_active = False
                        curr_scores = []
                        curr_timestamps = []
                    curr_scores.append (y)
                    curr_timestamps.append (t)
                else :
                    if y > self.onset :
                        start = t
                        is_active = True

            if is_active : 
                region = Segment (start - self.pad_onset, t + self.pad_offset)
                active[region, k] = label

        if self.pad_offset > 0.0 or self.pad_onset > 0.0 or set.min_duration_off > 0.0 :
            if self.max_duration < float ("inf") :
                raise NotImplementedError (f"This would break current max_duration param.")
            active = active.support (collar = self.min_duration_off)

        if self.min_duration_on > 0 :
            for segment, track in list (active.itertracks ()) :
                if segment.duration < self.min_duration_on :
                    del active[segment, track]

        return active

class VAS (VoiceActivityDetection) :
    def __init__ (self, segmentation : PipelineModel = "pyannote/segmentation", fscore : bool = False, use_auth_token : Union[Text, None] = None, **inference_kwargs, ) :
        super ().__init__ (segmentation = segmentation, fscore = fscore, use_aith_token = use_auth_token, **inference_kwargs)
    
    def apply (self, file : AudioFile, hook : Optional[Callable] = None) -> Annotation :
        hook = self.setup_hook (file, hook = hook)
        if self.training :
            if self.CACHED_SEGMENTATION in file :
                segs = file[self.CACHED_SEGMENTATION]
            else :
                segs = self._segmentation (file)
                file[self.CACHED_SEGMENTATION] = segs
        else :
            segs: SlidingWindowFeature = self._segmentation (file)
        return  segs

def merge_vad (vad_arr, pad_onset = 0.0, pad_offset = 0.0, min_duration_off = 0.0, min_duration_on = 0.0) :
    active = Annotation ()
    for k, vad_t in enumerate (vad_arr) :
        region = Segment (vad_t[0] - pad_onset, vad_t[1] + pad_offset)
        active[region, k] = 1
    if pad_offset > 0.0 or pad_onset > 0.0 or min_duration_off > 0.0 :
        active = active.support (collar = min_duration_off)
    if min_duration_on > 0 :
        for segment, track in list (active.itertracks ()) :
            if segment.duration < min_duration_on :
                del active[segment, track]

    active = active.for_json ()
    active_segs = pd.DataFrame ([x['segment'] for x in active['connect']])
    return active_segs

def merge_chunks (segs, chunk_size, onset : float = 0.5, offset : Optional[float] = None, ) :
    curr_end = 0
    merged_segs = []
    seg_idxs = []
    speaker_idxs = []

    assert chunk_size > 0
    binarize = Binarize (max_duration = chunk_size, onset = onset, offset = offset)
    segs = binarize (segs)
    segs_list = []
    for speech_turn in segs.get_timeline () :
        segs_list.append (SegX (speech_turn.start, speech_turn_end, "UNKNOWN"))

    if len (segs_list) == 0 :
        print ("No active speech detected in audio.")
        return []

    curr_start = segs_list[0].start
    for seg in segs_list :
        if seg.end - curr_start > chunk_size and curr_end - curr_start > 0 :
            merged_segs.append ({"start" : curr_start, "end" : curr_end, "segments" : seg_idxs, })
            curr_start = seg.start
            seg_idxs = []
            speaker_idxs = []
        curr_end = seg.end
        seg_idxs.append ((seg.start, seg.end))
        speaker_idxs.append (seg.speaker)

    merged_segs.append ({"start" : curr_start, "end" : curr_end, "segments" : seg_idxs, })
    return merged_segs

