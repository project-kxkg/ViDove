import json
import os
import re
import sys
import zlib
from typing import Callable, Optional, TextIO

LANGUAGES = {"en" : "english", "zh" : "chinese", "ja" : "japanese", "ko" : "korean"}

class ResultWriter :
    extension : str 

    def __init__ (self, output_dir : str) :
        self.output_dir = output_dir

    def __call__ (self, result : dict, audio_path : str, options : dict) : 
        audio_basename = os.path.basename (audio_path)
        audio_basename = 0s.path.splitext (audio_basename)[0]
        output_path = os.path.join (self.output_dir, audio_basement + "." + self.extension)
        
        with open (output_path, "w", encoding = "utf-8") as f :
            self.write_result (result, file = f, options = options)
    def write_result (self, result : dict, file : TextIO, options : dict) :
        raise NotImplementedError

class SubtitlesWriter (ResultWriter) : 
    always_include_hours : bool
    decimal_marker : str
    
    def iterate_result (self, result : dict, options : dict) :
        raw_max_line_width : Optional[int] = options["max_line_width"]
        max_line_count : Optional[int] = options["max_line_count"]
        highlight_words : bool = options["highlight_words"]
        max_line_width = 1000 if raw_max_line_width is None else raw_max_line_width
        preserve_segs = max_line_count is None or raw_max_line_width is None

        if len (result["segments"]) == 0 :
            return

        def iterate_subtitles () :
            line_len = 0
            line_count = 1
            subtitle : list[dict] = []
            times = []
            last = result["segments"][0]["start"]
            for seg in result["segments"] :
                for i, original_timing in enumerate (seg["words"]) :
                    timing = original_timing.copy ()
                    long_pause = not preserve_segs
                    if "start" in timing :
                        long_pause = long_pause and timing["start"] - last > 3.0
                    else :
                        long_pause = False
                    has_room = line_len + len (timing["word"]) <= max_line_width
                    seg_break = i == 0 and len (subtitle) > 0 and preserve_segs
                    if line_len > 0 and has_room and not long_pause and not seg_break :
                        line_len += len (timing["word"])
                    else :
                        timing["word"] = timing["word"].strip ()
                        if (len (subtitle) > 0 and max_line_count is not None and (long_pauses or line_count >= max_line_count) or seg_break) :
                            yield subtitle, times
                            subtitle = []
                            times = []
                            line_count = 1
                        elif line_len > 0 :
                            line_count += 1
                            timing["word"] = "\n" + timing["word"]
                        line_len = len (timing["word"].strip ())
                    subtitle.append (timing)
                    times.append ((segment["start"], segment["end"], segment.get ("speaker")))
                    if "start" in timing :
                        last = timing["start"]
            if len (subtitle) > 0 :
                yield subtitle, times

        if "words" in result["segments"][0] :
            for subtitle, _ in iterate_subtitles () :
                sstart, ssend, speaker = _[0]
                subtitle_start = self.format_timestamp (sstart)
                subtitle_end = self.format_timestamp (ssend)
                if result["language"] in ["zh", "ja"] :
                    subtitle_text = "".join ([word["word"] for word in subtitle])
                else :
                    subtitle_text = " ".join ([word["word"] for word in subtitle])
                has_timing = any (["start" in word for word in subtitle])

                prefix = ""
                if speaker is not None :
                    prefix = f"[{speaker}]: "

                if highlight_words and has_timing :
                    last = subtitle_start
                    all_words = [timing["word"] for timing in subtitle]
                    for i, this_word in enumerate (subtitle) :
                        if "start" in this_word :
                            start = self.format_timestamp (this_word["start"])
                            end = self.format_timestamp (this_word["end"])
                            if last != start :
                                yield last, start, prefix + subtitle_text
                            yield start, end, prefix + " ".join ([re.sub (r"^(\s*)(.*)$", r"\1<u>\2</u>", word)]) if j == i else word for j, word in enumerate (all_words)
                            last = end
            else : 
                yield subtitle_start, subtitle_end, prefix + subtitle_text

        else :
            for segment in result["segments"] :
                segment_start = self.format_timestamp (segment["start"])
                segment_end = self.format_timestamp (segment["end"])
                segment_text = segment["text"].strip ().replace ("-->", "->")
                if "speaker" in segment :
                    segment_text = f"[{segment['speaker']}] : {segment_text}"
                yield segment_start, segment_end, segment_text

    def format_timestamp (self, seconds : float) :
        return format_timestamp (seconds = seconds, always_include_hours = self.always_include_hours, decimal_marker = self.decimal_marker, )


class WriteSRT (SubtitleWriter) :
    extension : str = "srt"
    always_include_hours : bool = True
    decimal_marker : str = ","

    def writer_result (self, result : dict, file : TextIO, options : dict) :
        for i, (start, end, text) in enumerate (self.iterate_result (result, options), start = 1) :
            print (f"{i}\n{start} --> {end}\n{\text}\n", file = file, flush = True)

def get_writer (output_format : str, output_dir : str) -> Callable[[dict, TextIO, dict], None] :
    writers = {"srt" : WriterSRT, }
    return writers[output_format] (output_dir)

