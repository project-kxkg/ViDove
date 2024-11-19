import threading
import time

import openai

# pytube deprecated
# from pytube import YouTube

import yt_dlp
from os import getenv, getcwd
from pathlib import Path
from enum import Enum, auto
import logging
import subprocess
from src.srt_util.srt import SrtScript
from src.srt_util.srt2ass import srt2ass
from time import time, strftime, gmtime, sleep
from src.translators.translator import Translator
from src.ASR.ASR import get_transcript
from openai import OpenAI

import shutil
from datetime import datetime

class TaskStatus(str, Enum):
    """
    An enumeration class representing the different statuses a task can have in the translation pipeline.
    TODO: add translation progress indicator (%).
    """
    CREATED = 'CREATED'
    INITIALIZING_ASR = 'INITIALIZING_ASR'
    PRE_PROCESSING = 'PRE_PROCESSING'
    TRANSLATING = 'TRANSLATING'
    POST_PROCESSING = 'POST_PROCESSING'
    OUTPUT_MODULE = 'OUTPUT_MODULE'

class Task:
    """
    A class representing a task in the translation pipeline. It includes methods for handling different stages of the task.
    If one want to add a new entry type (e.g. add support for different video formats),
    one should extend this class and override the `run` method.
    """

    @property
    def status(self):
        with self.__status_lock:
            return self.__status

    @status.setter
    def status(self, new_status):
        """
        Sets the new status of the task, ensuring thread safety with a lock.
        """
        with self.__status_lock:
            self.__status = new_status

    def __init__(self, task_id, task_local_dir, task_cfg):
        """
        Constructor for initializing a task with its ID, local directory, and configuration settings.
        """
        self.__status_lock = threading.Lock()
        self.__status = TaskStatus.CREATED
        self.gpu_status = 0

        self.task_id = task_id
        
        self.task_local_dir = task_local_dir
        self.ASR_setting = task_cfg["ASR"]
        self.translation_setting = task_cfg["translation"]
        self.translation_model = self.translation_setting["model"]
        
        self.output_type = task_cfg["output_type"]
        self.target_lang = task_cfg["target_lang"]
        self.source_lang = task_cfg["source_lang"]
        self.field = task_cfg["field"]
        self.pre_setting = task_cfg["pre_process"]
        self.post_setting = task_cfg["post_process"]
        self.chunk_size = task_cfg["translation"]["chunk_size"]
        
        self.audio_path = None
        self.SRT_Script = None
        self.result = None
        self.s_t = None
        self.t_e = None
        self.t_s = time()

        # logging setting
        self.task_logger = logging.getLogger(f"task_{task_id}")
        logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        self.task_logger.setLevel(logging.INFO)
        self.log_dir = "{}/{}_{}.log".format(task_local_dir, f"task_{task_id}", datetime.now().strftime("%m%d%Y_%H%M%S"))
        task_file_handler = logging.FileHandler(self.log_dir, 'w', encoding='utf-8')
        task_file_handler.setFormatter(logging.Formatter(logfmt)) 
        self.task_logger.addHandler(task_file_handler)
        # logging.basicConfig(level=logging.INFO, format=logfmt, handlers=[
        #     logging.FileHandler(
        #         "{}/{}_{}.log".format(task_local_dir, f"task_{task_id}", datetime.now().strftime("%m%d%Y_%H%M%S")),
        #         'w', encoding='utf-8')])
        
        print(f"Task ID: {self.task_id}")
        self.task_logger.info(f"Task ID: {self.task_id}")
        if "OPENAI_API_KEY" in task_cfg:
            self.task_logger.info("Using OPENAI_API_KEY from gradio interface.")
            self.api_key = task_cfg["OPENAI_API_KEY"]
        else:
            self.task_logger.info("Using OPENAI_API_KEY from environment variable.")
            self.api_key = getenv("OPENAI_API_KEY")
        self.task_logger.info(f"{self.source_lang} -> {self.target_lang} task in {self.field}")
        self.task_logger.info(f"Translation Model: {self.translation_model}")
        self.task_logger.info(f"Chunk Size: {self.chunk_size}")
        self.task_logger.info(f"ASR Model: {self.ASR_setting['ASR_model']}")
        self.task_logger.info(f"subtitle_type: {self.output_type['subtitle']}")
        self.task_logger.info(f"video_ouput: {self.output_type['video']}")
        self.task_logger.info(f"bilingual_ouput: {self.output_type['bilingual']}")
        self.task_logger.info("Pre-process setting:")
        for key in self.pre_setting:
            self.task_logger.info(f"{key}: {self.pre_setting[key]}")
        self.task_logger.info("Post-process setting:")
        for key in self.post_setting:
            self.task_logger.info(f"{key}: {self.post_setting[key]}")
        
        # init openai client
        self.client = OpenAI(api_key = self.api_key)
        # initialize translator
        self.translator = Translator(self.translation_model, 
                                     self.source_lang, 
                                     self.target_lang, 
                                     self.field, 
                                     self.task_id,
                                     self.client,
                                     self.chunk_size)

    @staticmethod
    def fromYoutubeLink(youtube_url, task_id, task_dir, task_cfg):
        """
        Creates a YoutubeTask instance from a YouTube URL.
        """
        return YoutubeTask(task_id, task_dir, task_cfg, youtube_url)

    @staticmethod
    def fromAudioFile(audio_path, task_id, task_dir, task_cfg):
        """
        Creates an AudioTask instance from an audio file path.
        """
        return AudioTask(task_id, task_dir, task_cfg, audio_path)
    
    @staticmethod
    def fromVideoFile(video_path, task_id, task_dir, task_cfg):
        """
        Creates a VideoTask instance from a video file path.
        """
        return VideoTask(task_id, task_dir, task_cfg, video_path)
    
    @staticmethod
    def fromSRTFile(srt_path, task_id, task_dir, task_cfg):
        """
        Creates a SRTTask instance from a srt file path.
        """
        return SRTTask(task_id, task_dir, task_cfg, srt_path)
    
    # Module 1 ASR: audio --> SRT_script
    def get_srt_class(self, pre_load_asr_model = None):
        """
        Handles the ASR module to convert audio to SRT script format.
        """
        # Instead of using the script_en variable directly, we'll use script_input
        self.status = TaskStatus.INITIALIZING_ASR

        if self.SRT_Script != None:
            self.task_logger.info("SRT input mode, skip ASR Module")
            return
        # get configs
        # shoud be modified after we incorporate more ASR methods
        method = self.ASR_setting["ASR_model"]
        # whisper_model = self.ASR_setting["whisper_config"]["whisper_model"]
        src_srt_path = self.task_local_dir.joinpath(f"task_{self.task_id}_{self.source_lang}.srt")

        # get transcript
        transcript = get_transcript(method, 
                                    src_srt_path, 
                                    self.source_lang,
                                    self.video_path,
                                    self.audio_path, 
                                    self.client, 
                                    self.task_logger,
                                    pre_load_asr_model)

        if transcript != None:  # if the audio is transfered
            if isinstance(transcript, str):
                self.SRT_Script = SrtScript.parse_from_srt_file(self.source_lang, 
                                                                self.target_lang, 
                                                                self.task_logger, 
                                                                self.client, 
                                                                domain = self.field, 
                                                                srt_str = transcript.rstrip())
            else:
                self.SRT_Script = SrtScript(self.source_lang, 
                                            self.target_lang, 
                                            transcript, 
                                            self.task_logger,
                                            self.client, 
                                            self.field)
            # save the srt script to local
            self.SRT_Script.write_srt_file_src(src_srt_path)
        else:
            raise RuntimeError(f"Failed to get transcript from audio file: {self.audio_path}")
        
    # Module 2: SRT preprocess: perform preprocess steps
    def preprocess(self):
        """
        Performs preprocessing steps on the SRT script.
        """
        self.status = TaskStatus.PRE_PROCESSING
        self.task_logger.info("--------------------Start Preprocessing SRT class--------------------")
        if self.pre_setting["sentence_form"]:
            self.SRT_Script.form_whole_sentence()
        if self.pre_setting["spell_check"]:
            self.SRT_Script.spell_check_term()
        if self.pre_setting["term_correct"]:
            self.SRT_Script.correct_with_force_term()
        processed_srt_path_src = str(Path(self.task_local_dir) / f'{self.task_id}_processed.srt')
        self.SRT_Script.write_srt_file_src(processed_srt_path_src)

        if self.output_type["subtitle"] == "ass":
            self.task_logger.info("write English .srt file to .ass")
            assSub_src = srt2ass(processed_srt_path_src, "default", "No", "Modest")
            self.task_logger.info('ASS subtitle saved as: ' + assSub_src)
        self.script_input = self.SRT_Script.get_source_only()
        pass
    
    def update_translation_progress(self, new_progress):
        """
        (UNUSED)
        Updates the progress (%) of the translation process.
        """
        if self.progress == TaskStatus.TRANSLATING:
            self.progress = TaskStatus.TRANSLATING.value[0], new_progress

    # Module 3: perform srt translation
    def translation(self):
        """
        Handles the translation of the SRT script.
        """
        self.task_logger.info("---------------------Start Translation--------------------")
        self.translator.set_srt(self.SRT_Script)
        self.translator.translate()
    
    # Module 4: perform srt post process steps
    def postprocess(self):
        """
        Performs post-processing steps on the translated SRT script.
        """
        self.status = TaskStatus.POST_PROCESSING

        self.task_logger.info("---------------------Start Post-processing SRT class---------------------")
        if self.post_setting["check_len_and_split"]:
            self.SRT_Script.check_len_and_split()
        if self.post_setting["remove_trans_punctuation"]:
            self.SRT_Script.remove_trans_punctuation()
        self.task_logger.info("---------------------Post-processing SRT class finished---------------------")

    # Module 5: output module
    def output_render(self):
        """
        Handles the output rendering process, including video and subtitle generation.
        """
        self.status = TaskStatus.OUTPUT_MODULE
        video_out = self.output_type["video"]
        subtitle_type = self.output_type["subtitle"]
        is_bilingual = self.output_type["bilingual"]

        results_dir =f"{self.task_local_dir}/results"

        subtitle_path = f"{results_dir}/{self.task_id}_{self.target_lang}.srt"
        self.SRT_Script.write_srt_file_translate(subtitle_path)
        if is_bilingual:
            subtitle_path = f"{results_dir}/{self.task_id}_{self.source_lang}_{self.target_lang}.srt"
            self.SRT_Script.write_srt_file_bilingual(subtitle_path)

        if subtitle_type == "ass":
            self.task_logger.info("write .srt file to .ass")
            subtitle_path = srt2ass(subtitle_path, "default", "No", "Modest")
            self.task_logger.info('ASS subtitle saved as: ' + subtitle_path)

        final_res = subtitle_path

        # encode to .mp4 video file
        if video_out and self.video_path is not None:
            self.task_logger.info("encoding video file")
            self.task_logger.info(f'ffmpeg comand: \nffmpeg -i {self.video_path} -vf "subtitles={subtitle_path}" {results_dir}/{self.task_id}.mp4')
            subprocess.run(
                ["ffmpeg",
                    "-i", self.video_path,
                    "-vf", f"subtitles={subtitle_path}",
                    f"{results_dir}/{self.task_id}.mp4"])
            final_res = f"{results_dir}/{self.task_id}.mp4"

        self.t_e = time()
        self.task_logger.info(
            "Pipeline finished, time duration:{}".format(strftime("%H:%M:%S", gmtime(self.t_e - self.t_s))))
        return final_res
    
    def run_pipeline(self, pre_load_asr_model = None):
        """
        Executes the entire pipeline process for the task.
        """
        self.get_srt_class(pre_load_asr_model)
        self.preprocess()
        self.translation()
        self.postprocess()
        self.result = self.output_render()
        
        # print(self.result)

class YoutubeTask(Task):
    def __init__(self, task_id, task_local_dir, task_cfg, youtube_url):
        super().__init__(task_id, task_local_dir, task_cfg)
        self.task_logger.info("Task Creation method: Youtube Link")
        self.youtube_url = youtube_url
        self.video_resolution = task_cfg["video_download"]["resolution"]
        # self.model = model

    def run(self, pre_load_asr_model = None):

        self.task_logger.info(f"Youtube URL: {self.youtube_url}")
        self.task_logger.info(f"Video Resolution: {self.video_resolution}")
        video_download_path = f"{self.task_local_dir}/task_{self.task_id}.mp4"
        audio_download_dir = f"{self.task_local_dir}/task_{self.task_id}.mp3"

        if self.video_resolution == "best":
            video_format = "bestvideo[ext=mp4]+bestaudio/bestvideo"
        elif self.video_resolution in [360, 480, 720]:
            video_format = f"bestvideo[height={self.video_resolution}][ext=mp4]+bestaudio[ext=mp3]/worstvideo[ext=mp4]+bestaudio[ext=mp3]/worst[ext=mp4]"
        else:
            raise RuntimeError(f"Unsupported video resolution: {self.video_resolution}")
        
        video_opts = {
            'format': video_format, 
            'outtmpl': video_download_path,
        }

        audio_opts = {
            'format': 'bestaudio[ext=mp3]/bestaudio', 
            'outtmpl': audio_download_dir,
        }

        with yt_dlp.YoutubeDL(video_opts) as ydl:
            try:
                ydl.download([self.youtube_url])
            except yt_dlp.utils.DownloadError as e:
                self.task_logger.error(e)
                raise RuntimeError(f"Failed to download video {self.youtube_url}")
            ydl.close()
        
        with yt_dlp.YoutubeDL(audio_opts) as ydl:
            try:
                ydl.download([self.youtube_url])
            except yt_dlp.utils.DownloadError as e:
                self.task_logger.error(e)
                raise RuntimeError(f"Failed to download audio {self.youtube_url}")
            ydl.close()
        
        self.video_path = self.task_local_dir.joinpath(f"task_{self.task_id}.mp4")
        self.audio_path = self.task_local_dir.joinpath(f"task_{self.task_id}.mp3")

        self.task_logger.info(f" Video File Dir: {self.video_path}")
        self.task_logger.info(f" Audio File Dir: {self.audio_path}")
        self.task_logger.info(" Data Prep Complete. Start pipeline")

        super().run_pipeline(pre_load_asr_model)

class AudioTask(Task):
    def __init__(self, task_id, task_local_dir, task_cfg, audio_path):
        super().__init__(task_id, task_local_dir, task_cfg)
        # TODO: check audio format
        self.task_logger.info("Task Creation method: Audio File")
        self.audio_path = audio_path
        self.video_path = None

    def run(self, pre_load_asr_model = None):
        self.task_logger.info(f"Video File Dir: {self.video_path}")
        self.task_logger.info(f"Audio File Dir: {self.audio_path}")
        self.task_logger.info("Data Prep Complete. Start pipeline")
        super().run_pipeline(pre_load_asr_model)

class VideoTask(Task):
    def __init__(self, task_id, task_local_dir, task_cfg, video_path):
        super().__init__(task_id, task_local_dir, task_cfg)
        # TODO: check video format {.mp4}
        self.task_logger.info("Task Creation method: Video File")
        new_video_path = f"{task_local_dir}/task_{self.task_id}.mp4"
        self.task_logger.info(f"Copy video file to: {new_video_path}")
        shutil.copyfile(video_path, new_video_path)
        self.video_path = new_video_path

    def run(self, pre_load_asr_model = None):
        self.task_logger.info("using ffmpeg to extract audio")
        subprocess.run(
                ['ffmpeg', '-i', self.video_path, '-f', 'mp3',
                 '-ab', '192000', '-vn', self.task_local_dir.joinpath(f"task_{self.task_id}.mp3")])
        self.task_logger.info("audio extraction finished")

        self.audio_path = self.task_local_dir.joinpath(f"task_{self.task_id}.mp3")
        self.task_logger.info(f" Video File Dir: {self.video_path}")
        self.task_logger.info(f" Audio File Dir: {self.audio_path}")
        self.task_logger.info("Data Prep Complete. Start pipeline")
        super().run_pipeline(pre_load_asr_model)

class SRTTask(Task):
    def __init__(self, task_id, task_local_dir, task_cfg, srt_path):
        super().__init__(task_id, task_local_dir, task_cfg)
        self.task_logger.info("Task Creation method: SRT File")
        self.audio_path = None
        self.video_path = None
        new_srt_path = f"{task_local_dir}/task_{self.task_id}_{self.source_lang}.srt"
        self.task_logger.info(f"Copy video file to: {new_srt_path}")
        shutil.copyfile(srt_path, new_srt_path)
        self.SRT_Script = SrtScript.parse_from_srt_file(self.source_lang, 
                                                        self.target_lang, 
                                                        self.task_logger, 
                                                        self.client, 
                                                        domain = self.field, 
                                                        path = srt_path)

    def run(self):
        self.task_logger.info(f"Video File Dir: {self.video_path}")
        self.task_logger.info(f"Audio File Dir: {self.audio_path}")
        self.task_logger.info("Data Prep Complete. Start pipeline")
        super().run_pipeline()
