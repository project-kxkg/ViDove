import argparse
import shutil
from pathlib import Path
from uuid import uuid4

from yaml import Loader, load

from src.task import Task

"""
    Main entry for terminal environment.
    Use it for debug and development purpose. 
    Usage: python3 entries/run.py [-h] [--link LINK] [--video_file VIDEO_FILE] [--audio_file AUDIO_FILE] [--srt_file SRT_FILE] [--continue CONTINUE]
              [--launch_cfg LAUNCH_CFG] [--task_cfg TASK_CFG]
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--link", help="youtube video link here", default=None, type=str, required=False
    )
    parser.add_argument(
        "--video_file",
        help="local video path here",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--audio_file",
        help="local audio path here",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--srt_file",
        help="srt file input path here",
        default=None,
        type=str,
        required=False,
    )  # Deprecated
    # parser.add_argument("--continue", help="task_id that need to continue", default=None, type=str, required=False) # need implement
    parser.add_argument(
        "--launch_cfg",
        help="launch config path",
        default="./configs/local_launch.yaml",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--is_assistant",
        help="is assistant mode",
        default=False,
        type=bool,
        required=False,
    )
    parser.add_argument(
        "--task_cfg",
        help="task config path",
        default="./configs/task_config.yaml",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # read args and configs
    args = parse_args()
    launch_cfg = load(open(args.launch_cfg), Loader=Loader)
    task_cfg = load(open(args.task_cfg), Loader=Loader)

    # initialize dir
    local_dir = Path(launch_cfg["local_dump"])
    if not local_dir.exists():
        local_dir.mkdir(parents=False, exist_ok=False)

    # get task id
    task_id = str(uuid4())

    # create locak dir for the task
    task_dir = local_dir.joinpath(f"task_{task_id}")
    task_dir.mkdir(parents=False, exist_ok=False)
    task_dir.joinpath("results").mkdir(parents=False, exist_ok=False)

    # add is_assistant to task_cfg
    task_cfg["is_assistant"] = args.is_assistant

    # disable spell check and term correct for assistant mode
    if args.is_assistant:
        task_cfg["pre_process"]["spell_check"] = False
        task_cfg["pre_process"]["term_correct"] = False

    # Task create
    if args.link is not None:
        try:
            task = Task.fromYoutubeLink(args.link, task_id, task_dir, task_cfg)
        except:
            shutil.rmtree(task_dir)
            raise RuntimeError("failed to create task from youtube link")
    elif args.video_file is not None:
        try:
            task = Task.fromVideoFile(args.video_file, task_id, task_dir, task_cfg)
        except:
            shutil.rmtree(task_dir)
            raise RuntimeError("failed to create task from video file")
    elif args.audio_file is not None:
        try:
            task = Task.fromVideoFile(args.audio_file, task_id, task_dir, task_cfg)
        except:
            shutil.rmtree(task_dir)
            raise RuntimeError("failed to create task from audio file")
    elif args.srt_file is not None:
        try:
            task = Task.fromSRTFile(args.srt_file, task_id, task_dir, task_cfg)
        except:
            shutil.rmtree(task_dir)
            raise RuntimeError("failed to create task from srt file")

    # add task to the status queue
    task.run()
