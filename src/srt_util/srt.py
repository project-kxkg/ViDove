import os
import re
from copy import deepcopy
from datetime import timedelta

from tqdm import tqdm

from .. import dict_util

# punctuation dictionary for supported languages
punctuation_dict = {
    "EN": {
        "punc_str": ". , ? ! : ; - ( ) [ ] { }",
        "comma": ", ",
        "sentence_end": [".", "!", "?", ";"]
    },
    "ES": {
        "punc_str": ". , ? ! : ; - ( ) [ ] { } ¡ ¿",
        "comma": ", ",
        "sentence_end": [".", "!", "?", ";", "¡", "¿"]
    },
    "FR": {
        "punc_str": ".,?!:;«»—",
        "comma": ", ",
        "sentence_end": [".", "!", "?", ";"]
    },
    "DE": {
        "punc_str": ".,?!:;„“–",
        "comma": ", ",
        "sentence_end": [".", "!", "?", ";"]
    },
    "RU": {
        "punc_str": ".,?!:;-«»—",
        "comma": ", ",
        "sentence_end": [".", "!", "?", ";"]
    },
    "ZH": {
        "punc_str": "。，？！：；（）",
        "comma": "，",
        "sentence_end": ["。", "！", "？"]
    },
    "JA": {
        "punc_str": "。、？！：；（）",
        "comma": "、",
        "sentence_end": ["。", "！", "？"]
    },
    "AR": {
        "punc_str": ".,?!:;-()[]،؛ ؟ «»",
        "comma": "، ",
        "sentence_end": [".", "!", "?", ";", "؟"]
    },
    "KR": {
        "punc_str": ".,?!:;()[]{}",
        "comma": ", ",
        "sentence_end": [".", "!", "?", ";"]
    }
}

dict_path = "./domain_dict"

class SrtSegment(object):
    def __init__(self, src_lang, tgt_lang, *args) -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        if isinstance(args[0], dict):
            segment = args[0]
            self.start = segment['start']
            self.end = segment['end']
            self.start_ms = int((segment['start'] * 100) % 100 * 10)
            self.end_ms = int((segment['end'] * 100) % 100 * 10)

            if self.start_ms == self.end_ms and int(segment['start']) == int(segment['end']):  # avoid empty time stamp
                self.end_ms += 500

            self.start_time = timedelta(seconds=int(segment['start']), milliseconds=self.start_ms)
            self.end_time = timedelta(seconds=int(segment['end']), milliseconds=self.end_ms)
            if self.start_ms == 0:
                self.start_time_str = str(0) + str(self.start_time).split('.')[0] + ',000'
            else:
                self.start_time_str = str(0) + str(self.start_time).split('.')[0] + ',' + \
                                      str(self.start_time).split('.')[1][:3]
            if self.end_ms == 0:
                self.end_time_str = str(0) + str(self.end_time).split('.')[0] + ',000'
            else:
                self.end_time_str = str(0) + str(self.end_time).split('.')[0] + ',' + str(self.end_time).split('.')[1][
                                                                                      :3]
            self.source_text = segment['text'].lstrip()
            self.duration = f"{self.start_time_str} --> {self.end_time_str}"
            self.translation = ""

        elif isinstance(args[0], list):
            self.source_text = args[0][2]
            self.duration = args[0][1]
            self.start_time_str = self.duration.split(" --> ")[0]
            self.end_time_str = self.duration.split(" --> ")[1]

            # parse the time to float
            self.start_ms = int(self.start_time_str.split(',')[1]) / 10
            self.end_ms = int(self.end_time_str.split(',')[1]) / 10
            start_list = self.start_time_str.split(',')[0].split(':')
            self.start = int(start_list[0]) * 3600 + int(start_list[1]) * 60 + int(start_list[2]) + self.start_ms / 100
            end_list = self.end_time_str.split(',')[0].split(':')
            self.end = int(end_list[0]) * 3600 + int(end_list[1]) * 60 + int(end_list[2]) + self.end_ms / 100
            if len(args[0]) < 5:
                self.translation = ""
            else:
                self.translation = args[0][3]
            

    def merge_seg(self, seg):
        """
        Merge the segment seg with the current segment in place.
        :param seg: Another segment that is strictly next to current one.
        :return: None
        """
        # assert seg.start_ms == self.end_ms, f"cannot merge discontinuous segments."
        self.source_text += f' {seg.source_text}'
        self.translation += f' {seg.translation}'
        self.end_time_str = seg.end_time_str
        self.end = seg.end
        self.end_ms = seg.end_ms
        self.duration = f"{self.start_time_str} --> {self.end_time_str}"

    def __add__(self, other):
        """
        Merge the segment seg with the current segment, and return the new constructed segment.
        No in-place modification.
        This is used for '+' operator.
        :param other: Another segment that is strictly next to added segment.
        :return: new segment of the two sub-segments
        """
        result = deepcopy(self)
        result.merge_seg(other)
        return result

    def remove_trans_punc(self) -> None:
        """
        remove punctuations in translation text
        :return: None
        """
        punc_str = punctuation_dict[self.tgt_lang]["punc_str"]
        for punc in punc_str:
            self.translation = self.translation.replace(punc, ' ')
        # translator = str.maketrans(punc, ' ' * len(punc))
        # self.translation = self.translation.translate(translator)

    def __str__(self) -> str:
        return f'{self.duration}\n{self.source_text}\n\n'

    def get_trans_str(self) -> str:
        return f'{self.duration}\n{self.translation}\n\n'

    def get_bilingual_str(self) -> str:
        return f'{self.duration}\n{self.source_text}\n{self.translation}\n\n'


class SrtScript(object):
    def __init__(self, src_lang, tgt_lang, segments, task_logger, client, domain="General") -> None:
        self.task_logger = task_logger
        self.domain = domain
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.segments = [SrtSegment(self.src_lang, self.tgt_lang, seg) for seg in segments]
        self.client = client

        if self.domain != "General":
            if os.path.exists(f"{dict_path}/{self.domain}") and\
                              os.path.exists(f"{dict_path}/{self.domain}/{src_lang}.csv") and os.path.exists(f"{dict_path}/{self.domain}/{tgt_lang}.csv" ):
                # TODO: load dictionary
                self.dict = dict_util.term_dict(f"{dict_path}/{self.domain}", src_lang, tgt_lang)
                ...
            else:
                self.task_logger.error(f"domain {self.domain} or related dictionary({src_lang} or {tgt_lang}) doesn't exist, fallback to general domain, this will disable correct_with_force_term and spell_check_term")
                self.domain = "General"


    @classmethod
    def parse_from_srt_file(cls, src_lang, tgt_lang, task_logger, client, domain, path = None, srt_str = None):
        if path is not None:
            with open(path, 'r', encoding="utf-8") as f:
                script_lines = [line.rstrip() for line in f.readlines()]
        elif srt_str is not None:
            script_lines = srt_str.splitlines()
        else:
            raise RuntimeError("need input Srt Path or Srt String")

        bilingual = False
        if script_lines[2] != '' and script_lines[3] != '':
            bilingual = True
        segments = []
        if bilingual:
            for i in range(0, len(script_lines), 5):
                segments.append(list(script_lines[i:i + 5]))
        else:
            for i in range(0, len(script_lines), 4):
                segments.append(list(script_lines[i:i + 4]))
        return cls(src_lang, tgt_lang, segments, task_logger, client, domain)

    def merge_segs(self, idx_list) -> SrtSegment:
        """
        Merge entire segment list to a single segment
        :param idx_list: List of index to merge
        :return: Merged list
        """
        if not idx_list:
            raise NotImplementedError('Empty idx_list')
        seg_result = deepcopy(self.segments[idx_list[0]])
        if len(idx_list) == 1:
            return seg_result

        for idx in range(1, len(idx_list)):
            seg_result += self.segments[idx_list[idx]]

        return seg_result

    def form_whole_sentence(self):
        """
        Concatenate or Strip sentences and reconstruct segments list. This is because of
        improper segmentation from openai-whisper.
        :return: None
        """
        self.task_logger.info("Forming whole sentences...")
        merge_list = []  # a list of indices that should be merged e.g. [[0], [1, 2, 3, 4], [5, 6], [7]]
        sentence = []
        ending_puncs = punctuation_dict[self.src_lang]["sentence_end"]
        # Get each entire sentence of distinct segments, fill indices to merge_list
        for i, seg in enumerate(self.segments):
            if seg.source_text[-1] in ending_puncs and len(seg.source_text) > 10 and 'vs.' not in seg.source_text:
                sentence.append(i)
                merge_list.append(sentence)
                sentence = []
            else:
                sentence.append(i)

        # Reconstruct segments, each with an entire sentence
        segments = []
        for idx_list in merge_list:
            if len(idx_list) > 1:
                self.task_logger.info("merging segments: %s", idx_list)
            segments.append(self.merge_segs(idx_list))

        self.segments = segments

    def remove_trans_punctuation(self):
        """
        Post-process: remove all punc after translation and split
        :return: None
        """
        for i, seg in enumerate(self.segments):
            seg.remove_trans_punc()
        self.task_logger.info("Removed punctuation in translation.")

    def set_translation(self, translate: str, id_range: tuple, model, video_name, video_link=None):
        start_seg_id = id_range[0]
        end_seg_id = id_range[1]

        src_text = ""
        for i, seg in enumerate(self.segments[start_seg_id - 1:end_seg_id]):
            src_text += seg.source_text
            src_text += '\n\n'

        def inner_func(target, input_str):
            # handling merge sentences issue.

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "Your task is to merge or split sentences into a specified number of lines as required. You need to ensure the meaning of the sentences as much as possible, but when necessary, a sentence can be divided into two lines for output"},
                    {"role": "system", "content": "Note: You only need to output the processed {} sentences. If you need to output a sequence number, please separate it with a colon.".format(self.tgt_lang)},
                    {"role": "user", "content": 'Please split or combine the following sentences into {} sentences:\n{}'.format(target, input_str)}
                ],
                temperature=0.15
            )
            return response.choices[0].message.content.strip()

        # handling merge sentences issue.
        lines = translate.split('\n\n')
        if len(lines) < (end_seg_id - start_seg_id + 1):
            count = 0
            solved = True
            while count < 5 and len(lines) != (end_seg_id - start_seg_id + 1):
                count += 1
                print("Solving Unmatched Lines|iteration {}".format(count))
                self.task_logger.error("Solving Unmatched Lines|iteration {}".format(count))

                flag = True
                while flag:
                    flag = False
                    try:
                        translate = inner_func(end_seg_id - start_seg_id + 1, translate)
                    except Exception as e:
                        print("An error has occurred during solving unmatched lines:", e)
                        print("Retrying...")
                        self.task_logger.error("An error has occurred during solving unmatched lines:", e)
                        self.task_logger.error("Retrying...")
                        flag = True
                lines = translate.split('\n')

            if len(lines) < (end_seg_id - start_seg_id + 1):
                solved = False
                print("Failed Solving unmatched lines, Manually parse needed")
                self.task_logger.error("Failed Solving unmatched lines, Manually parse needed")

            # FIXME: put the error log in our log file
            if not os.path.exists("./logs"):
                os.mkdir("./logs")
            if video_link:
                log_file = "./logs/log_link.csv"
                log_exist = os.path.exists(log_file)
                with open(log_file, "a") as log:
                    if not log_exist:
                        log.write("range_of_text,iterations_solving,solved,file_length,video_link" + "\n")
                    log.write(str(id_range) + ',' + str(count) + ',' + str(solved) + ',' + str(
                        len(self.segments)) + ',' + video_link + "\n")
            else:
                log_file = "./logs/log_name.csv"
                log_exist = os.path.exists(log_file)
                with open(log_file, "a") as log:
                    if not log_exist:
                        log.write("range_of_text,iterations_solving,solved,file_length,video_name" + "\n")
                    log.write(str(id_range) + ',' + str(count) + ',' + str(solved) + ',' + str(
                        len(self.segments)) + ',' + video_name + "\n")
            # print(lines)

        for i, seg in enumerate(self.segments[start_seg_id - 1:end_seg_id]):
            # naive way to due with merge translation problem
            # TODO: need a smarter solution

            if i < len(lines):
                if "Note:" in lines[i]:  # to avoid note
                    lines.remove(lines[i])
                    max_num -= 1
                    if i == len(lines) - 1:
                        break
                if lines[i][0] in [' ', '\n']:
                    lines[i] = lines[i][1:]
                seg.translation = lines[i]

    def split_seg(self, seg, text_threshold, time_threshold):
        # evenly split seg to 2 parts and add new seg into self.segments
        # ignore the initial comma to solve the recursion problem
        src_comma_str = punctuation_dict[self.src_lang]["comma"]
        tgt_comma_str = punctuation_dict[self.tgt_lang]["comma"]

        if len(seg.source_text) > 2:
            if seg.source_text[:2] == src_comma_str:
                seg.source_text = seg.source_text[2:]
        if seg.translation[0] == tgt_comma_str:
            seg.translation = seg.translation[1:]

        source_text = seg.source_text
        translation = seg.translation

        # split the text based on commas
        src_commas = [m.start() for m in re.finditer(src_comma_str, source_text)]
        trans_commas = [m.start() for m in re.finditer(tgt_comma_str, translation)]
        if len(src_commas) != 0:
            src_split_idx = src_commas[len(src_commas) // 2] if len(src_commas) % 2 == 1 else src_commas[
                len(src_commas) // 2 - 1]
        else:
            # split the text based on spaces
            src_space = [m.start() for m in re.finditer(' ', source_text)]
            if len(src_space) > 0:
                src_split_idx = src_space[len(src_space) // 2] if len(src_space) % 2 == 1 else src_space[
                    len(src_space) // 2 - 1]
            else:
                src_split_idx = 0

        if len(trans_commas) != 0:
            trans_split_idx = trans_commas[len(trans_commas) // 2] if len(trans_commas) % 2 == 1 else trans_commas[
                len(trans_commas) // 2 - 1]
        else:
            trans_split_idx = len(translation) // 2

            # to avoid split English word
            for i in range(trans_split_idx, len(translation)):
                if not translation[i].encode('utf-8').isalpha():
                    trans_split_idx = i
                    break

        # split the time duration based on text length
        time_split_ratio = trans_split_idx / (len(seg.translation) - 1)

        src_seg1 = source_text[:src_split_idx]
        src_seg2 = source_text[src_split_idx:]
        trans_seg1 = translation[:trans_split_idx]
        trans_seg2 = translation[trans_split_idx:]

        start_seg1 = seg.start
        end_seg1 = start_seg2 = seg.start + (seg.end - seg.start) * time_split_ratio
        end_seg2 = seg.end

        seg1_dict = {}
        seg1_dict['text'] = src_seg1
        seg1_dict['start'] = start_seg1
        seg1_dict['end'] = end_seg1
        seg1 = SrtSegment(self.src_lang, self.tgt_lang, seg1_dict)
        seg1.translation = trans_seg1

        seg2_dict = {}
        seg2_dict['text'] = src_seg2
        seg2_dict['start'] = start_seg2
        seg2_dict['end'] = end_seg2
        seg2 = SrtSegment(self.src_lang, self.tgt_lang, seg2_dict)
        seg2.translation = trans_seg2

        result_list = []
        if len(seg1.translation) > text_threshold and (seg1.end - seg1.start) > time_threshold:
            result_list += self.split_seg(seg1, text_threshold, time_threshold)
        else:
            result_list.append(seg1)

        if len(seg2.translation) > text_threshold and (seg2.end - seg2.start) > time_threshold:
            result_list += self.split_seg(seg2, text_threshold, time_threshold)
        else:
            result_list.append(seg2)

        return result_list

    def check_len_and_split(self, text_threshold=30, time_threshold=1.0):
        # if sentence length >= threshold and sentence duration > time_threshold, split this segments to two
        self.task_logger.info("performing check_len_and_split")
        segments = []
        for i, seg in enumerate(self.segments):
            if len(seg.translation) > text_threshold and (seg.end - seg.start) > time_threshold:
                seg_list = self.split_seg(seg, text_threshold, time_threshold)
                self.task_logger.info("splitting segment {} in to {} parts".format(i + 1, len(seg_list)))
                segments += seg_list
            else:
                segments.append(seg)

        self.segments = segments
        self.task_logger.info("check_len_and_split finished")

    def check_len_and_split_range(self, range, text_threshold=30, time_threshold=1.0):
        # DEPRECATED
        # if sentence length >= text_threshold, split this segments to two
        start_seg_id = range[0]
        end_seg_id = range[1]
        extra_len = 0
        segments = []
        for i, seg in enumerate(self.segments[start_seg_id - 1:end_seg_id]):
            if len(seg.translation) > text_threshold and (seg.end - seg.start) > time_threshold:
                seg_list = self.split_seg(seg, text_threshold, time_threshold)
                segments += seg_list
                extra_len += len(seg_list) - 1
            else:
                segments.append(seg)

        self.segments[start_seg_id - 1:end_seg_id] = segments
        return extra_len

    def correct_with_force_term(self):
        ## force term correction
        self.task_logger.info("performing force term correction")

        # check domain
        if self.domain == "General":
            self.task_logger.info("General domain could not perform correct_with_force_term. skip this step.")
            pass
        else:
            keywords = list(self.dict.keys())
            keywords.sort(key=lambda x: len(x), reverse=True)

            for word in keywords:
                for i, seg in enumerate(self.segments):
                    if word in seg.source_text.lower():
                        seg.source_text = re.sub(fr"({word}es|{word}s?)\b", "{}".format(self.dict.get(word)),
                                                seg.source_text, flags=re.IGNORECASE)
                        self.task_logger.info(
                            "replace term: " + word + " --> " + self.dict.get(word) + " in time stamp {}".format(
                                i + 1))
                        self.task_logger.info("source text becomes: " + seg.source_text)


    def fetchfunc(self, word, threshold):
        import enchant
        result = word
        distance = 0
        threshold = threshold * len(word)
        temp = ""
        for matched in self.dict:
            if (" " in matched and " " in word) or (" " not in matched and " " not in word):
                if enchant.utils.levenshtein(word, matched) < enchant.utils.levenshtein(word, temp):
                    temp = matched
        if enchant.utils.levenshtein(word, temp) < threshold:
            distance = enchant.utils.levenshtein(word, temp)
            result = temp
        return distance, result

    def extract_words(self, sentence, n):
        # this function split the sentence to chunks by n of words
        # e.g. sentence: "this, is a sentence", n = 2
        # result: ["this,", "is", "a", ["sentence"], ["this,", "is"], "is a", "a sentence"]
        words = sentence.split()
        res = []
        for j in range(n, 0, -1):
            res += [words[i:i + j] for i in range(len(words) - j + 1)]
        return res

    def spell_check_term(self):
        self.task_logger.info("performing spell check")

        # check domain
        if self.domain == "General":
            self.task_logger.info("General domain could not perform spell_check_term. skip this step.")
            pass

        import enchant
        dict = enchant.Dict('en_US')

        for seg in tqdm(self.segments):
            ready_words = self.extract_words(seg.source_text, 2)
            for i in range(len(ready_words)):
                word_list = ready_words[i]
                word, real_word, pos = self.get_real_word(word_list)
                if not dict.check(real_word) and (real_word not in self.dict.keys()):
                    distance, correct_term = self.fetchfunc(real_word, 0.3)
                    if distance != 0:
                        seg.source_text = re.sub(word[:pos], correct_term, seg.source_text, flags=re.IGNORECASE)
                        self.task_logger.info(
                            "replace: " + word[:pos] + " to " + correct_term + "\t distance = " + str(distance))

    def get_real_word(self, word_list: list):
        word = ""
        for w in word_list:
            word += f"{w} "
        word = word[:-1]  # "this, is"
        if word[-2:] == ".\n":
            real_word = word[:-2].lower()
            n = -2
        elif word[-1:] in [".", "\n", ",", "!", "?"]:
            real_word = word[:-1].lower()
            n = -1
        else:
            real_word = word.lower()
            n = 0
        return word, real_word, len(word) + n

    ## WRITE AND READ FUNCTIONS ##

    def get_source_only(self):
        # return a string with pure source text
        result = ""
        for i, seg in enumerate(self.segments):
            result += f'{seg.source_text}\n\n\n'  # f'SENTENCE {i+1}: {seg.source_text}\n\n\n'

        return result

    def reform_src_str(self):
        result = ""
        for i, seg in enumerate(self.segments):
            result += f'{i + 1}\n'
            result += str(seg)
        return result

    def reform_trans_str(self):
        result = ""
        for i, seg in enumerate(self.segments):
            result += f'{i + 1}\n'
            result += seg.get_trans_str()
        return result

    def form_bilingual_str(self):
        result = ""
        for i, seg in enumerate(self.segments):
            result += f'{i + 1}\n'
            result += seg.get_bilingual_str()
        return result

    def write_srt_file_src(self, path: str):
        # write srt file to path
        with open(path, "w", encoding='utf-8') as f:
            f.write(self.reform_src_str())
        pass

    def write_srt_file_translate(self, path: str):
        self.task_logger.info("writing to " + path)
        with open(path, "w", encoding='utf-8') as f:
            f.write(self.reform_trans_str())
        pass

    def write_srt_file_bilingual(self, path: str):
        self.task_logger.info("writing to " + path)
        with open(path, "w", encoding='utf-8') as f:
            f.write(self.form_bilingual_str())
        pass

    def realtime_write_srt(self, path, range, length, idx):
        # DEPRECATED
        start_seg_id = range[0]
        end_seg_id = range[1]
        with open(path, "a", encoding='utf-8') as f:
            # for i, seg in enumerate(self.segments[start_seg_id-1:end_seg_id+length]):
            #     f.write(f'{i+idx}\n')
            #     f.write(seg.get_trans_str())
            for i, seg in enumerate(self.segments):
                if i < range[0] - 1: continue
                if i >= range[1] + length: break
                f.write(f'{i + idx}\n')
                f.write(seg.get_trans_str())
        pass

    def realtime_bilingual_write_srt(self, path, range, length, idx):
        # DEPRECATED
        start_seg_id = range[0]
        end_seg_id = range[1]
        with open(path, "a", encoding='utf-8') as f:
            for i, seg in enumerate(self.segments):
                if i < range[0] - 1: continue
                if i >= range[1] + length: break
                f.write(f'{i + idx}\n')
                f.write(seg.get_bilingual_str())
        pass

def split_script(script_in, chunk_size=1000):
    script_split = script_in.split('\n\n')
    script_arr = []
    range_arr = []
    start = 1
    end = 0
    script = ""
    for sentence in script_split:
        if len(script) + len(sentence) + 1 <= chunk_size:
            script += sentence + '\n\n'
            end += 1
        else:
            range_arr.append((start, end))
            start = end + 1
            end += 1
            script_arr.append(script.strip())
            script = sentence + '\n\n'
    if script.strip():
        script_arr.append(script.strip())
        range_arr.append((start, len(script_split) - 1))

    assert len(script_arr) == len(range_arr)
    return script_arr, range_arr
