# !pip install fairseq2
# !pip install pydub sentencepiece
# !pip install git+https://github.com/facebookresearch/seamless_communication.git
# !pip install zhconv
import numpy
import torchaudio
import torch

from IPython.display import Audio, display
import zhconv

from seamless_communication.inference import Translator

class Seamless_model():
    def __init__(self):
        self.model_name = "seamlessM4T_v2_large"
        self.vocoder_name = "vocoder_v2" if self.model_name == "seamlessM4T_v2_large" else "vocoder_36langs"
        self.translator = Translator(self.model_name, self.vocoder_name,
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            dtype = torch.float16,
        )

    def text2text(self, text):
        text_output, _ = self.translator.predict(  # supported language = ("spa", "fra", "deu", "ita", "hin", "cmn")
            input = text,
            task_str="t2tt",
            src_lang="eng",
            tgt_lang="cmn"
            )
        return zhconv.convert(str(text_output[0]), "zh-tw")

    def text2audio(self, text):
        _ , speech_output = self.translator.predict(
            input = text,
            task_str="t2st",
            tgt_lang="cmn",
            src_lang="cmn",
        )
        return speech_output

    def audio2text(self, audio):
        text_output, _ = self.translator.predict(
        input=audio,
        task_str="s2tt",
        tgt_lang="cmn",
        )
        return zhconv.convert(str(text_output[0]), "zh-tw")

translator = Seamless_model()
test = translator.text2text("Hello, welcome to Taiwan")
print(test)

sound_test = translator.text2audio("你好，很高興認識你")
audio_play = Audio(sound_test.audio_wavs[0][0].to(torch.float32).cpu(), rate = sound_test.sample_rate, autoplay=False, normalize=True)
display(audio_play)

audio2text_test = translator.audio2text(sound_test.audio_wavs[0][0])
print(audio2text_test)
