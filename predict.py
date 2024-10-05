# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import shutil
import subprocess
import time
import torch
from cog import BasePredictor, Input, Path
from melo.api import TTS

from openvoice import se_extractor
from openvoice.api import ToneColorConverter


SUPPORTED_LANGUAGES = ["EN_NEWEST", "EN", "ES", "FR", "ZH", "JP", "KR"]


MODEL_URL = "https://weights.replicate.delivery/default/myshell-ai/OpenVoice-v2.tar"
MODEL_CACHE = "model_cache"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        ckpt_converter = f"{MODEL_CACHE}/checkpoints_v2/converter"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tone_color_converter = ToneColorConverter(
            f"{ckpt_converter}/config.json", device=self.device
        )
        self.tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

    def predict(
        self,
        audio: Path = Input(description="Input reference audio"),
        text: str = Input(
            description="Input text",
            default="Did you ever hear a folk tale about a giant turtle?",
        ),
        language: str = Input(
            description="The language of the audio to be generated",
            choices=SUPPORTED_LANGUAGES,
            default="EN_NEWEST",
        ),
        speed: float = Input(
            description="Set speed scale of the output audio", default=1.0
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        target_dir = "exp_dir"
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        target_se, audio_name = se_extractor.get_se(
            str(audio),
            self.tone_color_converter,
            target_dir=f"{target_dir}/processed",
            vad=False,
        )

        model = TTS(language=language, device=self.device)
        speaker_ids = model.hps.data.spk2id

        src_path = f"{target_dir}/tmp.wav"
        out_path = "/tmp/out.wav"

        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace("_", "-")

            source_se = torch.load(
                f"{MODEL_CACHE}/checkpoints_v2/base_speakers/ses/{speaker_key}.pth",
                map_location=self.device,
            )
            model.tts_to_file(text, speaker_id, src_path, speed=speed)

            # Run the tone color converter
            encode_message = "@MyShell"
            self.tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=out_path,
                message=encode_message,
            )

        return Path(out_path)


# from cog import BasePredictor, Input, Path
# from melo.api import TTS
# from openvoice.api import ToneColorConverter
# import os
# import shutil
# import subprocess
# import time
# import torch

# SUPPORTED_LANGUAGES = ["EN_NEWEST", "EN", "ES", "FR", "ZH", "JP", "KR"]

# MODEL_URL = "https://weights.replicate.delivery/default/myshell-ai/OpenVoice-v2.tar"
# MODEL_CACHE = "model_cache"


# def download_weights(url, dest):
#     start = time.time()
#     print("downloading url: ", url)
#     print("downloading to: ", dest)
#     subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
#     print("downloading took: ", time.time() - start)


# class Predictor(BasePredictor):
#     def setup(self) -> None:
#         """Load the model into memory to make running multiple predictions efficient"""
#         if not os.path.exists(MODEL_CACHE):
#             download_weights(MODEL_URL, MODEL_CACHE)

#         ckpt_converter = f"{MODEL_CACHE}/checkpoints_v2/converter"
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         self.tone_color_converter = ToneColorConverter(
#             f"{ckpt_converter}/config.json", device=self.device
#         )
#         self.tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

#     def predict(
#         self,
#         text: str = Input(
#             description="Input text",
#             default="Did you ever hear a folk tale about a giant turtle?",
#         ),
#         language: str = Input(
#             description="The language of the audio to be generated",
#             choices=SUPPORTED_LANGUAGES,
#             default="EN_NEWEST",
#         ),
#         speaker_key: str = Input(
#             description="Speaker key for voice synthesis",
#             default="default_speaker",
#         ),
#         speed: float = Input(
#             description="Set speed scale of the output audio", default=1.0
#         ),
#     ) -> Path:
#         """Run a single prediction on the model"""

#         target_dir = "exp_dir"
#         if os.path.exists(target_dir):
#             shutil.rmtree(target_dir)

#         model = TTS(language=language, device=self.device)
#         speaker_ids = model.hps.data.spk2id

#         src_path = f"{target_dir}/tmp.wav"
#         out_path = "/tmp/out.wav"

#         # Use the provided speaker key to load the source speaker embedding
#         speaker_id = speaker_ids.get(speaker_key.lower().replace("_", "-"), None)

#         if speaker_id is None:
#             raise ValueError(f"Speaker key '{speaker_key}' not found.")

#         source_se = torch.load(
#             f"{MODEL_CACHE}/checkpoints_v2/base_speakers/ses/{speaker_key}.pth",
#             map_location=self.device,
#         )

#         model.tts_to_file(text, speaker_id, src_path, speed=speed)

#         # Run the tone color converter without a target embedding
#         encode_message = "@MyShell"
#         self.tone_color_converter.convert(
#             audio_src_path=src_path,
#             src_se=source_se,
#             tgt_se=source_se,  # Use source embedding as target for default conversion
#             output_path=out_path,
#             message=encode_message,
#         )

#         return Path(out_path)
