import os
import shutil
import torch
from typing import Optional, Dict, Tuple
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
from icecream import ic


class OpenVoiceTTS:
    def __init__(
        self, reference_speaker: str = "resources/demo_speaker2.mp3", speed: float = 1.1
    ):
        self.reference_speaker = reference_speaker
        self.speed = speed
        self.ckpt_converter = "checkpoints_v2/converter"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output_dir = os.path.join(os.getcwd(), "outputs_test_v2")
        self._setup_output_directory()
        self._initialize_models()

    def _setup_output_directory(self) -> None:
        try:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)
        except OSError as e:
            ic(f"Error setting up output directory: {e}")
            raise

    def _initialize_models(self) -> None:
        self.tone_color_converter = ToneColorConverter(
            f"{self.ckpt_converter}/config.json", device=self.device
        )
        self.tone_color_converter.load_ckpt(f"{self.ckpt_converter}/checkpoint.pth")
        self.target_se, _ = se_extractor.get_se(
            self.reference_speaker, self.tone_color_converter, vad=False
        )
        self.model = TTS(language="EN_NEWEST", device=self.device)
        self.speaker_ids = self.model.hps.data.spk2id
        ic("Available speaker IDs:", self.speaker_ids)

    def generate_audio(self, input_text: str) -> Optional[str]:
        src_path = os.path.join(self.output_dir, "tmp.wav")
        out_path = os.path.join(self.output_dir, "out.wav")

        for speaker_key, speaker_id in self.speaker_ids.items():
            speaker_key = speaker_key.lower().replace("_", "-")
            source_se = self._load_source_se(speaker_key)
            if source_se is None:
                continue

            if not self._generate_tts(input_text, speaker_id, src_path):
                continue

            if self._convert_tone_color(src_path, source_se, out_path):
                return out_path

        return None

    def _load_source_se(self, speaker_key: str) -> Optional[torch.Tensor]:
        try:
            return torch.load(
                f"checkpoints_v2/base_speakers/ses/{speaker_key}.pth",
                map_location=self.device,
            )
        except FileNotFoundError:
            ic(f"Speaker embedding file not found for {speaker_key}")
            return None

    def _generate_tts(self, input_text: str, speaker_id: int, src_path: str) -> bool:
        try:
            self.model.tts_to_file(input_text, speaker_id, src_path, speed=self.speed)
            return True
        except Exception as e:
            ic(f"Error generating audio for speaker {speaker_id}: {str(e)}")
            return False

    def _convert_tone_color(
        self, src_path: str, source_se: torch.Tensor, out_path: str
    ) -> bool:
        try:
            encode_message = "@MyShell"
            self.tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=self.target_se,
                output_path=out_path,
                message=encode_message,
            )
            ic(f"Successfully converted audio")
            return True
        except Exception as e:
            ic(f"Error converting audio: {str(e)}")
            return False


def get_prediction(input_text: str) -> Optional[str]:
    tts = OpenVoiceTTS()
    output_path = tts.generate_audio(input_text)
    if output_path:
        print(output_path)
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = "Hello, how are you today?"
    output_path = get_prediction(text)
    ic(output_path)
