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
        self,
        reference_speaker: str = "resources/voice_female1.mp3",
        speed: float = 1.0,
        output_dir: str = "/Users/bub/Desktop/APPS/ERU_general/eru/backend/server/audio_management/audio_transcriptions",
    ):
        self.reference_speaker = reference_speaker
        self.speed = speed
        self.ckpt_converter = "checkpoints_v2/converter"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output_dir = output_dir
        self._setup_output_directory()
        self._initialize_models()
        ic("OpenVoiceTTS initialized")

    def _setup_output_directory(self) -> None:
        try:
            os.makedirs(self.output_dir, exist_ok=True)
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

    def generate_audio(self, input_text: str, file_name: str) -> Optional[str]:
        src_path = os.path.join(self.output_dir, "tmp.wav")
        out_path = os.path.join(self.output_dir, file_name)

        for speaker_key, speaker_id in self.speaker_ids.items():
            ic("Processing speaker", speaker_key)
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
            ic("Speaker embedding file not found", speaker_key)
            return None

    def _generate_tts(self, input_text: str, speaker_id: int, src_path: str) -> bool:
        try:
            self.model.tts_to_file(input_text, speaker_id, src_path, speed=self.speed)
            return True
        except Exception as e:
            ic("Error generating audio", str(e))
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
            ic("Audio conversion successful")
            return True
        except Exception as e:
            ic("Error converting audio", str(e))
            return False


def get_prediction(input_text: str, output_dir: str, file_name: str) -> Optional[str]:
    tts = OpenVoiceTTS(output_dir=output_dir)
    output_path = tts.generate_audio(input_text, file_name)
    if output_path:
        print(output_path)
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        text = sys.argv[1]
        output_dir = sys.argv[2]
        file_name = sys.argv[3]
    else:
        text = "No audio for you bro"
        output_dir = (
            "/Users/bub/Desktop/APPS/ERU_general/eru/backend/models/OpenVoiceV2_tts"
        )
        file_name = "out.wav"

    output_path = get_prediction(text, output_dir, file_name)
    ic("Output path", output_path)
