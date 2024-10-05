import os
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from icecream import ic
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from typing import Dict, Tuple


def fetch_and_save_configs() -> Dict[str, str]:
    ic("Checking for config files...")
    config_dir = Path("configs/checkpoints")
    config_dir.mkdir(parents=True, exist_ok=True)

    config_files = {
        "base_config": (
            "myshell-ai/OpenVoice",
            "checkpoints/base_speakers/EN/config.json",
        ),
        "base_checkpoint": (
            "myshell-ai/OpenVoice",
            "checkpoints/base_speakers/EN/checkpoint.pth",
        ),
        "converter_config": (
            "myshell-ai/OpenVoice",
            "checkpoints/converter/config.json",
        ),
        "converter_checkpoint": (
            "myshell-ai/OpenVoice",
            "checkpoints/converter/checkpoint.pth",
        ),
        "en_default_se": (
            "myshell-ai/OpenVoice",
            "checkpoints/base_speakers/EN/en_default_se.pth",
        ),
    }

    downloaded_files = {}
    for key, (repo_id, filename) in config_files.items():
        local_path = config_dir / Path(filename).name
        if local_path.exists():
            ic(f"File {key} already exists locally: {local_path}")
            downloaded_files[key] = str(local_path)
        else:
            ic(f"Downloading {key} from Hugging Face Hub...")
            try:
                downloaded_files[key] = hf_hub_download(
                    repo_id=repo_id, filename=filename, local_dir=config_dir
                )
                ic(f"Downloaded {key}: {downloaded_files[key]}")
            except Exception as e:
                ic(f"Error downloading {key}: {str(e)}")
                raise

    return downloaded_files


def initialize_models(
    configs: Dict[str, str], device: str
) -> Tuple[BaseSpeakerTTS, ToneColorConverter]:
    ic("Initializing BaseSpeakerTTS...")
    base_speaker_tts = BaseSpeakerTTS(configs["base_config"], device=device)
    base_speaker_tts.load_ckpt(configs["base_checkpoint"])

    ic("Initializing ToneColorConverter...")
    tone_color_converter = ToneColorConverter(
        configs["converter_config"], device=device
    )
    tone_color_converter.load_ckpt(configs["converter_checkpoint"])

    return base_speaker_tts, tone_color_converter


def generate_voice(
    text: str,
    output_name: str,
    reference_speaker: str = "resources/example_reference.mp3",
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    configs = fetch_and_save_configs()
    base_speaker_tts, tone_color_converter = initialize_models(configs, device)

    ic("Loading source speaker embedding...")
    source_se = torch.load(configs["en_default_se"]).to(device)

    ic("Extracting target speaker embedding...")
    target_se, audio_name = se_extractor.get_se(
        reference_speaker, tone_color_converter, target_dir="processed", vad=True
    )
    save_path = f"{output_dir}/{output_name}.wav"

    # Run the base speaker tts
    src_path = f"{output_dir}/tmp.wav"
    ic("Generating base TTS...")
    base_speaker_tts.tts(
        text, src_path, speaker="default", language="English", speed=1.0
    )

    # Run the tone color converter
    encode_message = "@MyShell"
    ic("Converting tone color...")
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message,
    )

    ic("Process completed. Output saved to:", save_path)
    return save_path


if __name__ == "__main__":
    text_to_speak = "Italy's northern regions are facing an outbreak of African swine fever, which has significant implications for the production of prized prosciutto. The disease was first detected in late August, and efforts are underway to contain the spread and mitigate the impact on the local pork industry."
    output_filename = "output_en_default"
    generated_file = generate_voice(text_to_speak, output_filename)
    print(f"Generated audio file: {generated_file}")
