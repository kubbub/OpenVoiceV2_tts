import os
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from icecream import ic
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from typing import Dict, Tuple
import re


def fetch_and_save_configs() -> Dict[str, str]:
    ic("Checking for config files...")
    config_dir = Path("")
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
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    configs = fetch_and_save_configs()
    base_speaker_tts, tone_color_converter = initialize_models(configs, device)

    ic("Loading source speaker embedding...")
    source_se = torch.load(configs["en_default_se"]).to(device)

    # Use default speaker embedding directly
    target_se = source_se  # Use the same embedding for both source and target

    # Split text into sentences
    sentences = re.split("(?<=[.!?]) +", text)

    all_audio_paths = []
    for i, sentence in enumerate(sentences):
        ic(f"Generating TTS for sentence {i+1}/{len(sentences)}")
        tmp_path = f"{output_dir}/tmp_{i}.wav"
        base_speaker_tts.tts(
            sentence, tmp_path, speaker="default", language="English", speed=1.0
        )
        all_audio_paths.append(tmp_path)

    # Combine all audio files
    combined_path = f"{output_dir}/{output_name}_combined.wav"
    combine_audio_files(all_audio_paths, combined_path)

    # Convert tone color for the combined audio
    save_path = f"{output_dir}/{output_name}.wav"
    encode_message = "@MyShell"
    ic("Converting tone color...")
    tone_color_converter.convert(
        audio_src_path=combined_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message,
    )

    # Clean up temporary files
    for path in all_audio_paths + [combined_path]:
        os.remove(path)

    ic("Process completed. Output saved to:", save_path)
    return save_path


def combine_audio_files(audio_paths, output_path):
    import wave

    data = []
    for audio_path in audio_paths:
        w = wave.open(audio_path, "rb")
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()

    output = wave.open(output_path, "wb")
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()


if __name__ == "__main__":
    text_to_speak = """
    Italy's northern regions are facing an outbreak of African swine fever,
    which has significant implications for the production of prized prosciutto. 
    The disease was first detected in late August,
    and efforts are underway to contain the spread and mitigate the impact on the local pork industry.
    """
    output_filename = "italy_swine_fever_report"
    generated_file = generate_voice(text_to_speak, output_filename)
    print(f"Generated audio file: {generated_file}")
