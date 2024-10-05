import os
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from icecream import ic
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from typing import Dict, Tuple
import re


def get_config_paths() -> Dict[str, str]:
    config_dir = Path("configs/checkpoints/checkpoints")
    return {
        "base_config": str(config_dir / "base_speakers/EN/config.json"),
        "base_checkpoint": str(config_dir / "base_speakers/EN/checkpoint.pth"),
        "converter_config": str(config_dir / "converter/config.json"),
        "converter_checkpoint": str(config_dir / "converter/checkpoint.pth"),
        "en_default_se": str(config_dir / "base_speakers/EN/en_default_se.pth"),
    }


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

    configs = get_config_paths()
    base_speaker_tts, tone_color_converter = initialize_models(configs, device)

    ic("Loading source speaker embedding...")
    source_se = torch.load(configs["en_default_se"]).to(device)

    # Use default speaker embedding directly
    target_se = source_se  # Use the same embedding for both source and target

    # Split text into sentences
    sentences = re.split("(?<=[.!?]) +", text)

    all_audio_paths = []
    for i, sentence in enumerate(sentences):
        ic("Generating TTS for sentence:", i + 1, "/", len(sentences))
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
    The integration of AI into various industries has driven significant advancements in hardware technology. PC manufacturers are now offering AI-specific hardware to enhance end-user devices. Field-Programmable Gate Arrays (FPGAs) and Application-Specific Integrated Circuits (ASICs), such as Google's Tensor Processing Units (TPUs), are being utilized to accelerate machine learning workloads. These hardware solutions are particularly useful in edge computing scenarios, where AI capabilities need to be deployed close to the data source to reduce latency and improve decision-making. Microsoft has introduced Copilot+ PCs, which feature new silicon capable of performing over 40 trillion operations per second, enhancing AI capabilities such as real-time image generation and language translation. Apple has also integrated Apple silicon to handle advanced AI processing. These developments reflect the growing trend of AI processing moving closer to the user and the edge, rather than relying solely on cloud-based solutions.
    """
    output_filename = "tech_report"
    generated_file = generate_voice(text_to_speak, output_filename)
    ic("Generated audio file: ", generated_file)
