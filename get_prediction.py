import os
import shutil
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
from icecream import ic


speed = 1.0
reference_speaker = "resources/demo_speaker2.mp3"  # This is the voice you want to clone
input_text = "The integration of AI into various industries has driven significant advancements in hardware technology. PC manufacturers are now offering AI-specific hardware to enhance end-user devices. Field-Programmable Gate Arrays (FPGAs) and Application-Specific Integrated Circuits (ASICs), such as Google's Tensor Processing Units (TPUs), are being utilized to accelerate machine learning workloads. These hardware solutions are particularly useful in edge computing scenarios, where AI capabilities need to be deployed close to the data source to reduce latency and improve decision-making. Microsoft has introduced Copilot+ PCs, which feature new silicon capable of performing over 40 trillion operations per second, enhancing AI capabilities such as real-time image generation and language translation. Apple has also integrated Apple silicon to handle advanced AI processing. These developments reflect the growing trend of AI processing moving closer to the user and the edge, rather than relying solely on cloud-based solutions."

ckpt_converter = "checkpoints_v2/converter"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = os.path.join(os.getcwd(), "outputs_test_v2")

# Ensure the output directory exists and is writable
try:
    os.makedirs(output_dir, exist_ok=True)
except OSError as e:
    ic(f"Error creating output directory: {e}")
    raise

# Remove the existing output directory if it exists
if os.path.exists(output_dir):
    try:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    except OSError as e:
        ic(f"Error removing/recreating output directory: {e}")
        raise

tone_color_converter = ToneColorConverter(
    f"{ckpt_converter}/config.json", device=device
)
tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

# Initialize target_se
target_se, audio_name = se_extractor.get_se(
    reference_speaker, tone_color_converter, vad=False
)

src_path = os.path.join(output_dir, "tmp.wav")
out_path = os.path.join(output_dir, "out.wav")  # Changed from /tmp/out.wav

model = TTS(language="EN_NEWEST", device=device)
# Retrieve the speaker IDs from the model's hyperparameters
speaker_ids = model.hps.data.spk2id
ic("Available speaker IDs:", speaker_ids)  # Log the available speaker IDs

for speaker_key in speaker_ids.keys():
    speaker_id = speaker_ids[speaker_key]
    speaker_key = speaker_key.lower().replace("_", "-")

    try:
        source_se = torch.load(
            f"checkpoints_v2/base_speakers/ses/{speaker_key}.pth",
            map_location=device,
        )
    except FileNotFoundError:
        ic(f"Speaker embedding file not found for {speaker_key}")
        continue

    try:
        model.tts_to_file(input_text, speaker_id, src_path, speed=speed)
    except Exception as e:
        ic(f"Error generating audio for speaker {speaker_key}: {str(e)}")
        continue

    # Run the tone color converter
    try:
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_path,
            message=encode_message,
        )
        ic(f"Successfully converted audio for speaker {speaker_key}")
    except Exception as e:
        ic(f"Error converting audio for speaker {speaker_key}: {str(e)}")
