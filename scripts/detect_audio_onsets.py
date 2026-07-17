"""
Phase 2C (audio branch): Footwork audio onset detection.
Deliberately independent of the pose/landmark pipeline -- this reads a video's
audio track directly and never touches MediaPipe output. The idea (per the
tatkaar pilot plan): foot strikes produce audio transients that onset
detection can find even when visual foot tracking is unreliable or unrecorded.

Not yet validated against real ground truth -- no tatkaar clips have been
recorded. This script has only been smoke-tested (runs without crashing,
produces plausible-looking onsets) on non-footwork audio.

Usage:
    python detect_audio_onsets.py --video data/raw/tatkaar/tatkaar_01.mov
"""

import argparse
import os
import subprocess

import imageio_ffmpeg
import librosa


def extract_audio(video_path: str, out_wav: str, sr: int = 22050):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", str(sr), "-ac", "1", out_wav],
        check=True, capture_output=True,
    )


def detect_onsets(wav_path: str):
    y, sr = librosa.load(wav_path, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets_sec = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time", backtrack=True)
    return onsets_sec


def main():
    parser = argparse.ArgumentParser(description="Detect audio onsets (candidate foot-strike timestamps) in a clip")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output-dir", default="data/features", help="Output directory for onset CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(args.video))[0]
    wav_path = os.path.join(args.output_dir, f"{clip_name}_audio.wav")
    csv_path = os.path.join(args.output_dir, f"{clip_name}_onsets.csv")

    print(f"Extracting audio from {args.video}...")
    extract_audio(args.video, wav_path)

    print("Detecting onsets...")
    onsets = detect_onsets(wav_path)

    with open(csv_path, "w") as f:
        f.write("onset_time_s\n")
        for t in onsets:
            f.write(f"{t:.4f}\n")

    print(f"Found {len(onsets)} onsets, saved to {csv_path}")
    os.remove(wav_path)


if __name__ == "__main__":
    main()
