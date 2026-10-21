"""Audio alignment and muxing for game captures (no game or sound device needed)."""

import subprocess

import imageio_ffmpeg
import numpy as np

from cuphead_rl.game.capture import align_audio, mux_audio


def test_align_audio_pads_or_trims():
    x = np.ones((100, 2), np.float32)
    late = align_audio(x, audio_start=10.5, video_start=10.0, samplerate=100)  # audio began 0.5 s late
    assert late.shape == (150, 2) and not late[:50].any() and late[50:].all()
    early = align_audio(x, audio_start=9.8, video_start=10.0, samplerate=100)  # 0.2 s before the video
    assert early.shape == (80, 2)


def test_mux_audio_adds_a_track_of_the_video_length(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=64x64:r=30:d=2", "-pix_fmt", "yuv420p", str(video)], check=True,
                   stdin=subprocess.DEVNULL)
    sr = 48000
    tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)  # 1 s, shorter than the video
    mux_audio(video, np.stack([tone, tone], axis=1), sr)
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(video)], capture_output=True, text=True).stderr
    assert "Audio: aac" in probe and "Video: h264" in probe
    assert "Duration: 00:00:02.0" in probe
    assert not (tmp_path / "v.audio.wav").exists()
