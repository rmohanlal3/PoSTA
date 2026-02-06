spectrogram = fastpitch.generate_spectrogram(
    tokens=parsed,
    pace=0.9,   # slower = calmer
    pitch=0.95  # slightly lower = soothing
)
