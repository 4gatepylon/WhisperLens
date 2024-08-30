# %%
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import json

# %%
# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
# %%
print(model) # what's the architecture?
# %%
# load dummy dataset and read audio files
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
# %%
print("*"*100 + "\n" + "Sample")
print(type(sample)) # dict
print(sample.keys()) # ['path', 'array', 'sampling_rate']
print("Path:", sample['path'])
print("Array Type Type:", type(sample["array"])) # numpy.ndarray
print("Array Type shape:", sample["array"].shape) # (length_)
print("Array dtype:", sample["array"].dtype, ", Array Max, Min:", sample["array"].max(), sample["array"].min()) # why is the max/min so constrained?
# This should mean that it can get up to around 8 kHz OK, but upwards I am dubious...
# print("Sampling Rate:", sample["sampling_rate"]) # 16000 -> probably just how many times samples were taken per second?
# print(type(sample["array"])) # numpy.ndarray
# %%
# Processor allegedly turns this into a log-mel spectrogram, meaning that likely this is a set of frequency-per-timestep values
# Need to look at: WhisperFeatureExtractor
# 80 = num_features
# 3000 ~= (approx) num_measurements / chunk_size (i.e. seems to be the chunks, since this shit seems to be chunked)
#   (sanity check, if it was around 16K/sec, then the reduction by around 30 -> a few hundred per sec ->
#    definately realistic for what a human hears)
# NOTE
# 1. This input is unit mean and variance normalized (TODO: check) SOMETIMES
# 2. The input gets padded (TODO: how much? for this case what is cfg?)
# 3. Core function is `_torch_extract_fbank_features`
#   3.1 Hann window smooth (https://en.wikipedia.org/wiki/Hann_function)
#   3.2 Fourier Transform in overlapping windows
#   3.3 Mel features (seems to be a transform of x, i.e. the freq. axis): `mel_filter_bank`` in `audio_utils.py``
#   3.4 Log in base 10
#   3... NOTE that they only seem to use the aplitudes, and not so much the phases...
# TODO(Adriano) In the fourier transform and elsewhere, WTF are:
# feature_size=80,
# sampling_rate=16000,
# hop_length=160,
# chunk_length=30,
# n_fft=400,
# padding_value=0.0,
#
# Read more:
#  - https://en.wikipedia.org/wiki/Hann_function
#  - https://pytorch.org/docs/stable/generated/torch.hann_window.html
#  - https://pytorch.org/docs/stable/generated/torch.stft.html
#  - Code to understand the Mel transform better
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 
# %%
print("*"*100 + "\n" + "Input Features")
print(type(input_features)) # tensor
print(input_features.shape) # B x 
# print(input_features)
# %%
# generate token ids
predicted_ids = model.generate(input_features, language='en') # TODO(Adriano) en language required?
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
print("Transcription, NOT skipping special tokens")
print(transcription)
# %%
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print("Transcription, skipping special tokens")
print(transcription)
# You should see that it looks roughly correct from what we see:
# https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy
# %%
print(len(ds))
sample = ds[1]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 
predicted_ids = model.generate(input_features, language='en') # TODO(Adriano) en language required?
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
print("Transcription, NOT skipping special tokens", transcription)
# %%
################ Try Evaluation ################
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load
# Should start with at least 200GB free to begin with on disk for this notebook

# Requires >= 30GB
librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("cuda")

# This shit doesn't fit (requires >= 30GB)
def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = processor.tokenizer._normalize(batch['text'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch

result = librispeech_test_clean.map(map_to_pred)

wer = load("wer")
print(100 * wer.compute(references=result["reference"], predictions=result["prediction"]))

# %%
print(type(model).__mro__)
print(json.dumps([(n, ','.join(map(str, p.shape))) for n, p in model.named_parameters()], indent=4))
# %%
