# Hello World
A sort of hello world interpretability folder. This is basically a standalone project that you can imagine as behaving like its own repository. It includes dependencies and whatnot as required and basically just was my initial foray into Whisper's internal mechanisms.

TODO what sort of challenge can we do... other than reverse or that is more doable?

Let's figure it out bottom-up to begin with:
1. First task of the day (all ablations should be in two flavors: zero and data swap: with data swap we will pick a different random element from the `hf-internal-testing/librispeech_asr_dummy` dataset which we will use as a starter pack; we will run through every data point from this dataset, and for and random ablation we'll just pick a random swap `n=1` times for now; because there are around 73 rows, this should take around 146 forward passes, each taking around 1-2 seconds => around 2-5 minutes)
    - For each encoder layer ablate it (just call generate and see what comes out: put this all in some big text file to be able to compare )
    - For each decoder layer ablate it (same as above)
    - 
2. Be able to get the same performance from a raw pytorch model and my own generation loop. Use this to be able to transform the model into a transformer_lens format with `HookPoint` modules. Ideally we can just use the github directly from OAI. We should just use `transformer_lens` as a dependency and then put hookpoints in a new model and we should define a method to make it loadable. Then we should test that it is possible to run with hooks and make sure to make the ablations work for ^^^
3. See if we can copy https://github.com/soniajoseph/ViT-Prisma/blob/main/src/sae/main.py to leave an SAE just running overnight. Kind of important to try and run some simple SAEs over night.

Things that might be interesting ask:
- Is the model doing any sort of language modeling or just audio modeling? They claim that hallucinations can occur because it is conflating predicting the next word with transcribing audio. It's not at first clear that this should be possible since the model only uses cross-attention where there is text and there is no path backwards through any sort of self attention. In fact the final representation is completely generated
    - This seems like it should be a pretty straightforward case of just doing basic patching analysis. We could delete some of the blocks for the audio and also compare the magnitudes of the activation values for the residual and regular paths to get a notion for how much of the audio network is actually used.
    - We can do a sort of logit lens, where we could try and decompose for each cross attention, the component that is comparing to each stage of the encoder?

# Things to do in the future
- Set up audio streaming so that it's easy to quickly listen to a bunch of relevant clips
    - https://www.reddit.com/r/commandline/comments/thqow3/playing_audio_over_ssh_while_connected_remotely/ (seems pretty straightforward and you just need to set up a audio server locally that can be written to over SSH via a connection to the server, from the server)
- Might want to read
    - CTC Loss: https://paperswithcode.com/method/ctc-loss#:~:text=A%20Connectionist%20Temporal%20Classification%20Loss,series%20and%20a%20target%20sequence.
    - FT this shit: https://huggingface.co/blog/fine-tune-whisper


# REOPENME
https://github.com/soniajoseph/ViT-Prisma/blob/main/src/sae/main.py
https://github.com/TransformerLensOrg/TransformerLens
https://claude.ai/chat/5c824c7c-7cb6-498c-86b7-5a91df2678ea
https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy
https://drive.google.com/drive/home
https://www.runpod.io/console/pods
https://cdn.openai.com/papers/whisper.pdf
https://github.com/openai/whisper
https://huggingface.co/openai/whisper-small
https://huggingface.co/blog/fine-tune-whisper
https://github.com/4gatepylon/WhisperInterpretability
https://github.com/4gatepylon/TrainingTemperatureExperiments/tree/main
https://openai.com/index/whisper/
https://www.lesswrong.com/posts/bCtbuWraqYTDtuARg/towards-multimodal-interpretability-learning-sparse-2#Examples_of_SAE_Features