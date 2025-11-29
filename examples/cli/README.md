# whisper.cpp/examples/cli

This is the main example demonstrating most of the functionality of the Whisper model.
It can be used as a reference for using the `whisper.cpp` library in other projects.

```
./build/bin/whisper-cli -h

usage: ./build/bin/whisper-cli [options] file0 file1 ...
supported audio formats: flac, mp3, ogg, wav
supported video formats (with -vi flag): mp4, mkv, avi, mov, webm (requires ffmpeg in PATH)

options:
  -h,        --help              [default] show this help message and exit
  -t N,      --threads N         [4      ] number of threads to use during computation
  -p N,      --processors N      [1      ] number of processors to use during computation
  -ot N,     --offset-t N        [0      ] time offset in milliseconds
  -on N,     --offset-n N        [0      ] segment index offset
  -d  N,     --duration N        [0      ] duration of audio to process in milliseconds
  -mc N,     --max-context N     [-1     ] maximum number of text context tokens to store
  -ml N,     --max-len N         [0      ] maximum segment length in characters
  -sow,      --split-on-word     [false  ] split on word rather than on token
  -bo N,     --best-of N         [5      ] number of best candidates to keep
  -bs N,     --beam-size N       [5      ] beam size for beam search
  -ac N,     --audio-ctx N       [0      ] audio context size (0 - all)
  -wt N,     --word-thold N      [0.01   ] word timestamp probability threshold
  -et N,     --entropy-thold N   [2.40   ] entropy threshold for decoder fail
  -lpt N,    --logprob-thold N   [-1.00  ] log probability threshold for decoder fail
  -nth N,    --no-speech-thold N [0.60   ] no speech threshold
  -tp,       --temperature N     [0.00   ] The sampling temperature, between 0 and 1
  -tpi,      --temperature-inc N [0.20   ] The increment of temperature, between 0 and 1
  -debug,    --debug-mode        [false  ] enable debug mode (eg. dump log_mel)
  -tr,       --translate         [false  ] translate from source language to english
  -di,       --diarize           [false  ] stereo audio diarization
  -tdrz,     --tinydiarize       [false  ] enable tinydiarize (requires a tdrz model)
  -nf,       --no-fallback       [false  ] do not use temperature fallback while decoding
  -otxt,     --output-txt        [false  ] output result in a text file
  -ovtt,     --output-vtt        [false  ] output result in a vtt file
  -osrt,     --output-srt        [false  ] output result in a srt file
  -olrc,     --output-lrc        [false  ] output result in a lrc file
  -owts,     --output-words      [false  ] output script for generating karaoke video
  -fp,       --font-path         [/System/Library/Fonts/Supplemental/Courier New Bold.ttf] path to a monospace font for karaoke video
  -ocsv,     --output-csv        [false  ] output result in a CSV file
  -oj,       --output-json       [false  ] output result in a JSON file
  -ojf,      --output-json-full  [false  ] include more information in the JSON file
  -of FNAME, --output-file FNAME [       ] output file path (without file extension)
  -np,       --no-prints         [false  ] do not print anything other than the results
  -ps,       --print-special     [false  ] print special tokens
  -pc,       --print-colors      [false  ] print colors
  -pp,       --print-progress    [false  ] print progress
  -nt,       --no-timestamps     [false  ] do not print timestamps
  -l LANG,   --language LANG     [en     ] spoken language ('auto' for auto-detect)
  -dl,       --detect-language   [false  ] exit after automatically detecting language
             --prompt PROMPT     [       ] initial prompt (max n_text_ctx/2 tokens)
  -m FNAME,  --model FNAME       [models/ggml-base.en.bin] model path
  -f FNAME,  --file FNAME        [       ] input audio file path
  -oved D,   --ov-e-device DNAME [CPU    ] the OpenVINO device used for encode inference
  -dtw MODEL --dtw MODEL         [       ] compute token-level timestamps
  -ls,       --log-score         [false  ] log best decoder scores of tokens
  -ng,       --no-gpu            [false  ] disable GPU
  -fa,       --flash-attn        [false  ] flash attention
  -sns,      --suppress-nst      [false  ] suppress non-speech tokens
  --suppress-regex REGEX         [       ] regular expression matching tokens to suppress
  --grammar GRAMMAR              [       ] GBNF grammar to guide decoding
  --grammar-rule RULE            [       ] top-level GBNF grammar rule name
  --grammar-penalty N            [100.0  ] scales down logits of nongrammar tokens

Video Input options:
  -vi,       --video-input       [false  ] enable video input (extract audio using ffmpeg)
```

## Video Input Support

The whisper-cli tool supports processing video files (MP4, MKV, AVI, MOV, WebM, etc.) by automatically extracting and converting the audio track using FFmpeg. This feature requires FFmpeg to be installed and available in your system PATH.

### Prerequisites

Make sure FFmpeg is installed on your system:

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Usage

To transcribe a video file, use the `-vi` or `--video-input` flag:

```bash
# Transcribe audio from a video file
./build/bin/whisper-cli -m models/ggml-base.en.bin -vi -f video.mp4

# With subtitle output
./build/bin/whisper-cli -m models/ggml-base.en.bin -vi -osrt -f video.mp4

# With language auto-detection
./build/bin/whisper-cli -m models/ggml-base.en.bin -vi -l auto -f video.mp4

# Translate audio from a video file
./build/bin/whisper-cli -m models/ggml-base.bin -vi  -f ./examples/cli/story.mp4
```

### How It Works

When the `-vi` flag is enabled and a video file is provided:

1. **Video Detection**: The tool detects video files based on their extension (mp4, mkv, avi, mov, webm, flv, wmv, m4v, mpeg, mpg, 3gp)
2. **Audio Extraction**: FFmpeg is invoked to extract the audio track and convert it to:
   - 16kHz sample rate (required by Whisper)
   - Mono channel
   - 16-bit PCM WAV format
3. **Transcription**: The converted audio is processed by Whisper
4. **Cleanup**: Temporary audio files are automatically cleaned up after processing

### Supported Video Formats

With the `-vi` flag, the following video formats are supported:

| Format | Extensions |
|--------|------------|
| MP4    | .mp4, .m4v |
| Matroska | .mkv |
| AVI    | .avi |
| QuickTime | .mov |
| WebM   | .webm |
| Flash Video | .flv |
| Windows Media | .wmv |
| MPEG   | .mpeg, .mpg |
| 3GPP   | .3gp |

### Notes

- If you provide a video file without the `-vi` flag, the tool will display a warning message suggesting to use the flag.
- The FFmpeg conversion creates a temporary WAV file in the same directory as the input video, which is automatically deleted after processing.
- The audio extraction uses the following FFmpeg settings for optimal Whisper compatibility:
  ```bash
  ffmpeg -y -i input.mp4 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
  ```
