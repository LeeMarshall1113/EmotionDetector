# EmotionDetector

A text emotion classifier. It fine-tunes a DistilBERT model on the GoEmotions
dataset to predict which of 28 emotions (admiration, amusement, anger,
annoyance, approval, caring, confusion, curiosity, desire, disappointment,
disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy,
love, nervousness, optimism, pride, realization, relief, remorse, sadness,
surprise, neutral) best describes a piece of text, and gives a percentage
breakdown across all of them.

## How it works

- `emotions_trainer.py` loads the `go_emotions` dataset via Hugging Face
  `datasets`, reduces each example's multi-label annotation to a single
  label (the first label in the list), and fine-tunes
  `distilbert-base-uncased` as a 28-class sequence classifier using the
  Hugging Face `Trainer` API (3 epochs, batch size 16, learning rate 2e-5).
  Accuracy is computed with the `evaluate` library. The resulting model and
  tokenizer are saved to `./go_emotions_model`.
- `Emotions_LLM_use.py` loads a model and tokenizer from `./go_emotions_model`,
  tokenizes input text, runs inference, and applies softmax to the output
  logits to produce a percentage score for every one of the 28 emotions,
  printed from highest to lowest.

## Requirements

- Python 3 with the packages listed in `requirements.txt` (PyTorch,
  Hugging Face `transformers`, `datasets`, and `evaluate`)
- A CUDA-capable GPU is used automatically if available (`torch.cuda.is_available()`),
  otherwise the code falls back to CPU
- A trained model directory named `go_emotions_model` (produced by running
  `emotions_trainer.py`) must exist alongside `Emotions_LLM_use.py` before
  that script can run. The README's original author reported training taking
  about 3 hours on an RTX 3070; time will vary with GPU power.

## Usage

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Train the model (downloads the GoEmotions dataset and fine-tunes
   DistilBERT, saving the result to `./go_emotions_model`):

   ```
   python emotions_trainer.py
   ```

3. Run the interactive emotion detector against the trained model:

   ```
   python Emotions_LLM_use.py
   ```

   Type a message and press Enter to see its emotion breakdown. Type
   `exit` or `quit` to stop.

## Notes / limitations

- `emotions_trainer.py` collapses GoEmotions' multi-label annotations to a
  single label (the first one listed) per example, which is a simplification
  of the original multi-label dataset.
- `Emotions_LLM_use.py` will not run until a `go_emotions_model` directory
  exists; it is not included in the repository and must be produced by
  running the trainer first.
