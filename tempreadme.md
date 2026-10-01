Next Word Prediction Project Overview

Both humans and LLMs make next word predictions. In humans next word predictions can be measured through cloze probability tasks, reading times, reaction times, N400, surprisal, and priming. In LLMs, next word predictions are measured through logits.

There are a few open questions here:
Which, if any, LLMs capture human next word predictions? How correlated are human cloze probabilities and LLM log probabilities.
Under what conditions are LLMs good at predicting the next word?
In humans, cloze probabilities, N400, and reading times are all highly correlated. So, if an LLM is correlated to human cloze probability then that LLM should be correlated with N400 and reading times. Is this actually the case? If not why?
Holding a LLM constant, are the human measures correlated? For example, what is the correlation between Qwen logits and human cloze? Qwen and N400? Qwen and reaction times?
Does the LLM parameter size make a difference?

Existing work – Human prediction is multifaceted: cloze, N400, reading times, priming, and RTs are related but not interchangeable because they reflect different tasks, timescales, and mechanisms. Ryskin & Nieuwland frame prediction as adaptive and dependent on memory, attention, and experience, not just raw word probability. Larger/better LMs seem to better predict N400 amplitude, but not reading time (Michaelov & Bergen, 2026). Oh & Schuler also show that larger transformer surprisal can fit reading times worse, with an apparent sweet spot around about two billion training tokens. Meanwhile, “Clozing the Gap” argues that LM surprisal can outperform cloze because cloze is low-resolution, especially for low-probability and semantically similar alternatives.

The central question for the project: Do LLM next-word probabilities become more human-like with scale, or are they measure-specific, better for some human signals, worse for others? The project tests whether human next-word prediction is one thing or several. For each language model, we ask:
Does model log probability correlate with human cloze probability?
Does model surprisal predict N400 amplitude?
Does model surprisal predict reading time / RT / priming?
Are those relationships moderated by model size, word frequency, cloze entropy, semantic similarity, and item type?
Does cloze mediate the model-human relationship, or do LLMs explain N400/RT variance beyond cloze?

A kind of takeaway we want may look like: Better next-word prediction in LLMs does not necessarily mean more human-like language processing. Scaling LLMs may improve alignment with N400/semantic preactivation while reducing alignment with reading-time behavior, especially for rare words, named entities, low-cloze continuations, and high-entropy contexts.

Two main parts of the project: 1) Literature Review on reading times, cloze prob, reaction times, priming, surprisal theory, N400, and LLM logits. 2) Computational Pipeline.
