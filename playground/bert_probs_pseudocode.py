# Pseudocode for getting next word probabilities from BERT. 
# This is not a real implementation, just a sketch of how it could be done.

# ------ Necessary imports ------
import torch
import torch.nn.functional as F

# pip install transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ------ Load the model and tokenizer ------
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-large-uncased-whole-word-masking")

# ------ Function to get next word probabilities ------
def get_next_word_probability_distribution(sentence):
    '''return the probs of all next tokens/words wiht their associated probs gdiven"
    the input sentence with a [MASK] token at the end. '''
    # Returns all next tokens and their probabilities.
    inputs = tokenizer(sentence + " [MASK].", return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    mask_token_logits = logits[0, mask_token_index, :].squeeze()
    probs = torch.softmax(mask_token_logits, dim=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(range(len(mask_token_logits)))

    # Return the results as a list of (token, probability) tuples
    return list(zip(predicted_tokens, probs.tolist()))

# ------ Helper function to extract specific word probabilities from BERT ------
def get_specific_word_probability(word, next_word_probabilities):
    for token, prob in next_word_probabilities:
        if token == word:
            return prob
    return 0.0

    # TODO: HERE IS THE TODO FOR THIS PROJECT
    # Use Peelle data and BERT probability distribution function 
    # For each sentence in Peellee data, put a [MASK] at the end. Then pass this to BERT function. 
    # Then you have a distribution over the entire vocab for that sentence. Write a function that searches this 
    # distribution for only the words/specific sentence from Peelle and gets the word and the bert_prob.
    # When passing sentence to BERT, make sure it has this structure <space>[MASK]<period> - “He hated bees and 
    # feared encountering a [MASK].”
    # Results dataframe: Sentence, word, human_cloze, bert_prob
    # Sentence, word, human_cloze - from peelle data. 
    # Push to git. 
    # Put paper notes in lit review doc. 



# Pseudocode pipeline 
# for sentence in Peelle_data:
#     bert_prompt = sentence + " [MASK]."
#     bert_next_word_probabilities = get_next_word_probability_distribution(bert_prompt)

#     # Search bert_next_word_probabilities to get only the words that are associated with the sentence from Peelle_data. 

# df_results = pd.DataFrame(columns=["sentence", "word", "human_cloze_probability", "bert_probability"])