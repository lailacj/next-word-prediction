# Pseudocode for getting next word probabilities from BERT. 
# This is not a real implementation, just a sketch of how it could be done.

# ------ Necessary imports ------
import torch
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
def get_specific_word_probability(sentence_num, sentence 
                                  ,word_list, next_word_probabilities, outputfile_path):
    '''This function searches the BERT probability distribution for specific words and 
    stores the results in a dictionary.'''
    for token, prob in next_word_probabilities:
        if token not in word_list:
            continue
        else:
            with open(outputfile_path, "a", encoding="utf-8") as f:
                f.write(f"{sentence_num},'{sentence}',{token},{prob}\n")
               
