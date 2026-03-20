import sys
import os

# Add the parent directory (playground) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_code import get_word_probabilities
import csv_parser

def get_list_words_given_sentence(my_list):
    words = []
    for datum_tuple in my_list:
        #my schema data in my dictionary is {sentence_number: [(word, cloze_prob), (word, cloze_prob), ...], ...}
        words.append(datum_tuple[0])

    return words


def main():
     # Step 1: Parse the CSV file to get human-generated masked data
    file_path = '../../data/peelle_data/cloze_data.csv'  # Update this path to your actual CSV file
    output_path = '../../data/qwen/qwen_data.csv'  # Path to save the qwen results
    peelle_data = csv_parser.parse_csv(file_path)
    masked, sentences = csv_parser.get_human_masked_data(peelle_data)
  
    # write the headers in the output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("sentence_number,sentence,word,qwen_log_prob\n")

    # Step 2: for each sentence in pelle data, compute the log prob of the words associated with that sentennce
    for idx, sentence in enumerate(sentences, start=1):
        # initiliize the word_list for targeted words extracted from the peelle-data
        word_list = get_list_words_given_sentence(masked[str(idx)])
        
        # tokenize both the sentence 
        sentenc_token_ids = tokenize_sentence(sentence)

        #iterate through the word list and compute the log prob of each word given the sentence
        for word in word_list:
            #tokenize the word into a tensor space of token ids
            word_token_ids = tokenize_word(word)
            
            #computer the probabiliry of this word given the tokenize_sentence space
            qwen_prob = get_word_probabilities(sentenc_token_ids, word_token_ids)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(f"{idx},{sentence},{word},{qwen_prob}\n")


if __name__ == "__main__":
    main()