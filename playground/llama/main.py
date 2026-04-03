import os
import sys

# Add the parent directory (playground) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_code import *

from csv_parser import *

def get_list_words_given_sentence(my_list):
    words = []
    for datum_tuple in my_list:
        # my schema data in my dictionary is {sentence_number: [(word, cloze_prob), (word, cloze_prob), ...], ...}
        words.append(datum_tuple[0])

    return words


def main():
    # Step 1: Parse the CSV file to get human-generated masked data
    file_path = "../../data/peelle_data/cloze_data.csv"
    output_path = "../../data/llama/llama_data.csv"
    peelle_data = parse_csv(file_path)
    masked, sentences = get_human_masked_data(peelle_data)
    
    # Step 2: for each sentence in the data, compute the log prob of the words associated with that sentence
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("sentence_number,sentence,word,llama_log_prob\n")

        for idx, sentence in enumerate(sentences, start=1):
            # initialize the word list for targeted words extracted from the data
            word_list = get_list_words_given_sentence(masked[str(idx)])
           

            # tokenize the sentence once, then reuse a fresh copy for each candidate word
            sentence_token_ids = tokenize_sentence(sentence)

            # iterate through the word list and compute the log prob of each word given the sentence
            for word in word_list:
                word_token_ids = tokenize_word(word)

                if not word_token_ids:
                    continue

                llama_prob = get_word_probabilities(sentence_token_ids, word_token_ids)
                f.write(f"{idx},'{sentence}',{word},{llama_prob}\n")
                


if __name__ == "__main__":
    main()

