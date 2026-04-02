import csv
import os
import sys

# Add the parent directory (playground) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_code import *
import csv_parser


def get_list_words_given_sentence(my_list):
    words = []
    for datum_tuple in my_list:
        # my schema data in my dictionary is {sentence_number: [(word, cloze_prob), (word, cloze_prob), ...], ...}
        words.append(datum_tuple[0])

    return words


def main():
    # Step 1: Parse the CSV file to get human-generated masked data
    file_path = "../../data/bert_data/bert_cloze_data.csv"
    output_path = "../../data/deepseek/deepseek_data.csv"
    peelle_data = csv_parser.parse_csv(file_path)
    masked, sentences = csv_parser.get_human_masked_data(peelle_data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 2: for each sentence in the data, compute the log prob of the words associated with that sentence
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence_number", "sentence", "word", "deepseek_log_prob"])

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

                deepseek_prob = get_word_probabilities(sentence_token_ids.clone(), word_token_ids)
                writer.writerow([idx, sentence, word, deepseek_prob])


if __name__ == "__main__":
    main()