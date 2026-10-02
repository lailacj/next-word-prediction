import os
import sys

# Add the parent directory (playground) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csv_parser import *
from bert_code import *

def main():
    # Step 1: Parse the CSV file to get human-generated masked data
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(base_dir, "data", "peelle_data", "cloze_data.csv")
    output_path = os.path.join(base_dir, "data", "bert_data", "bert_cloze_data.csv")
    peelle_data = parse_csv(file_path)

    # extract data from the csv
    masked_words, sentences = get_human_masked_data(peelle_data)

    # Step 2: For each sentence in Peelle data, put a [MASK] at the end and pass this to BERT function
  
    for idx, sentence in enumerate(sentences, start=1):
        bert_next_word_probabilities = get_next_word_probability_distribution(sentence)

        # get the human words and their probabilities for the current sentence
        human_words_distribution = masked_words[str(idx)]
        human_words = [word for word, prob in human_words_distribution]

        get_specific_word_probability(idx, sentence, human_words, bert_next_word_probabilities, output_path)

if __name__ == "__main__":
    main()



