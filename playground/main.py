### FILES IMPORTS
import csv_parser
import bert_probs_pseudocode


def main():
    # Step 1: Parse the CSV file to get human-generated masked data
    # file_path = 'data/peelle_data/close_data.csv'  # Update this path to your actual CSV file
    # peelle_data = csv_parser.parse_csv(file_path)
    # masked_words, sentences = csv_parser.get_human_masked_data(peelle_data)

    # Step 2: For each sentence in Peelle data, put a [MASK] at the end and pass this to BERT function
    print(sorted(["Heal the", "world", "and", "make", "it a", "better"]))
    results = []
    prompt = "Heal the world and make it a better"

    bert_next_word_probabilities = bert_probs_pseudocode.get_next_word_probability_distribution(prompt)
    #print(bert_next_word_probabilities)

if __name__ == "__main__":
    main()



