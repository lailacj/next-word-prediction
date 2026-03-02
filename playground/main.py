### FILES IMPORTS
import csv_parser
import bert_probs_pseudocode


def main():
    # Step 1: Parse the CSV file to get human-generated masked data
    file_path = '/Users/dondestiniriho/Desktop/BLT lab/next-word-prediction/data/peelle_data/cloze_data.csv'  # Update this path to your actual CSV file
    peelle_data = csv_parser.parse_csv(file_path)
    masked_words, sentences = csv_parser.get_human_masked_data(peelle_data)

    # Step 2: For each sentence in Peelle data, put a [MASK] at the end and pass this to BERT function
    
    indx = '1'
    # for sentence in sentences:
    #     bert_next_word_probabilities = bert_probs_pseudocode.get_next_word_probability_distribution(sentence)
    ## test on the first sentecne to see if it works
    sentence = sentences[0]
    bert_next_word_probabilities = bert_probs_pseudocode.get_next_word_probability_distribution(sentence)

    human_words_distribution = masked_words[indx]
    human_words = [word for word, prob in human_words_distribution]
    bert_data = {}

    for token, prob in bert_next_word_probabilities:
        if token in human_words:
            bert_data[token] = prob

    print(bert_data)
    

if __name__ == "__main__":
    main()



