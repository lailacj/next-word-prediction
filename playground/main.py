### FILES IMPORTS
import csv_parser
from bert_probs_pseudocode import get_next_word_probability_distribution, get_specific_word_probability


def main():
    # Step 1: Parse the CSV file to get human-generated masked data
    file_path = '/Users/dondestiniriho/Desktop/BLT lab/next-word-prediction/data/peelle_data/cloze_data.csv'  # Update this path to your actual CSV file
    output_path = '/Users/dondestiniriho/Desktop/BLT lab/next-word-prediction/data/bert_data/bert_cloze_data.csv'  # Path to save the BERT results
    peelle_data = csv_parser.parse_csv(file_path)
    masked_words, sentences = csv_parser.get_human_masked_data(peelle_data)

    # Step 2: For each sentence in Peelle data, put a [MASK] at the end and pass this to BERT function

    for indx, sentence in enumerate(sentences, start=1):
        bert_next_word_probabilities = get_next_word_probability_distribution(sentence)

        # get the human words and their probabilities for the current sentence
        human_words_distribution = masked_words[str(indx)]
        human_words = [word for word, prob in human_words_distribution]

        get_specific_word_probability(indx, sentence, human_words, bert_next_word_probabilities, output_path)
        
    
    # def get_specific_word_probability(sentence_num, sentence 
    #                               ,word_list, next_word_probabilities):


    # test on the first sentecne to see if it works
    # sentence = sentences[0]
    # bert_next_word_probabilities = bert_probs_pseudocode.get_next_word_probability_distribution(sentence)

    # human_words_distribution = masked_words[indx]
    # human_words = [word for word, prob in human_words_distribution]
    # bert_data = {}

    # for token, prob in bert_next_word_probabilities:
    #     if token in human_words:
    #         bert_data[token] = prob

    # print(bert_data)
    

if __name__ == "__main__":
    main()



