"""
IMPORT LINES
"""
from pathlib import Path

from language_models import BertModel, QwenModel, LlamaModel, DeepSeekModel
from data_organization import load_cloze_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_list_words_given_sentence(my_list):
    words = []
    for datum_tuple in my_list:
        # my schema data in my dictionary is {sentence_number: [(word, cloze_prob), (word, cloze_prob), ...], ...}
        words.append(datum_tuple[0])

    return words


# main function to run the pipeline
def main():
    # load the data in the peelee data cloze dataset
    file_path = PROJECT_ROOT / "data" / "peelle_data" / "cloze_data.csv"

    masked, sentences = load_cloze_data(file_path)

    model = BertModel()
    for idx, sentence in enumerate(sentences, start=1):
        word_list = get_list_words_given_sentence(masked[str(idx)])
        sentence_token_ids = model.tokenize_sentense(sentence)

        for word in word_list:
            word_token_ids = model.tokenize_word(word)
            if not word_token_ids:
                 continue
            # get the prob numbers
            prob = model.predict_next_word(sentence_token_ids, word_token_ids)
            if prob is None:
                 continue
            # write in the output file 
            with open(model.get_ouptut_file(), 'a') as f:
                f.write(f"{idx},'{sentence}',{word},{prob}\n")


if __name__ == "__main__":
    main()
