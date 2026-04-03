from language_models import QwenModel
from data_organization import load_cloze_data


def get_list_words_given_sentence(my_list):
    words = []
    for datum_tuple in my_list:
        # my schema data in my dictionary is {sentence_number: [(word, cloze_prob), (word, cloze_prob), ...], ...}
        words.append(datum_tuple[0])

    return words


def main():
    file_path = "../data/peelle_data/cloze_data.csv"
    masked, sentences = load_cloze_data(file_path)

    model = QwenModel()

    for idx, sentence in enumerate(sentences, start=1):
        word_list = get_list_words_given_sentence(masked[str(idx)])
        sentence_token_ids = model.tokenize_sentense(sentence)

        for word in word_list:
            word_token_ids = model.tokenize_word(word)
            if not word_token_ids:
                continue
            qwen_prob = model.predict_next_word(sentence_token_ids, word_token_ids)
            print(f"{idx},'{sentence}',{word},{qwen_prob}")


if __name__ == "__main__":
    main()
