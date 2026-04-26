import pandas as pd


def load_cloze_data(file_path: str):
    '''
    this function parses a csv and file and returns a list of dictionaries,
    where each dictionary represents a row in the csv file.
    '''
    data = pd.read_csv(file_path, sep=',', quotechar='"', skipinitialspace=True, dtype=str, keep_default_na=False)  # Read the CSV file into a DataFrame
    data = data.to_dict(orient='records')

    '''
    this function takes a list of dictionaries and returns a list of masked words
    that are human-generated.
    '''

    # variables declaration
    masked_words = {}
    sentences = []

    # data schema: sentence_number,sentence,word,cloze_prob

    # loop through the data and extract the masked words and sentences
    for row in data:
        if row['sentence_number'] not in masked_words:
            masked_words[row['sentence_number']] = [tuple([row["word"], row["cloze_prob"]])]
            sentences.append(row['sentence'].strip("\"'")) #remove the leading and trailing quotes from the sentence to avoid tha they aloso get captured as part of the sentence as well. 

        else:
            masked_words[row['sentence_number']].append(tuple([row["word"], row["cloze_prob"]]))

    return masked_words, sentences

    # masked words schema: {sentence_number: [(word, cloze_prob), (word, cloze_prob), ...], ...}
    # sentences schema: [sentence1, sentence2, ...]
