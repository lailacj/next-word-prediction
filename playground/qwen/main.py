from qwen_code import get_word_probabilities

def main():
    sentence = "We are the world we are the"
    word = "children"
    print("WE JUST STARTED COOKING")

    # compute the probability of the target word being the next word given the sentence
    result = get_word_probabilities(sentence, word)
    print(f"Probability of '{result['word']}' being the next word, given the sentence '{result['sentence']}' is: {result['prob']:.6f}, and the raw (log probability is: {result['log_prob']:.6f}")    


if __name__ == "__main__":
    main()