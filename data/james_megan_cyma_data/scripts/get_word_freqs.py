import pandas as pd
import wordfreq


with open("../data/critical_words.txt", "r") as f:
    words = f.read().splitlines()
    
wordfreqs = []
numtoks =[]

for word in words:
    wordfreqs.append(wordfreq.zipf_frequency(word,"en","large"))
    numtoks.append(len(wordfreq.tokenize(word, "en")))
        
df = pd.DataFrame({"TargetWords":words,"ZipfFrequency":wordfreqs,"NumTokens":numtoks})

df.to_csv("../data/wordfreqs.csv", index=False)
