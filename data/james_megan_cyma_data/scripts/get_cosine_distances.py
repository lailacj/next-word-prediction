import numpy as np
from scipy.spatial.distance import cdist
import re
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Calculates cosine similarity \
                                    between a word and its context given word embeddings.')

    parser.add_argument('--stimuli', '-i', type=str,
                        help='path to stimuli to test')
    parser.add_argument('--stimuli_list', '-ii', type=str,
                        help='path to file containing list of paths to stimulus files to test')
    parser.add_argument('--output_directory','-o', type=str, required = True,
                        help='output directory')
    parser.add_argument('--embeddings','-m', type=str,
                        help='path to embeddings to use')
    parser.add_argument('--embeddings_list','-mm', type=str,
                        help='path to file with a list of paths to embeddings to use')
    parser.add_argument('--following_context', '-f', action="store_true", default=False,
                        help='include the following context of target words as context')
    parser.add_argument('--try_subwords_in_context', '-cs', action="store_true", default=False,
                        help='if input word(s) not present in context, try to decompose words \
                        with punctuation in them into their subwords')
    parser.add_argument('--ignore_oov_in_context', '-cv', action="store_true", default=False,
                        help='if input word(s) not present in context, calculate similarity \
                        without them (note: if --try_subwords argument is used, \
                        this applies to subwords too).')
    parser.add_argument('--try_subwords_in_target', '-ts', action="store_true", default=False,
                        help='if input word(s) not present in target, try to decompose words \
                        with punctuation in them into their subwords')
    parser.add_argument('--ignore_oov_in_target', '-tv', action="store_true", default=False,
                        help='if input word(s) not present in target, calculate similarity \
                        without them (note: if --try_subwords argument is used, \
                        this applies to subwords too).')
    parser.add_argument('--uncased', action="store_true", default=False,
                        help='use if embeddings are uncased - converts all embeddings\
                        to lowercase.')
    args = parser.parse_args()
    return args

def process_args(args):

    try:
        output_directory = args.output_directory
    except:
        print("Error: Please specify a valid output directory.")

    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
        except:
            print("Error: Cannot create output directory (Note: output directory does not already exist).")
    
    if args.stimuli_list:
        try:
            assert os.path.exists(args.stimuli_list)
            with open(args.stimuli_list, "r") as f:
                stimuli_list = f.read().splitlines()
        except:
            print("Error: 'stimuli_list' argument does not have a valid path. Trying to use individual stimulus set.")
            try:
                assert args.stimuli
                stimuli_list = [args.stimuli]
            except:
                print("Error: No stimuli specified")
    else:
        try:
            assert args.stimuli
            stimuli_list = [args.stimuli]
        except:
            print("Error: No stimuli specified")  
            
    
    if args.embeddings_list:
        try:
            assert os.path.exists(args.embeddings_list)
            with open(args.embeddings_list, "r") as f:
                embeddings_list = f.read().splitlines()
        except:
            print("Error: 'embeddings_list' argument does not have a valid path. Trying to use individual specified embeddings.")
            try:
                assert args.embeddings
                embeddings_list = [args.embeddings]
            except:
                print("Error: No embeddings specified")
    else:
        try:
            assert args.embeddings
            embeddings_list = [args.embeddings]
        except:
            print("Error: No embeddings specified")  
            
    return stimuli_list,embeddings_list,output_directory
            
def extract_embeddings_from_file(path):    
    # Get number of lines
    with open(path) as f:
        line_count = 0
        for line in f:
            line_count+=1
            if line_count==1:
                first_line = line.splitlines()[0]
            elif line_count==2:
                second_line = line.splitlines()[0]

    # Calculate vector length
    vec_length = len(second_line.split()[1:])

    # Process lines of file
    with open(path) as f:
        word_count =0
        # test if first line is header and adjust accordingly
        if not len(first_line.split())==len(second_line.split()):
            line_count-=1
            next(f)
        word_dict = dict()
        vector_array = np.zeros((line_count,vec_length),dtype="float64")
        for line in f:
            word = " ".join(line.rstrip("\n").split()[:-vec_length]) # relies on token on second line not involving a space
            word_dict[word] = word_count
            vec = line.rstrip("\n").split()[-vec_length:] # relies on token on second line not involving a space
            vector_array[word_count,:] = vec
            word_count+=1
    return word_dict,vector_array


def split_sentence(sentence,following_context):
    sentence_list = sentence.split("*")
    target = sentence_list[1]
    if following_context==False:
        context = sentence_list[0]
    elif following_context==True:
        context = sentence_list[0] + " " + sentence_list[2]
        context = context.replace("   "," ")
        context = context.replace("  "," ")
    return context,target


def get_sequence_embeddings(sequence,word_dict,vector_array,subwords=False,ignore_oov=False):
    sequence_list = sequence.split()
    vector_list = []
    for i in range(len(sequence_list)):
        current_word = sequence_list[i]
        if current_word in word_dict.keys():
            current_word_vector = vector_array[word_dict[current_word]] 
            vector_list.append(current_word_vector)
        else:
            if subwords==False and ignore_oov==False:
                return None
            elif subwords==False and ignore_oov==True:
                pass
            elif subwords==True:
                subword_list = re.findall(r'\W*\w*',current_word) # gets non-word + word combinations (e.g. "'s")
                subword_list.remove("")
                if len(subword_list)==np.sum([subword in word_dict.keys() for subword in subword_list]):
                    for subword in subword_list:
                        vector_list.append(vector_array[word_dict[subword]])
                else:
                    subword_list = re.findall(r'\w+|\W+',current_word) # gets all word or nonword seqs
                    for subword in subword_list:
                        if subword in word_dict.keys():
                            vector_list.append(vector_array[word_dict[subword]])
                        else:
                            if ignore_oov==False:
                                return None
                            elif ignore_oov==True:
                                pass
    return np.array(vector_list).mean(0)

def cosine_similarity(current_sentence,word_dict,vector_array,args):
    context,target = split_sentence(current_sentence,following_context=args.following_context)
    context_vector = get_sequence_embeddings(context,word_dict,vector_array,subwords=args.try_subwords_in_context,ignore_oov=args.ignore_oov_in_context)
    target_vector = get_sequence_embeddings(target,word_dict,vector_array,subwords=args.try_subwords_in_target,ignore_oov=args.ignore_oov_in_target)
    if isinstance(context_vector,np.ndarray) and isinstance(target_vector,np.ndarray):
        cosine_similarity = 1-cdist([context_vector],[target_vector], 'cosine')
        return cosine_similarity[0,0]
    else:
        return None

def get_embeddings(args):
    stimuli_list,embeddings_list,output_directory = process_args(args)
    for i in range(len(embeddings_list)):
        current_embedding_path = embeddings_list[i]
        current_embedding_name = ".".join(current_embedding_path.split('/')[-1].split('.')[:-1])
        current_embedding_name = current_embedding_name.replace(".","-")
        word_dict,vector_array = extract_embeddings_from_file(current_embedding_path)
        for j in range(len(stimuli_list)):
            current_stimuli_path = stimuli_list[j]
            current_stimuli_name = ".".join(current_stimuli_path.split('/')[-1].split('.')[:-1])
            current_stimuli_name = current_stimuli_name.replace(".","-")
            output_file_name = output_directory + "/" + current_stimuli_name + "." + current_embedding_name.replace("-","_")
            with open(output_file_name,"w") as f:
                f.write("FullSentence\tTargetWords\tCosineSimilarity\n")
            with open(current_stimuli_path) as f:
                current_stimuli_list = f.read().splitlines()
            for k in range(len(current_stimuli_list)):
                current_stimulus = current_stimuli_list[k]
                full_sentence = current_stimulus.replace("*","")
                if args.uncased == True:
                    current_stimulus = current_stimulus.lower()
                stimulus_cosine_similarity = cosine_similarity(current_stimulus,word_dict,vector_array,args)
                with open(output_file_name,"a") as f:
                    f.write("{0}\t{1}\t{2}\n".format(
                    full_sentence,
                    current_stimulus.split("*")[1],
                    str(stimulus_cosine_similarity).replace("None","")))
        
def main():
    args = parse_args()
    get_embeddings(args)

if __name__ == "__main__":
    main()
