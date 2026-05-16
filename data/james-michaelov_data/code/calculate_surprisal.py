import os
import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoConfig
import torch
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Calculates surprisal and other \
                                    metrics (in development) of transformers language models')

    parser.add_argument('--stimuli', '-i', type=str, default=None,
                        help='stimuli to test')
    parser.add_argument('--stimuli_list', '-ii', type=str, default=None,
                        help='line-separated list of stimuli to test')
    parser.add_argument('--output_directory','-o', type=str, default="results",
                        help='output directory')
    parser.add_argument('--model','-m', type=str, default="gpt2",
                        help='select a model to use')
    parser.add_argument('--checkpoint','-c', type=str, default="main",
                        help='select a model to use')
    parser.add_argument('--dtype', type=str, default="auto",
                        help='select a model to use')
    parser.add_argument('--eos_as_bos', action='store_true', default=False,
                        help='use eos token as bos token (make sure this makes sense before selecting)')
    parser.add_argument('--truncate_context', action='store_true', default=False,
                        help='limit context length to fit into the context window (2048 if none specified in model config)')
    args = parser.parse_args()
    return args



class MetricCalculator():
    def __init__(self,args):
        self.model_name=args.model
        self.model_name_cleaned = self.model_name.replace("/","__")
        self.calculate_tokens=args.calculate_tokens
        self.eos_as_bos = args.eos_as_bos
        self.output_directory = args.output_directory
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)
        self.strip_critical_words = args.strip_critical_words
        self.output_type = args.output_type
        self.checkpoint=args.checkpoint
        self.dtype=args.dtype
        self.truncate_context = args.truncate_context
        
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        if not hasattr(args,"tokenizer"):
            self.tokenizer_name = self.model_name
        else:
            self.tokenizer_name = args.tokenizer
        if not hasattr(args,"config"):
            self.config_name = self.model_name
        else:
            self.config_name = args.config
        
        self.determine_model_type()
        self.create_model()
            
    
    def determine_model_type(self):
        try:
            self.model_config = AutoConfig.from_pretrained(self.model_name)
            causal_architectures = ["causal", "lmhead", "bloom"]
            for causal_architecture in causal_architectures:
                if causal_architecture in self.model_config.architectures[0].lower():
                    self.model_type="autoregressive"
            causal_types = ["gpt","bloom","xglm","llama","mistral","qwen","deepseek"]
            for causal_type in causal_types:
                if causal_type in self.model_config.model_type.lower():
                    self.model_type="autoregressive" 
            for causal_type in causal_types:
                if causal_type in self.model_name:
                    self.model_type="autoregressive" 
            assert self.model_type == "autoregressive"
        except:
            raise Exception("Model needs to be autoregressive/causal") 
            
    def create_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        enc1=self.tokenizer.encode("1")
        enc2=self.tokenizer.encode("2")
        if enc1[0]==enc2[0]:
            self.tokenizer.add_bos_token = True
            if self.tokenizer.bos_token_id != enc1[0]:
                self.tokenizer.bos_token_id = enc1[0]
        else:
            self.tokenizer.add_bos_token = False
        if self.tokenizer.bos_token is None and self.eos_as_bos==True and self.tokenizer.eos_token is not None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
        self.tokenizer.add_special_tokens({"additional_special_tokens":["[!StimulusMarker!]"]})
        self.tokenizer.stimulus_marker_idx = self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index("[!StimulusMarker!]")]
        if self.model_type=="autoregressive":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,revision=self.checkpoint,torch_dtype=self.dtype).to(self.device)
        config_dict = self.model_config.to_dict()
        if self.truncate_context == False:
            self.max_len = torch.inf
        else:
            if "n_positions" in config_dict:
                self.max_len = config_dict["n_positions"]
            elif "max_position_embeddings" in config_dict:
                self.max_len = config_dict["max_position_embeddings"]
            elif "max_seq_len" in config_dict:
                self.max_len = config_dict["max_position_embeddings"]
            else:
                self.max_len = 2048
        
            
    def get_metrics(self,stimulus_list):
        for stimulus_set in stimulus_list:
            current_stimulus_set = None
            stim_name = stimulus_set.split("/")[-1].split("\\")[-1]
            stim_name_split = stim_name.split(".")
            sn_ext = stim_name_split[-1]
            if sn_ext==".stims" or sn_ext==".txt" or sn_ext==".tsv":
                stim_name = ".".join(stim_name_split[:-1])
            with open(stimulus_set,"r") as f:
                firstline = f.readline()
            firstline_split = firstline.split("\t")
            if ("PrecedingContext" in firstline_split) and ("CriticalWords" in firstline_split):
                current_stimulus_set = pd.read_csv(stimulus_set,sep="\t")
                self.run_current_stimuli_tsv(current_stimulus_set,stim_name)
            else:
                with open(stimulus_set,"r") as f:
                    current_stimulus_set = f.read().splitlines()
                self.run_current_stimuli_list(current_stimulus_set,stim_name)
            
            
    def run_current_stimuli_list(self,current_stimulus_set,stim_name):
        if self.calculate_tokens=="default":
            tok_calc_type = "critical_word"
        stim_file_path = self.output_directory + "/" + stim_name + "___" + self.model_name_cleaned + "___" + self.checkpoint + "____"+tok_calc_type+".tsv"
        stim_df = pd.DataFrame(columns=["FullText","CriticalWords","Surprisal","NumTokens"])
        stim_df.to_csv(stim_file_path,sep="\t",mode="w",header=True,index=False)
        for i in range(len(current_stimulus_set)):
            current_stimulus = current_stimulus_set[i]
            current_stimulus_rows = self.get_sentence_surprisals(current_stimulus)
            current_stimulus_rows.to_csv(stim_file_path,sep="\t",mode="a",header=False,index=False)
    
    def get_sentence_surprisals(self,current_stimulus):
        full_text = current_stimulus.replace("*","")
        
        sentence = current_stimulus
        sentence_cleaned = sentence.replace("*","")
        
        encoded_sentence_no_markers = self.tokenizer.encode(sentence_cleaned,return_tensors="pt")
        first_token = max(0,len(encoded_sentence_no_markers[0])-(self.max_len-4))
        encoded_sentence_no_markers = encoded_sentence_no_markers[:,first_token:]
        if encoded_sentence_no_markers[0,0]!= self.tokenizer.bos_token_id:
            encoded_sentence_no_markers = torch.cat((torch.LongTensor([[self.tokenizer.bos_token_id]]),encoded_sentence_no_markers),dim=-1)
        encoded_sentence_no_markers = encoded_sentence_no_markers[0]

        if self.calculate_tokens=="default":
            sentence_marked = sentence.replace(" *","* ")
            sentence_marked = sentence_marked.replace("*","[!StimulusMarker!]")
        else:
            raise Exception("Invalid 'calculate_tokens' argument")

        encoded_sentence_with_markers = self.tokenizer.encode(sentence_marked,return_tensors="pt")
        encoded_sentence_with_markers = encoded_sentence_with_markers[:,first_token:]
        if encoded_sentence_with_markers[0,0]!= self.tokenizer.bos_token_id:
            encoded_sentence_with_markers = torch.cat((torch.LongTensor([[self.tokenizer.bos_token_id]]),encoded_sentence_with_markers),dim=-1)
        encoded_sentence_with_markers = encoded_sentence_with_markers[0]

        marker_locations = torch.where(encoded_sentence_with_markers==self.tokenizer.stimulus_marker_idx)[-1]
        for i in range(1,len(marker_locations)):
            marker_locations[i:]= marker_locations[i:]-1
            
        if len(marker_locations)%2==0:
            pass
        else:
            raise Exception("Stimulus incorrectly formatted: {}\nPlease review README.".format(sentence))  
            
        marker_locations = torch.reshape(marker_locations.clone(),(int(marker_locations.shape[0]/2),2))
        
        if self.calculate_tokens=="word":
            for i in range(1,len(marker_locations)):
                marker_locations[i:]= marker_locations[i:]-1


        stimulus_rows = pd.DataFrame(columns=["FullText","CriticalWords","Surprisal","NumTokens"])

        
        for i in range(len(marker_locations)):
            start_token,end_token = marker_locations[i]
            critical_word = self.tokenizer.decode(encoded_sentence_no_markers[start_token:end_token])
            surprisals = self.get_raw_surprisals(sentence_cleaned,first_token)[1][start_token-1:end_token-1]
            surprisal=surprisals.sum().item()
            num_token = len(surprisals)
            if self.strip_critical_words:
                critical_word = critical_word.strip()

            item_row = pd.DataFrame({"FullText":[full_text],
                                     "CriticalWords":[critical_word],
                                     "Surprisal":[surprisal],
                                     "NumTokens":[num_token]})
            stimulus_rows = pd.concat([stimulus_rows.astype(item_row.dtypes),item_row])
        return stimulus_rows
    
    def get_raw_surprisals(self,current_stimulus_text,first_token=0):
        encoded = self.tokenizer.encode(current_stimulus_text,return_tensors="pt").to(self.device)
        encoded = encoded[:,first_token:]
        if encoded[0,0]!= self.tokenizer.bos_token_id:
            encoded = torch.cat((torch.LongTensor([[self.tokenizer.bos_token_id]]).to(self.device),encoded),dim=-1)
        with torch.no_grad():
            logits = self.model(encoded).logits
        surprisals = -logits.softmax(-1).log()[:, :-1, :]
        all_surprisals = torch.gather(surprisals, -1, encoded[:, 1:, None]).squeeze(-1).flatten()
        return encoded.flatten(),all_surprisals



def main():
    args = parse_args()

    args.task_list = ["surprisal"]
    args.calculate_tokens = "default"
    args.output_type = "default" 
    args.strip_critical_words = True
    if args.stimuli_list is not None:
        with open(args.stimuli_list,"r") as f:
            stim_list = f.read().splitlines()
    elif args.stimuli is not None:
        stim_list = [args.stimuli]
    else:
        print("Invalid stimulus file path(s)")
    obj = MetricCalculator(args)
    obj.get_metrics(stim_list)


if __name__ == "__main__":
    main()
