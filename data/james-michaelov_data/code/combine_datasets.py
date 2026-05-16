import pandas as pd
import os
import csv
import numpy as np
from tqdm import tqdm

stim_folder = "../stimuli_details/"
surprisal_folder = "../results/"

stims_filenames = [x for x in os.listdir(stim_folder) if ".tsv" in x]

surprisal_folder_filenames = [x for x in os.listdir(surprisal_folder) if ".tsv" in x]

for i in tqdm(range(len(stims_filenames))):
    stim_filename = stims_filenames[i]
    reduced_stim_name = stim_filename.split("_")[0]
    print("Combining data for: {}".format(stim_filename.split(".")[0]))
    stim_file_path = stim_folder + "/" + stim_filename
    stims = pd.read_csv(stim_file_path,sep="\t",doublequote=False,escapechar=None,quoting=csv.QUOTE_NONE)
    stims=stims.rename({"FullSentence":"FullText"},axis=1)
    for j in tqdm(range(len(surprisal_folder_filenames))):
        surprisal_folder_file = surprisal_folder_filenames[j]
        model_name = "___".join([surprisal_folder_file.split("___")[1],surprisal_folder_file.split("___")[2]])
        if reduced_stim_name in surprisal_folder_file:
            current_path = surprisal_folder + "/" + surprisal_folder_file
            current_data = pd.read_csv(current_path,sep="\t",doublequote=False,escapechar=None,quoting=csv.QUOTE_NONE)
            current_data = current_data[["FullText","Surprisal"]].rename({"Surprisal":"LMSurprisal__{}".format(model_name)},axis=1)
            stims = pd.merge(left=stims,right=current_data,how="left").drop_duplicates().dropna()
    if ("luke" in reduced_stim_name) or ("smith" in reduced_stim_name) or \
    ("kennedy" in reduced_stim_name) or ("futrell" in reduced_stim_name):
        stims = stims.drop(columns=["FullText"])
        
    stims.to_csv("../results_merged/{}".format(stim_filename),sep="\t",doublequote=False,escapechar=None,quoting=csv.QUOTE_NONE,index=False)