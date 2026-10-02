import json
import pandas as pd
import os

total_gpu_time = 0
results_folder = "../eval_results"

all_results = pd.DataFrame(columns=["ModelName","ModelCheckpoint","ModelParams","Task","MetricName","MetricValue"])

for folder in os.listdir(results_folder):
    current_folder_path = results_folder + "/" + folder
    if os.path.isdir(current_folder_path):
        for file in os.listdir(current_folder_path):
            if "results_" in file:
                current_file_path = current_folder_path + "/" + file
                with open(current_file_path, "r") as f:
                    current_results_str = f.read()
                    current_results=json.loads(current_results_str)

                total_gpu_time += float(current_results["total_evaluation_time_seconds"])
                model_name = current_results["model_name_sanitized"]
                checkpoint = current_results["config"]["model_revision"]
                model_params = current_results["config"]["model_num_parameters"]

                for result in current_results["results"]:
                    for metric_name in current_results["results"][result]:
                        metric_name_cleaned = metric_name.replace(",none","")
                        metric_value = current_results["results"][result][metric_name]
                        current_result = pd.DataFrame({"ModelName":[model_name],
                                                    "ModelCheckpoint":[checkpoint],
                                                    "ModelParams": [model_params],
                                                    "Task":[result],
                                                    "MetricName":[metric_name_cleaned],
                                                    "MetricValue":[metric_value]})
                        all_results = pd.concat([all_results,current_result]).reset_index(drop=True).drop_duplicates()


all_results["ModelName"] = all_results["ModelName"]
all_results.to_csv("../results_merged/eval_results.tsv",sep="\t",index=False)

with open("../results_merged/total_time.txt","w") as f:
    f.write("{}".format(total_gpu_time))
