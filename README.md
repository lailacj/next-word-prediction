# next-word-prediction

## TODO: 
- file path to run (one at a time: change it on *line 20 in run_pipeline.py*):
    > ../data/parsed_data/michaelov_2024.csv
    > ../data/parsed_data/nieuwland_2018.csv
    > ../data/parsed_data/szewczyk_2022.csv


- for each file, run the model, one at a time, and after running all the model for each (*run_pipeline.py, like 23*).
    
- Right now, each model is set up to write the data of the `../data/parsed_data/michaelov_2024.csv` dataset (this one is ready to run), but after it has finished running, you have to manually change the output file path in each model to write the data into their appropriate output file directory

    >the path schema looks like this:
        ../data/Model_michaelov/model_name/model_name_data.csv
        ../data/Model_nieuwland/model_name/model_name_data.csv
        ../data/Model_szewczyk/model_name/model_name_data.csv
            
You only need to change the `Model_**` part for the path; the rest should stay the same. Line to change, *language_models.py, line39, line112, line169, line241*.