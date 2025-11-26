# Lora_experiment.ipynb

### Template
It is run on `kaggle`.
Copy the `lora_experiment.ipynb` onto kaggle, path of ipynb: `python_project\experiments\lora_experiment.ipynb`.
Then run those cell one by one. 

### Factory Reset
Remember to click the `Factory Reset` bottom **every time finish running a cell** (Or it may OOM) As the system would not release the memory used on GPU/CPU automatically. And in this experiment we have to record the memory usage, reset the memory for each experiment can help compare the result.

Every time reset the environment, remember to run the code `!pip install evaluate` again.

### Mount dataset
One more thing important is to **mount those dataset onto kaggle**, make sure the path corresponding to the code written. Mount the whole folder(classification-data/generation-data). 

path: 
`python_project\data\classification` we would use the enron_spam_data.csv (non augmented version) in it.
`python_project\data\email_reply_dataset` Use synthetic_reply_dataset.csv (non augmented version).

Already provided the code to check path name of the kaggle(the code in second cell).
The prefix of input path should be '/kaggle/input'

### Output saving
If want to save the output file, please also set the path correctly. The prefix of input path should be '/kaggle/working'. The output file will be deleted if the connection is cut, remember to save them properly.

# Zero-shot.ipynb 

Run the code one by one on `kaggle`. Copy the zero-shot-few-shot.ipynb onto kaggle. path: `python_project\experiments\zero-shot-few-shot.ipynb`

### Factory Reset
As we don't need to record the memory usage and the memory usage is not that extensive, if no OOM happen, no need to implement `Factory Reset`.

### Dataset
Also should **mount those dataset onto kaggle**. Guidance the same as above.

### Output saving
Same as above.