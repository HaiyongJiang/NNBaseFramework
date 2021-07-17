import os
import time
import argparse
from tqdm import tqdm
import pandas as pd

eval_dict = {
    'idx': idx,
    'class id': category_id,
    'class name': category_name,
    'modelname':modelname,
}
eval_dicts.append(eval_dict)
eval_data = trainer.eval_step(data)[-1]
eval_dict.update(eval_data)
time_eval += time.time() - t_start
print("Time summary: %s"%(time_eval/len(test_loader)))


# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()
eval_df_class.to_csv(out_file_class)

# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
print(eval_df_class)

