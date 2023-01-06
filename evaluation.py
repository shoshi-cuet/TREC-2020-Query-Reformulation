import pandas as pd
from trectools import TrecQrel,TrecRun

QRELS = TrecQrel("data/qrels")

def score_table_by_turn(run:TrecRun)-> pd.DataFrame:
    """
    Function to making a scoring dataframe based on a run

    run: An object of TrecRun

    returns pandas DataFrame
    """
    topic_number = []
    df_list = []

    for i in run.index:
        r = i.split('_')
        topic_number.append(int(r[0]))

    for i,topic in enumerate(set(topic_number)):
        z = run[run.index.str.startswith(str(topic))].reset_index()
        z[run.columns[0]] = z[run.columns[0]].fillna(0)
        z = z.pivot_table(values=run.columns[0], columns='query')
        z.rename(columns=lambda s: int(s.split('_')[1]),index={z.index[0]:topic},inplace=True)
        df_list.append(z)
        
    df = pd.concat(df_list)
    df = df.reindex(sorted(df.columns), axis=1)
    df.loc['Mean per turn',:]= df.mean(axis=0,skipna=True)

    return df
