import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

df = pd.read_csv('train.csv')
df['kfold'] = -1
#shuffling the data
df = df.sample(frac=1).reset_index(drop = True)

X = df.image_id.values
y= df[["grapheme_root", 'vowel_diacritic', 'consonant_diacritic']].values

mskf = MultilabelStratifiedKFold(n_splits=5)
for fold, (train_ind, val_ind) in enumerate(mskf.split(X,y)):
    print('TRAIN:', train_ind, 'VAL: ', val_ind)
    df.loc[val_ind,'kfold'] = fold

    
print(df.kfold.value_counts)
df.to_csv('train_folds.csv', index = False)
