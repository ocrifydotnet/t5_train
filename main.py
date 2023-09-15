from datasets import Dataset,load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
def main():
    sentences = load_dataset("hantech/correct_dataset")
    df_new = pd.DataFrame( sentences['train'] )

    train_df, test_df = train_test_split(df_new, test_size=0.2)
    
    from simplet5 import SimpleT5

    model = SimpleT5()
    model.from_pretrained(model_type="byt5", model_name="google/byt5-small")
    model.train(train_df=train_df[:500000],
                eval_df=test_df[:12000],
                source_max_token_len=128,
                target_max_token_len=50,
                batch_size=8, max_epochs=3, use_gpu=False)
if __name__ == '__main__':
    main()