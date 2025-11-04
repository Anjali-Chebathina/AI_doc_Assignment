# data_loader.py
import os
import pandas as pd
from datasets import load_dataset

def load_resume_csv():
   resume_path = os.path.join("data", "resume_data.csv")
   if not os.path.exists(resume_path):
        raise FileNotFoundError(f" resume_data.csv not found at {resume_path}")
   df = pd.read_csv(resume_path)
   df.fillna('', inplace=True)
   print(f"Loaded resume dataset with {len(df)} records.")
   return df


def load_cord_dataset():
   print("Loading CORD-v2 dataset from Hugging Face...")
   ds = load_dataset("naver-clova-ix/cord-v2")
   print(f" CORD dataset loaded successfully with splits: {list(ds.keys())}")
   return ds


if __name__ == "__main__":
    # Test data loading
    resumes = load_resume_csv()
    print(resumes.head())

    cord = load_cord_dataset()
    print(cord)
