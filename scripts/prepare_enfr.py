from homework2.utils.text_prepro import prepare_eng_fra_splits

if __name__ == "__main__":
    res = prepare_eng_fra_splits(
        file_path="data/eng-fra.txt",
        outdir="data/enfr_splits",
        train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1,
        min_len=1, max_len=60,
        lower=True, dedupe=True,
        save_format="paired",
        seed=54321,
    )
    print("saved to:", res["paths"])