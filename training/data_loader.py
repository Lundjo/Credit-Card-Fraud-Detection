import pandas as pd


class DataLoader:
    def __init__(self, data_path, sample_fraction, random_seed):
        self.data_path = data_path
        self.sample_fraction = sample_fraction
        self.random_seed = random_seed
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

        if self.sample_fraction < 1.0:
            # da bi se osiguralo da ostaje isti odnos uzima se isti procenat od obe klase
            self.data = (
                self.data
                .groupby('Class', group_keys=False) # false da bi svi u klasi imali isti index, lakse za obradu
                .sample(frac=self.sample_fraction, random_state=self.random_seed)
                .reset_index(drop=True) # resetuje indexe zbog brze obrade
            )

            # shuffle podataka
            self.data = self.data.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        return self.data

    def split_data(self, initial_split):
        if self.data is None:
            raise ValueError("Podaci nisu učitani! Pozovi prvo load_data()")

        split_idx = int(len(self.data) * initial_split)

        # podela podataka offline i online
        initial_data = self.data.iloc[:split_idx].copy()
        streaming_data = self.data.iloc[split_idx:].copy()

        return initial_data, streaming_data
