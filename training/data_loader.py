import pandas as pd

class DataLoader:
    def __init__(self, data_path, sample_fraction=1.0, random_seed=42):
        self.data_path = data_path
        self.sample_fraction = sample_fraction
        self.random_seed = random_seed
        self.data = None

    def load_data(self):
        print(f"Učitavam podatke iz: {self.data_path}")

        self.data = pd.read_csv(self.data_path)

        print(f"Ukupno transakcija: {len(self.data):,}")
        print(
            f"Prevare: {len(self.data[self.data['Class'] == 1]):,} ({len(self.data[self.data['Class'] == 1]) / len(self.data) * 100:.2f}%)")

        if self.sample_fraction < 1.0:
            print(f"\nKoristi se {self.sample_fraction * 100}% podataka (sample)")

            # da bi se osiguralo da ostaje isti odnos uzima se isti procenat od obe klase
            self.data = (
                self.data
                .groupby('Class', group_keys=False) # false da bi svi u klasi imali isti index, lakse za obradu
                .sample(frac=self.sample_fraction, random_state=self.random_seed)
                .reset_index(drop=True) # resetuje indexe zbog brze obrade
            )

            print(f"Nakon samplinga: {len(self.data):,} transakcija")
            print(f"Prevare: {len(self.data[self.data['Class'] == 1]):,}")

            # shuffle podataka
            self.data = self.data.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        return self.data

    def split_data(self, initial_split=0.7):
        if self.data is None:
            raise ValueError("Podaci nisu učitani! Pozovi prvo load_data()")

        split_idx = int(len(self.data) * initial_split)

        # podela podataka offline i online
        initial_data = self.data.iloc[:split_idx].copy()
        streaming_data = self.data.iloc[split_idx:].copy()

        print(f"\n=== PODELA PODATAKA ===")
        print(f"Inicijalni set: {len(initial_data):,} transakcija")
        print(f"  - Prevare: {len(initial_data[initial_data['Class'] == 1]):,}")
        print(f"Streaming set: {len(streaming_data):,} transakcija")
        print(f"  - Prevare: {len(streaming_data[streaming_data['Class'] == 1]):,}")

        return initial_data, streaming_data

    def get_feature_names(self):
        if self.data is None:
            raise ValueError("Podaci nisu učitani!")

        return [col for col in self.data.columns if col not in ['Class', 'Time']]