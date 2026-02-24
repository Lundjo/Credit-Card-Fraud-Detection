from river import forest, metrics
from config import Config

class OnlineModel:
    def __init__(self, config, threshold):
        self.config = config
        self.model = None
        self.threshold = threshold

        # online metrike koje se azuriraju sa svakom transakcijom
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.f1 = metrics.F1()
        self.auc = metrics.ROCAUC()

    def initialize(self):
        print("\n" + "=" * 60)
        print("INICIJALIZACIJA ONLINE MODELA (ARF Classifier)")
        print("=" * 60)
        print(f"Broj stabala: {self.config.ARF_N_MODELS}")
        print(f"Max features: {self.config.ARF_MAX_FEATURES}")
        print(f"Lambda: {self.config.ARF_LAMBDA}")
        print(f"Threshold: {self.threshold}")

        self.model = forest.ARFClassifier(
            n_models=self.config.ARF_N_MODELS,
            max_features=self.config.ARF_MAX_FEATURES,
            lambda_value=self.config.ARF_LAMBDA,
            seed=self.config.RANDOM_SEED
        )

        print("✓ Online model kreiran!")
        return self.model

    # metoda za online ucenje
    def learn_one(self, x_dict, y_true): # x_dict zato sto ih u tom formatu ocekuje online forest
        if self.model is None:
            raise ValueError("Model nije inicijalizovan! Pozovi initialize()")

        y_pred = self.predict_one(x_dict)
        y_pred_proba = self.predict_proba_one(x_dict).get(True, 0) # vraca true probability ili 0 ako je prazno

        self.accuracy.update(y_true, y_pred)
        self.precision.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
        self.f1.update(y_true, y_pred)
        self.auc.update(y_true, y_pred_proba)

        self.model.learn_one(x_dict, y_true)

    # predvidjanje i zbog poboljsanja sume
    def predict_one(self, x_dict):
        if self.model is None:
            raise ValueError("Model nije inicijalizovan!")

        proba = self.model.predict_proba_one(x_dict)

        return proba.get(True, 0) > self.threshold

    # verovatnoca tacnosti
    def predict_proba_one(self, x_dict):
        if self.model is None:
            raise ValueError("Model nije inicijalizovan!")

        return self.model.predict_proba_one(x_dict)

    def get_metrics(self):
        return {
            'accuracy': self.accuracy.get(),
            'precision': self.precision.get(),
            'recall': self.recall.get(),
            'f1': self.f1.get(),
            'auc': self.auc.get()
        }

    def reset_metrics(self):
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.f1 = metrics.F1()
        self.auc = metrics.ROCAUC()

    def set_threshold(self, new_threshold):
        self.threshold = new_threshold
        print(f"✓ Threshold promenjen na: {self.threshold}")