from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns

class KNNTrainer:
    def __init__(self):
        self.model = None

    def train_with_random_search(self, X, y, params, cv=5, n_iter=100):
        random = RandomizedSearchCV(
            estimator=KNeighborsClassifier(),
            param_distributions=params,
            n_iter=n_iter,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
        random.fit(X, y)
        self.model = KNeighborsClassifier(**random.best_params_)
        self.model.fit(X, y)
        return random.best_params_

    def train_with_optuna(self, X, y, n_trials=100):
        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int("n_neighbors", 1, 100),
                'weights': trial.suggest_categorical("weights", ["uniform", "distance"]),
                'p': trial.suggest_float("p", 1.0, 100.0)
            }
            model = KNeighborsClassifier(**params)
            return cross_val_score(model, X, y, cv=3, scoring='f1', n_jobs=-1).mean()

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        self.model = KNeighborsClassifier(**best_params)
        self.model.fit(X, y)
        return best_params

    def predict(self, X_test):
        return self.model.predict(X_test)
#EVALUACION DEL MODELO
    def evaluar_modelo(self, X_test, y_test, nombre="Modelo KNN"):
        y_pred = self.model.predict(X_test)

        print(f"\nEvaluación para {nombre}:")
        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.grid(False)
        disp.plot(ax=ax)
        ax.set_title(nombre)
        plt.show()
