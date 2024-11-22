import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from typing import Dict
import time

class BreastCancerModelTrainer:
    def __init__(self, experiment_name: str = "breast_cancer_comparison"):
        try:
            mlflow.end_run()
        except Exception:
            pass
            
        mlflow.set_experiment(experiment_name)
        self.models = {
            'logistic': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'max_iter': [200],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'knn': {
                'model': KNeighborsClassifier,
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'svm': {
                'model': SVC,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'probability': [True]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'ada_boost': {
                'model': AdaBoostClassifier,
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'algorithm': ['SAMME']  # Changé de SAMME.R à SAMME
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier,
                'params': {
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5],
                    'criterion': ['gini', 'entropy']
                }
            },
            'naive_bayes': {
                'model': GaussianNB,
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                }
            }
        }
        self.load_data()

    def load_data(self):
        data = load_breast_cancer()
        self.X = pd.DataFrame(data.data, columns=data.feature_names)
        self.y = data.target
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def create_model_signature(self, model):
        """Crée une signature MLflow pour le modèle"""
        from mlflow.models.signature import infer_signature
        
        # Utiliser un petit échantillon des données pour inférer la signature
        signature = infer_signature(
            self.X_train_scaled[:5],
            model.predict(self.X_train_scaled[:5])
        )
        return signature

    def train_and_evaluate(self, model_name: str) -> Dict[str, float]:
        model_info = self.models[model_name]
        results = {}
        
        try:
            with mlflow.start_run(run_name=f"{model_name}_training"):
                print(f"\nEntraînement du modèle: {model_name}")
                
                mlflow.log_params({
                    "model_type": model_name,
                    "search_params": str(model_info['params'])
                })
                
                grid_search = GridSearchCV(
                    model_info['model'](),
                    model_info['params'],
                    cv=5,
                    scoring=['accuracy', 'precision', 'recall', 'roc_auc'],
                    refit='accuracy',
                    n_jobs=-1
                )
                
                start_time = time.time()
                grid_search.fit(self.X_train_scaled, self.y_train)
                training_time = time.time() - start_time
                
                y_pred = grid_search.predict(self.X_test_scaled)
                y_pred_proba = grid_search.predict_proba(self.X_test_scaled)[:, 1]
                
                results = {
                    'model': model_name,
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                    'training_time': training_time,
                    'cv_accuracy_mean': grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_],
                    'cv_accuracy_std': grid_search.cv_results_['std_test_accuracy'][grid_search.best_index_],
                    'best_params': str(grid_search.best_params_)
                }
                
                # Log métriques et paramètres
                mlflow.log_metrics({k: v for k, v in results.items() if isinstance(v, (int, float))})
                mlflow.log_params(grid_search.best_params_)
                
                # Créer et logger le modèle avec signature
                signature = self.create_model_signature(grid_search.best_estimator_)
                input_example = self.X_train_scaled[:5]
                
                mlflow.sklearn.log_model(
                    grid_search.best_estimator_, 
                    f"best_{model_name}",
                    signature=signature,
                    input_example=input_example
                )
                
                # Feature importance
                if hasattr(grid_search.best_estimator_, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': self.X.columns,
                        'importance': grid_search.best_estimator_.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    feature_importance.to_csv('data/feature_importance.csv', index=False)
                    mlflow.log_artifact('data/feature_importance.csv')
                    
                    print(f"Performance {model_name}: accuracy={results['accuracy']:.3f}, "
                          f"ROC AUC={results['roc_auc']:.3f}, "
                          f"Training Time={results['training_time']:.2f}s")
                
        except Exception as e:
            print(f"Erreur lors de l'entraînement de {model_name}: {str(e)}")
            results = None
        
        finally:
            mlflow.end_run()
            
        return results

    def train_all_models(self) -> pd.DataFrame:
        results = []
        
        for model_name in self.models.keys():
            model_results = self.train_and_evaluate(model_name)
            if model_results is not None:
                results.append(model_results)
        
        return pd.DataFrame(results)

if __name__ == "__main__":
    try:
        mlflow.end_run()
    except Exception:
        pass
        
    trainer = BreastCancerModelTrainer()
    results_df = trainer.train_all_models()
    
    if not results_df.empty:
        print("\nRésultats finaux:")
        print(results_df.sort_values('accuracy', ascending=False))
        
        # Sauvegarder les résultats
        results_df.to_csv('data/model_comparison_results.csv', index=False)
        print("\nRésultats sauvegardés dans 'data/model_comparison_results.csv'")
    else:
        print("Aucun résultat n'a été obtenu.")