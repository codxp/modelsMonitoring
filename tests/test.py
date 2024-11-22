import sys
sys.path.append("/Users/flo/Documents/DevIA-1/monitoring/")
import unittest
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from models.models_training import BreastCancerModelTrainer
import mlflow
import os
import warnings
from sklearn.preprocessing import StandardScaler

class TestBreastCancerModelTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour tous les tests"""
        cls.trainer = BreastCancerModelTrainer(experiment_name="test_experiment")
        cls.data = load_breast_cancer()
        
    def setUp(self):
        """Configuration pour chaque test"""
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        
    def test_data_loading(self):
        """Test du chargement des données"""
        # Vérifier que les données sont chargées correctement
        self.assertIsNotNone(self.trainer.X)
        self.assertIsNotNone(self.trainer.y)
        
        # Vérifier les dimensions
        self.assertEqual(self.trainer.X.shape[1], 30)  # 30 features
        self.assertEqual(len(self.trainer.y), len(self.trainer.X))
        
        # Vérifier que les noms des colonnes correspondent
        self.assertListEqual(list(self.trainer.X.columns), list(self.data.feature_names))

    def test_train_test_split(self):
        """Test de la séparation train/test"""
        # Vérifier les proportions
        train_size = len(self.trainer.X_train)
        test_size = len(self.trainer.X_test)
        total_size = train_size + test_size
        
        self.assertAlmostEqual(test_size / total_size, 0.2, places=1)
        self.assertAlmostEqual(train_size / total_size, 0.8, places=1)
        
        # Vérifier qu'il n'y a pas de fuite de données
        train_indices = set(self.trainer.X_train.index)
        test_indices = set(self.trainer.X_test.index)
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)

    def test_data_scaling(self):
        """Test de la standardisation des données"""
        # Vérifier que les données sont standardisées
        X_train_mean = np.mean(self.trainer.X_train_scaled, axis=0)
        X_train_std = np.std(self.trainer.X_train_scaled, axis=0)
        
        # Les moyennes devraient être proches de 0
        np.testing.assert_array_almost_equal(X_train_mean, np.zeros_like(X_train_mean), decimal=1)
        # Les écarts-types devraient être proches de 1
        np.testing.assert_array_almost_equal(X_train_std, np.ones_like(X_train_std), decimal=1)

    def test_model_configs(self):
        """Test des configurations des modèles"""
        # Vérifier que tous les modèles ont les paramètres requis
        for model_name, model_info in self.trainer.models.items():
            self.assertIn('model', model_info)
            self.assertIn('params', model_info)
            self.assertTrue(callable(model_info['model']))
            self.assertIsInstance(model_info['params'], dict)

    def test_single_model_training(self):
        """Test de l'entraînement d'un seul modèle"""
        # Tester avec un modèle simple (Naive Bayes)
        results = self.trainer.train_and_evaluate('naive_bayes')
        
        # Vérifier les résultats
        self.assertIsNotNone(results)
        self.assertIn('accuracy', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('roc_auc', results)
        
        # Vérifier que les métriques sont dans des plages valides
        self.assertTrue(0 <= results['accuracy'] <= 1)
        self.assertTrue(0 <= results['precision'] <= 1)
        self.assertTrue(0 <= results['recall'] <= 1)
        self.assertTrue(0 <= results['roc_auc'] <= 1)

    def test_mlflow_logging(self):
        """Test du logging MLflow"""
        model_name = 'naive_bayes'
        self.trainer.train_and_evaluate(model_name)
        
        # Vérifier que l'expériment existe
        experiment = mlflow.get_experiment_by_name("test_experiment")
        self.assertIsNotNone(experiment)
        
        # Vérifier les runs
        runs = mlflow.search_runs([experiment.experiment_id])
        self.assertGreater(len(runs), 0)

    def test_feature_importance(self):
        """Test de l'importance des features"""
        # Tester avec Random Forest qui a feature_importances_
        self.trainer.train_and_evaluate('random_forest')
        
        # Vérifier que le fichier d'importance des features est créé
        self.assertTrue(os.path.exists('data/feature_importance.csv'))
        
        # Charger et vérifier le fichier
        feature_importance = pd.read_csv('data/feature_importance.csv')
        self.assertEqual(len(feature_importance), 30)  # 30 features
        self.assertTrue('importance' in feature_importance.columns)
        self.assertTrue('feature' in feature_importance.columns)

# Tests Pytest
@pytest.fixture
def trainer():
    """Fixture pour créer une instance du trainer"""
    return BreastCancerModelTrainer(experiment_name="pytest_experiment")

def test_data_shape(trainer):
    """Test des dimensions des données"""
    assert trainer.X.shape[1] == 30
    assert trainer.X_train.shape[0] + trainer.X_test.shape[0] == len(trainer.X)

def test_data_consistency(trainer):
    """Test de la cohérence des données"""
    # Vérifier qu'il n'y a pas de valeurs manquantes
    assert not trainer.X.isna().any().any()
    assert not pd.Series(trainer.y).isna().any()
    
    # Vérifier que les labels sont binaires
    assert set(trainer.y) == {0, 1}

def test_model_parameters(trainer):
    """Test des paramètres des modèles"""
    for model_name, model_info in trainer.models.items():
        # Vérifier que les paramètres sont valides pour le modèle
        model = model_info['model']()
        valid_params = model.get_params().keys()
        
        for param in model_info['params'].keys():
            assert param in valid_params

@pytest.mark.parametrize("model_name", [
    'logistic',
    'knn',
    'random_forest',
    'naive_bayes'
])
def test_individual_models(trainer, model_name):
    """Test paramétrique pour chaque modèle"""
    results = trainer.train_and_evaluate(model_name)
    assert results is not None
    assert results['accuracy'] > 0.5  # Devrait être meilleur qu'un classifieur aléatoire

def test_data_leakage(trainer):
    """Test pour vérifier qu'il n'y a pas de fuite de données"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(trainer.X_train)
    X_test_scaled = scaler.transform(trainer.X_test)
    
    # Vérifier que les statistiques de scaling sont différentes
    train_mean = X_train_scaled.mean()
    test_mean = X_test_scaled.mean()
    assert not np.allclose(train_mean, test_mean)

if __name__ == '__main__':
    # Exécuter les tests unittest
    unittest.main(verbosity=2)