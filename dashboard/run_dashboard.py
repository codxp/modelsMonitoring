import streamlit as st
import mlflow
import plotly.express as px
import plotly.graph_objects as go

class MLflowDashboard:
    def __init__(self):
        self.experiment_name = "breast_cancer_comparison"
        
    def load_mlflow_runs(self):
        """Charge les données depuis MLflow"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs([experiment.experiment_id])
        return runs

    def create_metrics_df(self, runs):
        """Crée un DataFrame formaté pour les métriques"""

        metrics_df = runs[['metrics.accuracy', 'metrics.precision', 
                          'metrics.recall', 'metrics.roc_auc', 
                          'metrics.training_time', 'metrics.cv_accuracy_mean',
                          'metrics.cv_accuracy_std', 'params.model_type']]
        
        metrics_df.columns = ['Accuracy', 'Precision', 'Recall', 
                            'ROC AUC', 'Training Time', 'CV Accuracy Mean',
                            'CV Accuracy Std', 'Model']
        return metrics_df

    def run_dashboard(self):
        """Exécute le dashboard Streamlit"""

        st.set_page_config(page_title="Breast Cancer Models Dashboard", layout="wide")
        st.title("Dashboard de Comparaison des Modèles - Cancer du Sein")

        # Chargement des données
        runs = self.load_mlflow_runs()
        metrics_df = self.create_metrics_df(runs)

        # Sidebar
        st.sidebar.title("Filtres et Options")
        
        # Sélection des modèles
        selected_models = st.sidebar.multiselect(
            "Sélectionner les modèles",
            options=metrics_df['Model'].unique(),
            default=metrics_df['Model'].unique()
        )

        # Sélection des métriques
        selected_metrics = st.sidebar.multiselect(
            "Sélectionner les métriques à afficher",
            options=['Accuracy', 'Precision', 'Recall', 'ROC AUC'],
            default=['Accuracy', 'Precision', 'Recall', 'ROC AUC']
        )

        # Filtrer les données
        filtered_df = metrics_df[metrics_df['Model'].isin(selected_models)]

        # Layout principal
        st.header("Vue d'ensemble des Performances")
        
        # Graphique radar des performances
        fig_radar = go.Figure()
        for model in selected_models:
            model_data = filtered_df[filtered_df['Model'] == model]
            if not model_data.empty:
                fig_radar.add_trace(go.Scatterpolar(
                    r=[model_data[metric].iloc[0] for metric in selected_metrics],
                    theta=selected_metrics,
                    name=model
                ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0.9, 1])),
            title="Comparaison des métriques par modèle"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Temps d'entraînement et performance
        col1, col2 = st.columns(2)
        
        with col1:
            fig_time = px.bar(filtered_df, 
                            x='Model', 
                            y='Training Time',
                            title='Temps d\'Entraînement par Modèle',
                            color='Model')
            fig_time.update_layout(
                xaxis_title="Modèle",
                yaxis_title="Temps (secondes)"
            )
            st.plotly_chart(fig_time)

        with col2:
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(
                name='CV Accuracy',
                x=filtered_df['Model'],
                y=filtered_df['CV Accuracy Mean'],
                error_y=dict(
                    type='data',
                    array=filtered_df['CV Accuracy Std'],
                    visible=True
                )
            ))
            fig_cv.update_layout(
                title='Accuracy CV avec Écart-type',
                xaxis_title="Modèle",
                yaxis_title="Accuracy"
            )
            st.plotly_chart(fig_cv)

        # Métriques clés
        st.header("Métriques Clés")
        col3, col4, col5, col6 = st.columns(4)
        
        best_accuracy = filtered_df.loc[filtered_df['Accuracy'].idxmax()]
        best_precision = filtered_df.loc[filtered_df['Precision'].idxmax()]
        best_recall = filtered_df.loc[filtered_df['Recall'].idxmax()]
        best_roc = filtered_df.loc[filtered_df['ROC AUC'].idxmax()]

        with col3:
            st.metric(
                "Meilleure Accuracy",
                f"{best_accuracy['Accuracy']:.3f}",
                f"Modèle: {best_accuracy['Model']}"
            )
        
        with col4:
            st.metric(
                "Meilleure Precision",
                f"{best_precision['Precision']:.3f}",
                f"Modèle: {best_precision['Model']}"
            )

        with col5:
            st.metric(
                "Meilleur Recall",
                f"{best_recall['Recall']:.3f}",
                f"Modèle: {best_recall['Model']}"
            )

        with col6:
            st.metric(
                "Meilleur ROC AUC",
                f"{best_roc['ROC AUC']:.3f}",
                f"Modèle: {best_roc['Model']}"
            )

        # Tableau de comparaison détaillé
        st.header("Comparaison Détaillée des Modèles")
        comparison_df = filtered_df[['Model', 'Accuracy', 'Precision', 'Recall', 'ROC AUC', 'CV Accuracy Mean']]
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, color='lightgreen')
                        .format({col: "{:.3f}" for col in comparison_df.columns if col != 'Model'})
        )

if __name__ == "__main__":
    dashboard = MLflowDashboard()
    dashboard.run_dashboard()