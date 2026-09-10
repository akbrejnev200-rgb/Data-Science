"""
Pipeline d'entraînement AML sur Vertex AI Pipelines (KFP v2).

    train ──▶ [ auc_pr ≥ MIN_AUC_PR ] ──▶ register   (nouvelle version au Registry)
          └─▶ [ sinon ]               ──▶ report_rejection

Compilation :  python pipelines/training_pipeline.py   ->  training_pipeline.json
Soumission  :  python pipelines/run_pipeline.py
"""

from typing import NamedTuple

from kfp import compiler, dsl
from kfp.dsl import Input, Metrics, Model, Output

from aml_detection import config

TRAINER_IMAGE = (
    f"{config.REGION}-docker.pkg.dev/{config.PROJECT_ID}/aml-detection/trainer:v2"
)
KFP_PIN = "kfp==2.17.0"
MIN_AUC_PR = 0.08
PIPELINE_NAME = "aml-training"


@dsl.component(base_image=TRAINER_IMAGE, packages_to_install=[KFP_PIN])
def train(
    project_id: str,
    region: str,
    model: Output[Model],
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [("auc_pr", float)]):
    """Exécute le pipeline aml_detection : BigQuery -> entraînement -> évaluation.

    Les artefacts (model.bst, metrics.json, .joblib) sont publiés dans `model.uri`
    (dossier GCS géré par Vertex Pipelines, donc tracé automatiquement).
    """
    import json
    import os

    os.environ["AML_PROJECT_ID"] = project_id
    os.environ["AML_REGION"] = region
    os.environ["AML_ARTIFACTS_URI"] = model.uri

    from aml_detection.main import main as run_training

    run_training()

    m = json.loads(open("/app/models/metrics.json", encoding="utf-8").read())
    t = m["test"]
    metrics.log_metric("auc_pr_test", t["auc_pr"])
    metrics.log_metric("auc_roc_test", t["auc_roc"])
    metrics.log_metric("recall_suspect", t["recall_suspect"])
    metrics.log_metric("precision_suspect", t["precision_suspect"])
    metrics.log_metric("decision_threshold", m["decision_threshold"])
    model.metadata["model_name"] = m["model_name"]

    out = NamedTuple("Outputs", [("auc_pr", float)])
    return out(float(t["auc_pr"]))


@dsl.component(base_image=TRAINER_IMAGE, packages_to_install=[KFP_PIN, "google-cloud-aiplatform"])
def register(project_id: str, region: str, model: Input[Model]) -> str:
    """Crée une nouvelle version du modèle dans le Vertex AI Model Registry."""
    from aml_detection.registry import register_version

    resource_name = register_version(project_id, region, model.uri)
    return resource_name or ""


@dsl.component(base_image=TRAINER_IMAGE, packages_to_install=[KFP_PIN])
def report_rejection(auc_pr: float, min_auc_pr: float):
    """Chemin « modèle rejeté » : on trace pourquoi, rien n'est enregistré."""
    print(f"Modèle REJETÉ : AUC-PR test {auc_pr:.3f} < seuil {min_auc_pr:.3f}.")
    print("Aucune nouvelle version enregistrée — la version courante reste en place.")


@dsl.pipeline(name=PIPELINE_NAME, description="Entraînement AML + enregistrement conditionnel")
def training_pipeline(
    project_id: str = config.PROJECT_ID,
    region: str = config.REGION,
    min_auc_pr: float = MIN_AUC_PR,
):
    train_task = train(project_id=project_id, region=region)
    train_task.set_display_name("Entraînement + évaluation")
    # Entraînement long et non déterministe côté ressources : pas de cache.
    train_task.set_caching_options(False)

    with dsl.If(train_task.outputs["auc_pr"] >= min_auc_pr, name="modele-accepte"):
        register(
            project_id=project_id, region=region, model=train_task.outputs["model"]
        ).set_display_name("Enregistrement au Model Registry")

    with dsl.Else(name="modele-rejete"):
        report_rejection(
            auc_pr=train_task.outputs["auc_pr"], min_auc_pr=min_auc_pr
        ).set_display_name("Rapport de rejet")


if __name__ == "__main__":
    target = "pipelines/training_pipeline.json"
    compiler.Compiler().compile(pipeline_func=training_pipeline, package_path=target)
    print(f"Compilé -> {target}")
