"""
Soumet un run du pipeline d'entraînement à Vertex AI Pipelines.

    python pipelines/run_pipeline.py [--min-auc-pr 0.08] [--no-wait]

Compile d'abord le pipeline si le JSON est absent.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from google.cloud import aiplatform

from aml_detection import config

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "training_pipeline.json"
PIPELINE_ROOT = f"gs://{config.GCS_BUCKET or 'mlops-vertex-demo-aml'}/pipeline-root"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-auc-pr", type=float, default=None,
                    help="surcharge le seuil d'enregistrement du pipeline")
    ap.add_argument("--no-wait", action="store_true", help="ne pas bloquer sur la fin du run")
    args = ap.parse_args()

    if not TEMPLATE.exists():
        subprocess.run([sys.executable, str(HERE / "training_pipeline.py")], check=True)

    params = {}
    if args.min_auc_pr is not None:
        params["min_auc_pr"] = args.min_auc_pr

    aiplatform.init(
        project=config.PROJECT_ID,
        location=config.REGION,
        # évite l'avertissement "bucket squatting" du pré-check auto du SDK
        staging_bucket=PIPELINE_ROOT,
    )
    job = aiplatform.PipelineJob(
        display_name="aml-training",
        template_path=str(TEMPLATE),
        pipeline_root=PIPELINE_ROOT,
        parameter_values=params or None,
        enable_caching=False,
    )
    job.submit()
    run_id = job.resource_name.split("/")[-1]
    print(f"Run soumis : {job.resource_name}")
    print(
        "Console : https://console.cloud.google.com/vertex-ai/locations/"
        f"{config.REGION}/pipelines/runs/{run_id}?project={config.PROJECT_ID}"
    )

    if not args.no_wait:
        job.wait()
        print(f"État final : {job.state}")


if __name__ == "__main__":
    main()
