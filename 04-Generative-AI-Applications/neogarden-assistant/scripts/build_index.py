"""Construit l'index FAISS et le sauvegarde sur disque (dossier `vectorstore/`).

À lancer une fois avant une démo : l'appli démarre ensuite sans recalculer les
embeddings. Si l'index existe déjà et que les documents n'ont pas changé, il est
simplement rechargé.

Usage :
    python scripts/build_index.py            # construit si nécessaire
    python scripts/build_index.py --force    # reconstruit dans tous les cas
"""

import argparse
import logging

from rag_garden import config
from rag_garden.embeddings import load_or_build_vectorstore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Lit les options de la ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="reconstruit l'index même s'il est à jour",
    )
    return parser.parse_args()


def main() -> None:
    """Construit (ou recharge) l'index et indique où il se trouve."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    vectorstore = load_or_build_vectorstore(force_rebuild=args.force)
    logger.info(
        "Index prêt : %d vecteurs dans %s",
        vectorstore.index.ntotal,
        config.VECTORSTORE_DIR,
    )


if __name__ == "__main__":
    main()
