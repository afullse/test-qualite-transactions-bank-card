# Test assurance qualité d'un modèle de détection de fraude

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
## Exécution
```bash
pytest --cov=app --cov-report=term-missing       # tests (données synthétiques)
ruff check .                                     # style
python -m app.run_experiment --data data/creditcard.csv   # expérience réelle (Kaggle)
```
## Baseline (non-régression)
`reports/baseline.json` est **figée et versionnée**. Mise à jour uniquement manuelle, via PR revue :
`UPDATE_BASELINE=1 pytest tests/test_regression.py`. La CI ne la recalcule jamais.

## Preuve d'un échec volontaire
1. Créer une branche `demo/echec`, modifier `app/training.py` : `class_weight=None` et `C=1e-4`.
2. Ouvrir une PR → `test_no_regression_vs_baseline` échoue (rappel en baisse) → workflow rouge.
3. Capturer l'écran, fermer la PR.

## Protection de branche
Settings → Branches → `main` : exiger PR + statut « AI-TestGuard CI / tests » réussi.

Documents : `PLAN_DE_TEST.md`, `RAPPORT_QUALIFICATION.md`.
