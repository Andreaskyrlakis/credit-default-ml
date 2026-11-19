"""Builds notebooks/walkthrough.ipynb: the project in short cells, explained in simple French.

Run:  python notebooks/build_walkthrough.py
then: jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb
(needs the outputs of run_all.py: data/processed/ and results/)
"""

from pathlib import Path

import nbformat as nbf

cells = []
md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))  # noqa: E731
code = lambda text: cells.append(nbf.v4.new_code_cell(text))  # noqa: E731

md("""# Prédire le défaut de crédit, puis calculer les pertes attendues IFRS 9

Ce carnet raconte le projet en cellules courtes. Le calcul complet est dans `scripts/`,
le rapport dans `report/main.pdf`. Tous les chiffres viennent des fichiers `results/*.json`
produits par `python run_all.py`.

**Les données** : 620 779 prêts réels de Lending Club (prêts à la consommation américains,
36 mois, émis de 2007 à 2015, tous terminés : remboursés ou passés en perte).""")

code("""import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from IPython.display import Image, display
from src import config
from src.data import load_loans, modelling_sample, time_split

R = lambda name: json.load(open(ROOT / "results" / f"{name}.json"))
fig = lambda name: display(Image(filename=str(ROOT / "figures" / f"{name}.png"), width=750))""")

md("""## 1. Les données et la découpe dans le temps

Un prêt est « en défaut » s'il a été passé en perte (*charged off*).
On découpe **dans le temps**, comme en vrai : on apprend sur le passé, on teste sur le futur.
- entraînement : prêts émis de 2007 à 2014 ;
- validation : janvier à juin 2015 (pour régler et calibrer) ;
- test : juillet à décembre 2015 (touché une seule fois, à la fin).""")

code("""loans = modelling_sample(load_loans())
train, valid, test = time_split(loans)
for name, d in [("train", train), ("valid", valid), ("test", test)]:
    print(f"{name:6s} {len(d):>8,} prêts   taux de défaut {d['default'].mean():.1%}")""")

code("""fig("fig01_volume_default_rate")""")

md("""Le taux de défaut monte avec le temps (13 % en entraînement, 15 % en 2015) :
Lending Club a prêté à des clients plus risqués. C'est pour ça qu'il faudra **recalibrer**
les probabilités sur la période la plus récente (partie 5).""")

md("""## 2. La carte de score classique : WoE et IV

On coupe chaque variable en tranches. Pour chaque tranche :
- **WoE** = ln(part des bons clients / part des mauvais). Positif = tranche plus sûre ;
- **IV** (information value) = force de la variable. On garde les variables avec IV ≥ 0,02.

Puis une régression logistique sur les WoE donne des **points** : +20 points = risque divisé par 2.""")

code("""sc = R("scorecard")
pd.Series(sc["iv"]).round(3).head(10)""")

code("""fig("fig04_woe_bins")""")

code("""pts = pd.read_csv(ROOT / "results" / "scorecard_points.csv")
pts[pts.feature == "fico"].round(1)""")

md("""Le score FICO est de loin la variable la plus forte. Résultat de la carte de score sur le test :""")

code("""print("carte de score   AUC", round(sc["test"]["auc"], 3), " Gini", round(sc["test"]["gini"], 3))
print("sous-note Lending Club AUC", round(sc["lc_subgrade_test"]["auc"], 3), "(référence : LC a plus d'informations)")""")

md("""## 3. Comparer 8 modèles

Réglage des hyperparamètres : recherche aléatoire, validation croisée **mélangée** à 5 plis,
sur 60 000 prêts d'entraînement. Puis chaque modèle est réentraîné sur tout l'entraînement.

Mesures sur le test, avec un intervalle à 95 % par **bootstrap** (on retire 300 fois les prêts au hasard) :
- **AUC** : probabilité qu'un mauvais client ait un score plus risqué qu'un bon ;
- **Gini** = 2 × AUC − 1 (la mesure des banques) ;
- **KS** : plus grand écart entre les deux distributions de score ;
- **Brier** : erreur quadratique de la probabilité (plus bas = mieux) ;
- **PR-AUC** : aire sous la courbe précision-rappel (utile car les défauts sont rares).""")

code("""m = R("models")
rows = {k: {mm: v["test"][mm]["value"] for mm in ["auc", "gini", "ks", "brier", "pr_auc"]}
        for k, v in m["results"].items()}
table = pd.DataFrame(rows).T.sort_values("auc", ascending=False).round(4)
table""")

code("""fig("fig05_model_auc")""")

code("""d = m["lightgbm_minus_scorecard_auc"]
print(f"LightGBM - carte de score : {d['diff']:+.4f} AUC  (IC 95 % : {d['low']:+.4f} à {d['high']:+.4f})")""")

md("""Le gain du boosting sur la carte de score est réel (l'intervalle ne contient pas 0) mais
modeste : avec ces variables, la limite vient surtout de l'information disponible, pas du modèle.""")

md("""## 4. Courbes d'apprentissage

Est-ce que plus de données aideraient ? On entraîne sur 2 000 à 338 000 prêts et on mesure l'AUC en validation.""")

code("""fig("fig07_learning_curves")""")

md("""## 5. Calibration : une PD doit être une vraie probabilité

Pour IFRS 9 et le prix du crédit, il ne suffit pas de bien classer : si le modèle dit 10 %,
il faut ~10 % de défauts. On recalibre sur la validation avec **Platt** (logistique sur le score)
et **isotonique** (fonction en escalier croissante), et on choisit la méthode par validation croisée.""")

code("""c = R("calibration")
print("modèle champion :", c["champion"], " | méthode retenue :", c["chosen_method"])
pd.DataFrame(c["test"]).T[["auc", "brier", "ece", "mean_pd"]].round(4)""")

code("""fig("fig08_calibration")""")

md("""## 6. Expliquer les prédictions avec SHAP

SHAP découpe chaque prédiction en contributions de chaque variable (en log-odds).
La somme des contributions + la moyenne = la prédiction, exactement.
On explique LightGBM (SHAP exact et rapide pour les arbres) : son AUC est presque celle du MLP champion.""")

code("""s = R("shap")
print("erreur d'additivité max :", s["additivity_max_error"])
pd.Series(s["mean_abs_shap_grouped"]).head(8).round(4)""")

code("""fig("fig10_shap_beeswarm")""")

code("""fig("fig11_shap_waterfall")""")

md("""## 7. Stabilité et sous-groupes

**PSI** (population stability index) : la population qui arrive ressemble-t-elle à celle
de l'entraînement ? < 0,10 stable ; 0,10–0,25 à surveiller ; > 0,25 changement important.""")

code("""st = R("stability")
print("PSI du score, train vs test :", round(st["psi_score"]["train_vs_test"], 3))
pd.DataFrame(st["psi_by_quarter"]).T.round(3).head(12)""")

code("""fig("fig12_psi")""")

code("""pd.DataFrame(st["groups"]["income_quintile"]).round(3)""")

md("""## 8. IFRS 9 : de la PD à la perte attendue (ECL)

Portefeuille : les prêts du test, vus au **30 juin 2016**.
- **Stage 1** : pas de dégradation → perte attendue sur **12 mois** ;
- **Stage 2** : dégradation significative (30 jours de retard, ou PD annualisée ×2) → perte sur **toute la durée restante** ;
- **Stage 3** : en défaut (> 90 jours de retard) → PD = 100 %.

ECL = somme sur les mois de PD marginale × LGD × EAD × actualisation.
La LGD vient des récupérations réelles sur les prêts en perte ; l'EAD du tableau d'amortissement.""")

code("""e = R("ifrs9")
print(f"LGD (pondérée par l'EAD) : {e['lgd_ead_weighted']:.1%}")
print(f"part des défauts dans les 12 premiers mois : {e['timing_share_first_12m']:.1%}")
pd.DataFrame(e["stages"]).T.round(3)""")

code("""fig("fig14_timing_lgd")""")

md("""Trois scénarios macroéconomiques : le chômage américain baisse de 1 point, reste stable,
ou monte de 3 points. Le lien chômage → défaut est estimé sur les données FRED (taux de
pertes des banques sur les crédits à la consommation depuis 1985).""")

code("""pd.DataFrame(e["scenarios"]).T.round(2)""")

code("""fig("fig16_ecl")""")

code("""bt = e["backtest"]
print(f"défauts prédits après le 30-06-2016 : {bt['predicted_remaining_lifetime_pd']:.1%}")
print(f"défauts réalisés                     : {bt['realised_default_rate']:.1%}")
print(f"perte prédite (base, durée de vie)   : {bt['predicted_lifetime_loss_undiscounted_base']/1e6:.1f} M USD")
print(f"perte réalisée                       : {bt['realised_principal_loss']/1e6:.1f} M USD")""")

md("""## 9. Le seuil d'octroi

Refuser un bon client coûte sa marge ; accepter un mauvais client coûte sa perte.
Les deux coûts sont mesurés sur les prêts réels de l'entraînement. Le seuil optimal théorique
(Bayes) est PD* = coût du refus / (coût du refus + coût du défaut).""")

code("""t = R("threshold")
print(f"coût d'un bon client refusé : {t['c_fp_refuse_good']:.1%} du montant")
print(f"coût d'un défaut accepté    : {t['c_fn_accept_bad']:.1%} du montant")
print(f"seuil de Bayes : {t['bayes_cutoff']:.1%}   seuil qui maximise le profit (validation) : {t['profit_max_cutoff_valid']:.1%}")
pd.DataFrame(t["test"]).T.round(3)""")

code("""fig("fig17_threshold")""")

md("""## Limites (à dire franchement)
- Lending Club n'est pas une banque suisse : prêts non garantis américains, 2007–2015.
- La date de défaut et les retards au 30-06-2016 sont reconstruits à partir de la date du dernier paiement.
- Pas de nouvelle note de crédit mensuelle : le test de dégradation (SICR) est simplifié.
- Les poids et les chocs des scénarios sont des hypothèses ; le lien macro vient d'un autre portefeuille (banques US).
- Pas d'âge ni de sexe dans les données : l'équité est vérifiée par segment, pas sur des attributs protégés.""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).with_name("walkthrough.ipynb"))
print("written", len(cells), "cells")
