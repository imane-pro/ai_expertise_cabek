"""
Cabek — Expertise Automobile IA
================================
Interface Streamlit pour l'analyse automatisée de dommages véhicule :
détection des pièces, détection des dommages, association pièce ↔ dommage,
calcul d'un indice de gravité, estimation de coût de réparation.

Lancement :
    streamlit run app_cabek_expertise_ia.py

Dépendances :
    pip install streamlit ultralytics opencv-python-headless pillow numpy plotly
  

"""
import streamlit as st
import time
from pathlib import Path

import cv2
import numpy as np

from bareme_c2 import (
    calculer_cout_dommage,
    TARIF_REPARATION_DH,
    TARIF_MOP_DH,
    AI_PIECE_VERS_CANONICAL,
    DOMMAGES_HORS_BAREME_REPARATION,
)

import streamlit as st
from PIL import Image
from ultralytics import YOLO

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

# ══════════════════════════════════════════════════════════════════
# 1. CONFIGURATION DE LA PAGE
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Cabek — Expertise IA",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# 2. DESIGN SYSTEM — CSS
# ══════════════════════════════════════════════════════════════════
# Direction : tableau de bord / instrument de diagnostic automobile.
# Fond graphite profond, accents "voyants" (ambre / rouge / vert de
# jauge), typographie technique (Rajdhani pour les titres — évoque un
# HUD — Inter pour le texte, JetBrains Mono pour les données chiffrées).

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:          #F3F5F9;
    --panel:       #FFFFFF;
    --panel-2:     #F0F2F7;
    --line:        #E1E5EC;
    --text:        #1B2028;
    --text-dim:    #6B7280;
    --accent:      #2F5FE0;
    --accent-soft: rgba(47,95,224,0.10);
    --sev-low:     #1E9E5A;
    --sev-mid:     #C4870F;
    --sev-high:    #D93B32;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.stApp {
    background: var(--bg);
}

/* ── Bandeau supérieur "scan line" ── */
.cabek-scanline {
    height: 2px;
    width: 100%;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    background-size: 200% 100%;
    animation: scan 3.5s linear infinite;
    margin-bottom: 28px;
    opacity: 0.8;
}
@keyframes scan {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ── En-tête ── */
.cabek-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 4px;
}
.cabek-wordmark {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    letter-spacing: 0.02em;
    color: var(--text);
    line-height: 1;
}
.cabek-wordmark span { color: var(--accent); }
.cabek-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-dim);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.cabek-subhead {
    color: var(--text-dim);
    font-size: 0.95rem;
    margin-top: 6px;
    margin-bottom: 28px;
}

/* ── Cartes ── */
.cabek-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.cabek-card-title {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 12px;
}

/* ── Ligne de dommage ── */
.cabek-damage-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    gap: 16px;
}
.cabek-damage-row.sev-low   { border-left-color: var(--sev-low); }
.cabek-damage-row.sev-mid   { border-left-color: var(--sev-mid); }
.cabek-damage-row.sev-high  { border-left-color: var(--sev-high); }

/* ── Colonne gauche (nom / méta / détail coût) ── */
.cabek-damage-left {
    flex: 1 1 auto;
    min-width: 0;
}
.cabek-damage-name {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    text-transform: capitalize;
    color: var(--text);
}
.cabek-damage-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-top: 2px;
}
.cabek-cost-detail {
    opacity: 0.85;
    margin-top: 4px;
}

/* ── Colonne droite (badge / coût total) ── */
.cabek-damage-right {
    flex: 0 0 auto;
    text-align: right;
    white-space: nowrap;
}
.cabek-damage-cost {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text);
    margin-top: 6px;
}

.cabek-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    white-space: nowrap;
}
.cabek-badge.sev-low  { background: rgba(30,158,90,0.12); color: var(--sev-low); }
.cabek-badge.sev-mid  { background: rgba(196,135,15,0.12); color: var(--sev-mid); }
.cabek-badge.sev-high { background: rgba(217,59,50,0.12);  color: var(--sev-high); }

/* ── Etats de chiffrage ── */
.cabek-damage-cost.replacement { color: var(--sev-high); }
.cabek-damage-cost.included { color: var(--accent); }

/* ── État vide ── */
.cabek-empty {
    text-align: center;
    padding: 48px 24px;
    color: var(--text-dim);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    border: 1px dashed var(--line);
    border-radius: 10px;
}

/* ── KPI ── */
.cabek-kpi-value {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    line-height: 1;
    color: var(--text);
}
.cabek-kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Widgets Streamlit ── */
.stButton > button {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 0.5rem 1.2rem;
}
.stButton > button:hover { background: #3D6BE0; }

[data-testid="stFileUploaderDropzone"] {
    background: var(--panel-2);
    border: 1.5px dashed var(--line);
    border-radius: 10px;
}

hr { border-color: var(--line); }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 3. CONSTANTES MÉTIER
# ══════════════════════════════════════════════════════════════════

POIDS_CATEGORIE = {
    "d_brise":           0.9,
    "d_piece_manquante": 0.9,
    "d_casse":           0.8,
    "d_lampe_casse":     0.6,
    "d_bosse":           0.5,
    "d_fissure":         0.5,
    "d_crevaison":       0.7,
    "d_rayure":          0.3,
}

CLASSES_SANS_SURFACE = ["d_crevaison"]

SOLUTIONS = {
    "d_rayure":          "Polissage / peinture partielle",
    "d_bosse":           "Débosselage / redressage",
    "d_fissure":         "Réparation / mastic",
    "d_brise":           "Remplacement vitrage",
    "d_casse":           "Échange pièce",
    "d_lampe_casse":     "Réparation ou remplacement optique",
    "d_piece_manquante": "Échange pièce",
    "d_crevaison":       "Réparation / remplacement pneu",
}

# ══════════════════════════════════════════════════════════════════
# RÈGLES MÉTIER — PRIORITÉ AU REMPLACEMENT
# ══════════════════════════════════════════════════════════════════
# Une pièce qui présente un dommage imposant son remplacement est
# chiffrée une seule fois : les autres dommages de cette pièce sont
# considérés comme inclus dans le remplacement.

DOMMAGES_REMPLACEMENT_DIRECT = {
    "d_piece_manquante",
    "d_casse",
    "d_crevaison",
}

DOMMAGES_REMPLACEMENT_SI_GRAVE = {
    "d_fissure",
    "d_bosse",
    "d_brise",
    "d_lampe_casse",
}

# Classes du modèle dommages : d_bosse_slight, d_bosse_medium,
# d_bosse_severe, d_fissure_slight, etc.
SUFFIXE_GRAVITE_CLASSE = {
    "slight": "low",
    "leger": "low",
    "light": "low",
    "medium": "mid",
    "moderate": "mid",
    "modere": "mid",
    "mid": "mid",
    "severe": "high",
    "grave": "high",
    "high": "high",
}

GRAVITE_ORDRE = {"low": 0, "mid": 1, "high": 2}


def normaliser_classe_dommage(nom_classe: str):
    """Normalise une classe IA et récupère sa gravité explicite éventuelle."""
    nom = str(nom_classe).strip().lower()
    if not nom.startswith("d_"):
        return nom, None

    base, suffixe = nom.rsplit("_", 1)
    if suffixe in SUFFIXE_GRAVITE_CLASSE:
        return base, SUFFIXE_GRAVITE_CLASSE[suffixe]
    return nom, None


def gravite_max(niveau_a: str, niveau_b: str) -> str:
    """Retourne le niveau de gravité le plus élevé."""
    if GRAVITE_ORDRE.get(niveau_b, 0) > GRAVITE_ORDRE.get(niveau_a, 0):
        return niveau_b
    return niveau_a


def necessite_remplacement(item: dict) -> bool:
    """Détermine si le dommage impose le remplacement complet de la pièce."""
    type_dommage = item.get("type_brut", "")
    niveau = item.get("niveau", "low")

    if type_dommage in DOMMAGES_REMPLACEMENT_DIRECT:
        return True

    return (
        type_dommage in DOMMAGES_REMPLACEMENT_SI_GRAVE
        and niveau == "high"
    )


def appliquer_priorite_remplacement(instances: list) -> list:
    """
    Applique la règle métier au niveau de chaque pièce.

    Si une pièce contient au moins un dommage de remplacement :
      - le dommage déclencheur est marqué remplacement_requis ;
      - les autres dommages sont bloqués et leur coût est annulé ;
      - ils ne sont pas comptés comme lignes à chiffrer.
    """
    pieces_a_remplacer = {}

    for item in instances:
        piece = item.get("piece_ai_brut")
        if piece is not None and necessite_remplacement(item):
            pieces_a_remplacer.setdefault(piece, []).append(item)

    for item in instances:
        piece = item.get("piece_ai_brut")

        if piece is None or piece not in pieces_a_remplacer:
            item["remplacement_requis"] = False
            item["chiffrage_bloque"] = False
            continue

        if necessite_remplacement(item):
            item["remplacement_requis"] = True
            item["chiffrage_bloque"] = False
            item["cout"] = None
            item["cout_detail"] = {
                "type": "remplacement",
                "remplacement_requis": True,
                "total": None,
                "avertissement": (
                    "Ce dommage nécessite le remplacement complet de la pièce. "
                    "Le prix de la pièce et la main-d'œuvre de remplacement ne "
                    "sont pas couverts par le barème C2 actuel. À chiffrer "
                    "manuellement par un expert."
                ),
            }
            item["solution"] = "Remplacement complet de la pièce"
        else:
            item["remplacement_requis"] = False
            item["chiffrage_bloque"] = True
            item["cout"] = None
            item["cout_detail"] = {
                "type": "inclus_dans_remplacement",
                "remplacement_requis": True,
                "total": None,
                "avertissement": (
                    "Dommage non chiffré séparément : la pièce présente un "
                    "dommage nécessitant son remplacement complet. Ce dommage "
                    "est considéré comme inclus dans le remplacement de la pièce."
                ),
            }
            item["solution"] = "Non chiffré séparément — remplacement de la pièce requis"

    return instances

# ── Le calcul de coût réel (MO Réparation + MOP + MET, catégorie C2)
# vit maintenant dans bareme_c2.py — importé en haut du fichier.
# La liste de marques reste affichée à titre informatif (catégorie du
# véhicule), mais ne pilote plus le prix : seule la catégorie C2 est
# couverte par le barème actuel.
CATEGORIES_MARQUE = {
    "Éco":     ["Dacia", "Fiat", "Seat", "Skoda", "Lada"],
    "Standard": ["Renault", "Peugeot", "Citroën", "Ford", "Toyota", "Hyundai",
                 "Kia", "Volkswagen", "Nissan", "Opel", "Chevrolet", "Suzuki"],
    "Premium": ["Mercedes", "BMW", "Audi", "Lexus", "Volvo", "Land Rover", "Porsche"],
}


def get_categorie_marque(marque: str) -> str:
    for cat, marques in CATEGORIES_MARQUE.items():
        if marque in marques:
            return cat
    return "Standard"


# ══════════════════════════════════════════════════════════════════
# 4. CHARGEMENT DES MODÈLES (mis en cache)
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def charger_modele(chemin: str):
    return YOLO(chemin)


# ══════════════════════════════════════════════════════════════════
# 5. LOGIQUE D'ANALYSE
# ══════════════════════════════════════════════════════════════════

def calculer_overlap(mask_d: np.ndarray, mask_p: np.ndarray) -> float:
    mask_d_bool = mask_d.astype(bool)
    mask_p_bool = mask_p.astype(bool)
    intersection = np.logical_and(mask_d_bool, mask_p_bool).sum()
    surface_dommage = mask_d_bool.sum()
    if surface_dommage == 0:
        return 0.0
    return float(intersection / surface_dommage)


def position_bucket(mask_d: np.ndarray, mask_p: np.ndarray) -> str:
    """
    Situe approximativement un dommage à l'intérieur de sa pièce (quadrant
    haut/bas × gauche/droite), pour pouvoir reconnaître "le même dommage vu
    sous un autre angle" entre plusieurs photos, sans reconstruction 3D.
    Retourne "inconnu" si la pièce ou le dommage n'a pas de pixels exploitables.
    """
    ys_p, xs_p = np.where(mask_p)
    if len(xs_p) == 0:
        return "inconnu"
    x_min, x_max = xs_p.min(), xs_p.max()
    y_min, y_max = ys_p.min(), ys_p.max()

    ys_d, xs_d = np.where(mask_d)
    if len(xs_d) == 0:
        return "inconnu"
    cx, cy = xs_d.mean(), ys_d.mean()

    fx = (cx - x_min) / max(x_max - x_min, 1)
    fy = (cy - y_min) / max(y_max - y_min, 1)

    horiz = "G" if fx < 0.5 else "D"
    vert = "H" if fy < 0.5 else "B"
    return f"{vert}{horiz}"


def calculer_score_gravite(nom_dommage: str, ratio_surface: float, confiance: float,
                            alpha: float = 0.2, beta: float = 0.2, gamma: float = 0.6) -> float:
    poids = POIDS_CATEGORIE.get(nom_dommage, 0.5)
    if nom_dommage in CLASSES_SANS_SURFACE:
        beta_a = beta / (beta + gamma)
        gamma_a = gamma / (beta + gamma)
        return beta_a * confiance + gamma_a * poids
    return alpha * ratio_surface + beta * confiance + gamma * poids


def score_vers_gravite(score: float) -> str:
    if score < 0.35:
        return "low"
    elif score < 0.6:
        return "mid"
    return "high"


LABEL_GRAVITE = {"low": "Léger", "mid": "Modéré", "high": "Grave"}


def analyser(modele_pieces, modele_dommages, image_np: np.ndarray,
             conf_pieces: float, conf_dommages: float, iou_min: float,
             marque: str, nom_image: str = ""):
    res_p = modele_pieces.predict(image_np, conf=conf_pieces, agnostic_nms=True, verbose=False)[0]
    res_d = modele_dommages.predict(image_np, conf=conf_dommages, agnostic_nms=True, verbose=False)[0]

    instances = []

    if res_d.masks is not None and len(res_d.boxes) > 0:
        for i, mask_d in enumerate(res_d.masks.data):
            nom_d_ia = modele_dommages.names[int(res_d.boxes.cls[i])]
            nom_d, gravite_classe = normaliser_classe_dommage(nom_d_ia)
            conf_d = float(res_d.boxes.conf[i])
            mask_d_np = mask_d.cpu().numpy().astype(bool)
            surface_d = mask_d_np.sum()

            meilleure_piece, meilleur_overlap = None, 0.0

            if res_p.masks is not None and len(res_p.boxes) > 0:
                for j, mask_p in enumerate(res_p.masks.data):
                    nom_p = modele_pieces.names[int(res_p.boxes.cls[j])]
                    conf_p = float(res_p.boxes.conf[j])
                    mask_p_np = mask_p.cpu().numpy().astype(bool)

                    if mask_d_np.shape != mask_p_np.shape:
                        mask_d_resized = cv2.resize(
                            mask_d_np.astype(np.uint8),
                            (mask_p_np.shape[1], mask_p_np.shape[0])
                        ).astype(bool)
                    else:
                        mask_d_resized = mask_d_np

                    overlap = calculer_overlap(mask_d_resized, mask_p_np)
                    if overlap > meilleur_overlap:
                        meilleur_overlap = overlap
                        meilleure_piece = {"nom": nom_p, "conf": conf_p, "mask": mask_p_np,
                                            "mask_d_aligne": mask_d_resized}

            piece_ai_brut = None  # nom de classe BRUT (ex: "p_Aile arriere droite"), pour le barème
            bucket = "inconnu"    # position approx. du dommage sur la pièce (pour dédup multi-images)
            if meilleure_piece and meilleur_overlap >= iou_min:
                surface_p = meilleure_piece["mask"].sum()
                if surface_p >= surface_d:
                    ratio_surface = surface_d / (surface_p + 1e-6)
                    piece_ai_brut = meilleure_piece["nom"]
                    piece_nom = meilleure_piece["nom"].replace("p_", "").replace("_", " ")
                    bucket = position_bucket(meilleure_piece["mask_d_aligne"], meilleure_piece["mask"])
                else:
                    h, w = image_np.shape[:2]
                    ratio_surface = surface_d / (h * w + 1e-6)
                    piece_nom = "zone non identifiée"
            else:
                h, w = image_np.shape[:2]
                ratio_surface = surface_d / (h * w + 1e-6)
                piece_nom = "zone non identifiée"

            score = calculer_score_gravite(nom_d, ratio_surface, conf_d)
            niveau_calcule = score_vers_gravite(score)

            # Si la classe IA indique explicitement une gravité, elle est
            # utilisée comme gravité métier de la classe. Sinon on conserve
            # le calcul surface + confiance + poids.
            niveau = gravite_classe if gravite_classe is not None else niveau_calcule

            # ── Coût réel : MO Réparation + MOP + MET (catégorie C2) ──
            if piece_ai_brut is not None:
                detail_cout = calculer_cout_dommage(piece_ai_brut, nom_d, niveau)
            else:
                detail_cout = {
                    "avertissement": "Pièce non identifiée avec certitude — chiffrage impossible automatiquement.",
                    "total": None,
                }

            instances.append({
                "type": nom_d.replace("d_", "").replace("_", " "),
                "type_brut": nom_d,
                "classe_ia": nom_d_ia,
                "gravite_classe_ia": gravite_classe,
                "piece_ai_brut": piece_ai_brut,              # ex: "p_Aile arriere droite" ou None
                "position_bucket": bucket,                   # ex: "HD", "BG"... ou "inconnu"
                "source_image": nom_image,                   # nom du fichier photo d'origine
                "confiance": conf_d,
                "piece": piece_nom,
                "surface_pct": ratio_surface * 100,
                "score": score,
                "niveau": niveau,
                "solution": SOLUTIONS.get(nom_d, "À vérifier par un expert"),
                "cout": detail_cout.get("total"),          # None si non chiffrable
                "cout_detail": detail_cout,                 # détail complet (heures, avertissements...)
                "n_vues": 1,                                  # nb de photos où ce dommage a été vu (rempli après dédup)
                "images_sources": [nom_image],
                "remplacement_requis": False,
                "chiffrage_bloque": False,
            })

    return {
        "instances": instances,
        "n_pieces": len(res_p.boxes) if res_p.boxes is not None else 0,
        "n_dommages": len(res_d.boxes) if res_d.boxes is not None else 0,
        "img_pieces": res_p.plot()[:, :, ::-1],
        "img_dommages": res_d.plot()[:, :, ::-1],
    }




# ==========================================================
# DÉDUPLICATION DES DOMMAGES
# ==========================================================

def dedupliquer_instances(toutes_instances: list) -> list:
    """
    Déduplique les dommages détectés.

    RÈGLE :
    - Même pièce + même type de dommage = un seul dommage retenu.
    - Seule la détection avec le score de confiance LE PLUS ÉLEVÉ est conservée 
      (pour éviter de doubler le coût de réparation).
    - Les pièces non identifiées ne sont pas fusionnées.
    """
    groupes = {}

    for inst in toutes_instances:
        # --------------------------------------------------
        # 1. Pièce non identifiée (zone non identifiée)
        # --------------------------------------------------
        if inst.get("piece_ai_brut") is None:
            # On conserve chaque détection séparément
            cle = ("__unique__", id(inst))

        # --------------------------------------------------
        # 2. Pièce identifiée
        # --------------------------------------------------
        else:
            # Même pièce + même type de dommage
            cle = (
                inst["piece_ai_brut"],
                inst["type_brut"],
            )

        groupes.setdefault(cle, []).append(inst)

    resultat = []

    # ------------------------------------------------------
    # Selection de la détection avec la meilleure confiance
    # ------------------------------------------------------
    for membres in groupes.values():
        # Garder la détection avec la confiance maximale
        representant = max(
            membres,
            key=lambda x: x["confiance"]
        ).copy()

        # Conserver les informations sur l'historique des détections
        representant["n_vues"] = len(membres)
        representant["images_sources"] = sorted(
            {m["source_image"] for m in membres}
        )
        representant["confiance_max"] = representant["confiance"]
        representant["n_detections_fusionnees"] = len(membres) - 1

        resultat.append(representant)

    return resultat


# ==========================================================
# 2. TRAITEMENT ET AFFICHAGE STREAMLIT
# ==========================================================

# Exemple de structure d'exécution dans Streamlit :
# (Remplacez `toutes_instances` par votre variable contenant les détections brutes)

def calculer_agregats(instances: list) -> dict:
    """
    Calcule les totaux après application des règles métier.

    Un dommage secondaire d'une pièce déjà condamnée au remplacement
    n'est PAS compté comme ligne à chiffrer manuellement.
    """
    score_global = (
        float(np.mean([it["score"] for it in instances]))
        if instances else 0.0
    )

    cout_total = sum(
        it["cout"] for it in instances
        if it.get("cout") is not None
    )

    n_a_verifier = 0
    for it in instances:
        if it.get("chiffrage_bloque") is True:
            continue
        if it.get("remplacement_requis") is True:
            n_a_verifier += 1
            continue
        if it.get("cout") is None:
            n_a_verifier += 1

    total_mo_reparation = sum(
        it["cout_detail"].get("cout_reparation", 0) or 0
        for it in instances if it.get("cout") is not None
    )
    total_mo_peinture = sum(
        it["cout_detail"].get("cout_mop", 0) or 0
        for it in instances if it.get("cout") is not None
    )
    total_produit_peinture = sum(
        it["cout_detail"].get("cout_met", 0) or 0
        for it in instances if it.get("cout") is not None
    )
    total_fourniture = 0  # remplacement/fourniture hors barème actuel

    return {
        "score_global": score_global,
        "cout_total": cout_total,
        "n_a_verifier": n_a_verifier,
        "total_mo_reparation": total_mo_reparation,
        "total_mo_peinture": total_mo_peinture,
        "total_produit_peinture": total_produit_peinture,
        "total_fourniture": total_fourniture,
    }


# ══════════════════════════════════════════════════════════════════
# 6. COMPOSANTS D'AFFICHAGE
# ══════════════════════════════════════════════════════════════════

def carte_kpi(col, valeur, label):
    col.markdown(
        f"""<div class="cabek-card" style="text-align:center;">
                <div class="cabek-kpi-value">{valeur}</div>
                <div class="cabek-kpi-label">{label}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def ligne_dommage(item: dict):
    sev = item["niveau"]
    detail = item.get("cout_detail", {})

    # ============================================================
    # AFFICHAGE DU COÛT
    # ============================================================

    if item.get("chiffrage_bloque") is True:

        cout_html = "Inclus dans remplacement"
        cout_class = "included"
        detail_html = (
            "Non chiffré séparément — la pièce nécessite un remplacement complet. "
            "Ce dommage est inclus dans le remplacement."
        )

    elif item.get("remplacement_requis") is True:

        cout_html = "Remplacement"
        cout_class = "replacement"
        detail_html = (
            "Remplacement complet de la pièce — à chiffrer manuellement par un expert."
        )

    elif item.get("cout") is not None:

        cout_html = f"{item['cout']:,.0f} MAD"
        cout_class = ""

        heures_rep = detail.get("heures_reparation")
        heures_mop = detail.get("heures_mop")
        cout_met = detail.get("cout_met")

        if heures_rep is not None:
            mo_html = f"MO Réparation : {heures_rep}h × {TARIF_REPARATION_DH} DH"
        else:
            mo_html = "MO Réparation : —"

        if heures_mop is not None:
            mop_html = f"MOP : {heures_mop}h × {TARIF_MOP_DH} DH"
        else:
            mop_html = "MOP : —"

        if cout_met is not None:
            met_html = f"MET : {cout_met} MAD"
        else:
            met_html = "MET : —"

        detail_html = f"{mo_html} · {mop_html} · {met_html}"

    else:

        cout_html = "À chiffrer"
        cout_class = ""
        detail_html = (
            detail.get("avertissement")
            or detail.get("erreur")
            or "Donnée manquante"
        )

    # ============================================================
    # HTML
    # ============================================================

    n_vues = item.get("n_vues", 1)
    if n_vues > 1:
        images_liste = ", ".join(item.get("images_sources", []))
        vues_html = f"""<div class="cabek-damage-meta" style="margin-top:4px; color:var(--accent);">
            👁 Vu sur {n_vues} photos ({images_liste}) — fusionné, compté une seule fois
        </div>"""
    else:
        vues_html = ""

    html = f"""
<div class="cabek-damage-row sev-{sev}">

    <div class="cabek-damage-left">

        <div class="cabek-damage-name">
            {item['type']}
        </div>

        <div class="cabek-damage-meta">
            pièce : {item['piece']} ·
            confiance {item['confiance'] * 100:.0f}% ·
            surface touchée {item['surface_pct']:.1f}%
        </div>

        <div class="cabek-damage-meta cabek-cost-detail">
            {detail_html}
        </div>

        {vues_html}

    </div>

    <div class="cabek-damage-right">

        <span class="cabek-badge sev-{sev}">
            {LABEL_GRAVITE[sev]}
        </span>

        <div class="cabek-damage-cost {cout_class}">
            {cout_html}
        </div>

    </div>

</div>
"""

    # Enlève l'indentation qui transforme le HTML en bloc de code
    # ⚠️ IMPORTANT : on retire l'indentation de CHAQUE ligne individuellement
    # (pas juste l'indentation commune via dedent) — sinon les lignes internes
    # encore indentées de 4+ espaces sont interprétées par Markdown comme un
    # bloc de code brut, et le HTML s'affiche tel quel au lieu d'être rendu.
    html = "\n".join(ligne.strip() for ligne in html.strip().splitlines())

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════
# 7. SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """<div style="font-family:'Rajdhani',sans-serif; font-weight:700; font-size:1.5rem;">
               CABEK<span style="color:#4C7EFF;">.AI</span>
           </div>
           <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#8792A2; letter-spacing:0.06em;">
               EXPERTISE AUTOMOBILE ASSISTÉE PAR IA
           </div><hr>""",
        unsafe_allow_html=True,
    )

    st.markdown("**Modèles**")
    chemin_pieces = st.text_input("Modèle pièces (.pt)", value="best_pieces.pt")
    chemin_dommages = st.text_input("Modèle dommages (.pt)", value="best_degats_v2.pt")

    st.markdown("**Seuils de détection**")
    conf_pieces = st.slider("Confiance — pièces", 0.05, 0.95, 0.25, 0.05)
    conf_dommages = st.slider("Confiance — dommages", 0.05, 0.95, 0.15, 0.05)
    iou_min = st.slider("Overlap minimum (association)", 0.0, 0.5, 0.05, 0.01)

    st.markdown("**Photos multiples**")
    dedupliquer = st.checkbox(
        "Fusionner les dommages vus sur plusieurs photos (recommandé)",
        value=True,
        help="Évite de compter deux fois le même dommage physique photographié "
             "sous plusieurs angles (même pièce + même type de dommage + même "
             "position sur la pièce = fusionné en une seule ligne)."
    )

    st.markdown("**Véhicule**")
    toutes_marques = sum(CATEGORIES_MARQUE.values(), [])
    marque = st.selectbox("Marque", sorted(toutes_marques), index=sorted(toutes_marques).index("Renault"))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Cabek — Expertise Automobile · Outil interne d'aide au diagnostic. "
               "Les estimations ne remplacent pas la validation d'un expert agréé.")


# ══════════════════════════════════════════════════════════════════
# 8. EN-TÊTE
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="cabek-scanline"></div>', unsafe_allow_html=True)
st.markdown(
    """<div class="cabek-header">
           <div class="cabek-wordmark">Diagnostic<span>IA</span></div>
       </div>
       <div class="cabek-tagline">Analyse automatisée des dommages véhicule</div>
       <div class="cabek-subhead">
           Déposez une photo du véhicule. Le système détecte les pièces et les dommages,
           associe chaque dommage à sa pièce, puis calcule un indice de gravité et une
           estimation de réparation.
       </div>""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════
# 9. ZONE D'UPLOAD & ANALYSE (plusieurs photos)
# ══════════════════════════════════════════════════════════════════

fichiers = st.file_uploader(
    "Photos du véhicule",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not fichiers:
    st.markdown(
        """<div class="cabek-empty">
               EN ATTENTE D'IMAGE(S)<br><br>
               Glissez une ou plusieurs photos du véhicule ci-dessus (JPG ou PNG) pour
               lancer l'analyse. Plusieurs angles du même dommage sont automatiquement
               reconnus et fusionnés (option activable dans la barre latérale).
           </div>""",
        unsafe_allow_html=True,
    )
    st.stop()

# Chargement des modèles (une seule fois, même pour plusieurs photos)
try:
    with st.spinner("Chargement des modèles..."):
        modele_pieces = charger_modele(chemin_pieces)
        modele_dommages = charger_modele(chemin_dommages)
except Exception as e:
    st.error(
        f"Impossible de charger les modèles. Vérifiez les chemins renseignés dans la barre "
        f"latérale.\n\nDétail : {e}"
    )
    st.stop()

# ── Analyse de CHAQUE photo séparément ──
t0 = time.time()
resultats_par_image = []
with st.spinner(f"Analyse de {len(fichiers)} photo(s) en cours..."):
    for f in fichiers:
        image_pil = Image.open(f).convert("RGB")
        image_np = np.array(image_pil)
        r = analyser(modele_pieces, modele_dommages, image_np,
                     conf_pieces, conf_dommages, iou_min, marque, nom_image=f.name)
        resultats_par_image.append({"nom": f.name, **r})
duree = time.time() - t0

# ══════════════════════════════════════════════════════════════════
# 10. RÉSULTATS — IMAGES (galerie par onglets, une photo par onglet)
# ══════════════════════════════════════════════════════════════════

if len(resultats_par_image) == 1:
    r = resultats_par_image[0]
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="cabek-card-title">Pièces détectées</div>', unsafe_allow_html=True)
        st.image(r["img_pieces"], use_container_width=True)
    with col_b:
        st.markdown('<div class="cabek-card-title">Dommages détectés</div>', unsafe_allow_html=True)
        st.image(r["img_dommages"], use_container_width=True)
else:
    onglets = st.tabs([r["nom"] for r in resultats_par_image])
    for onglet, r in zip(onglets, resultats_par_image):
        with onglet:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="cabek-card-title">Pièces détectées</div>', unsafe_allow_html=True)
                st.image(r["img_pieces"], use_container_width=True)
            with col_b:
                st.markdown('<div class="cabek-card-title">Dommages détectés</div>', unsafe_allow_html=True)
                st.image(r["img_dommages"], use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 10ter. FUSION DES DOMMAGES ENTRE PHOTOS
# ══════════════════════════════════════════════════════════════════

toutes_instances = [it for r in resultats_par_image for it in r["instances"]]

if dedupliquer:
    instances_finales = dedupliquer_instances(toutes_instances)

    n_fusionnes = sum(
        1 for it in instances_finales
        if it.get("n_detections_fusionnees", 0) > 0
    )

    if n_fusionnes > 0:
        st.markdown(
            f"""
            🔗 {n_fusionnes} dommage(s) détecté(s) plusieurs fois
            et fusionné(s) automatiquement — comptés une seule fois.
            """,
            unsafe_allow_html=True,
        )
else:
    instances_finales = toutes_instances

# ══════════════════════════════════════════════════════════════════
# PRIORITÉ AU REMPLACEMENT — après fusion multi-photos
# ══════════════════════════════════════════════════════════════════
instances_finales = appliquer_priorite_remplacement(instances_finales)

agregats = calculer_agregats(instances_finales)
n_pieces_total = sum(r["n_pieces"] for r in resultats_par_image)

# ══════════════════════════════════════════════════════════════════
# 10bis. RÉSULTATS — DÉCOMPOSITION DU COÛT PAR POSTE
# ══════════════════════════════════════════════════════════════════
# Récap des 4 postes du barème (uniquement sur les dommages déjà chiffrés) :
# Main d'œuvre Tôlerie (MO Réparation), Main d'œuvre Peinture (MOP),
# Produit peinture (MET), Fourniture (prix pièce — non couvert, cf. bareme_c2.py).

if instances_finales:
    st.markdown('<div class="cabek-card">', unsafe_allow_html=True)
    st.markdown('<div class="cabek-card-title">Décomposition du coût par poste</div>', unsafe_allow_html=True)

    cp1, cp2, cp3, cp4 = st.columns(4)
    carte_kpi(cp1, f"{agregats['total_mo_reparation']:,.0f} MAD", "Main d'œuvre Tôlerie")
    carte_kpi(cp2, f"{agregats['total_mo_peinture']:,.0f} MAD", "Main d'œuvre Peinture")
    carte_kpi(cp3, f"{agregats['total_produit_peinture']:,.0f} MAD", "Produit peinture (MET)")
    carte_kpi(cp4, f"{agregats['total_fourniture']:,.0f} MAD", "Fourniture")

    st.markdown(
        """<div class="cabek-damage-meta" style="margin-top:4px;">
               Fourniture = prix de la pièce en cas de remplacement — non couverte par ce
               barème (MOT volontairement exclu, cf. lignes "à chiffrer manuellement").
           </div>""",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 11. RÉSULTATS — KPI
# ══════════════════════════════════════════════════════════════════

if not instances_finales:
    st.markdown(
        """<div class="cabek-empty">
               AUCUN DOMMAGE DÉTECTÉ<br><br>
               Aucune anomalie n'a franchi le seuil de confiance actuel sur les photos fournies.
               Essayez de réduire le seuil « Confiance — dommages » dans la barre latérale, ou
               vérifiez le cadrage des photos.
           </div>""",
        unsafe_allow_html=True,
    )
    st.stop()

k1, k2, k3, k4 = st.columns(4)
carte_kpi(k1, len(instances_finales), "Dommages détectés (après fusion)")
carte_kpi(k2, n_pieces_total, "Pièces identifiées (toutes photos)")
carte_kpi(k3, f"{agregats['cout_total']:,.0f} MAD", "Coût estimé (C2)")
carte_kpi(k4, agregats["n_a_verifier"], "Lignes à chiffrer manuellement")

if agregats["n_a_verifier"] > 0:
    st.markdown(
        f"""<div class="cabek-card" style="border-left:3px solid var(--sev-mid);">
                ⚠️ {agregats['n_a_verifier']} ligne(s) nécessitent une intervention de l'expert
                pour le chiffrage (remplacement, pièce non identifiée ou donnée absente du barème).
                Les dommages secondaires d'une pièce à remplacer ne sont pas comptés séparément.
            </div>""",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# 12. RÉSULTATS — DÉTAIL PAR DOMMAGE (après fusion multi-photos)
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="cabek-card">', unsafe_allow_html=True)
st.markdown('<div class="cabek-card-title">Détail par dommage</div>', unsafe_allow_html=True)

instances_triees = sorted(instances_finales, key=lambda x: x["score"], reverse=True)
for item in instances_triees:
    ligne_dommage(item)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 13. EXPORT DU RAPPORT PDF — DESIGN CABEK
# ══════════════════════════════════════════════════════════════════

def generer_rapport_pdf(
    instances_triees,
    agregats,
    marque,
    n_photos,
    tarif_reparation,
    tarif_mop,
):
    """
    Génère un rapport PDF professionnel CABEK à partir
    des résultats de l'analyse IA.
    """

    buffer = BytesIO()

    # --------------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------------

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="Rapport d'expertise CABEK",
        author="CABEK",
    )

    # --------------------------------------------------------------
    # COULEURS CABEK
    # --------------------------------------------------------------

    GRAPHITE = colors.HexColor("#18212B")
    DARK = colors.HexColor("#26313B")
    GRAY = colors.HexColor("#66717D")
    LIGHT_GRAY = colors.HexColor("#F3F5F7")
    BORDER = colors.HexColor("#D9DEE3")
    WHITE = colors.white
    ACCENT = colors.HexColor("#D99A2B")
    GREEN = colors.HexColor("#3D8B65")
    RED = colors.HexColor("#C94A4A")

    # --------------------------------------------------------------
    # STYLES
    # --------------------------------------------------------------

    styles = getSampleStyleSheet()

    style_titre = ParagraphStyle(
        "TitreCabek",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=GRAPHITE,
        spaceAfter=4,
    )

    style_sous_titre = ParagraphStyle(
        "SousTitreCabek",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=GRAY,
    )

    style_section = ParagraphStyle(
        "SectionCabek",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=GRAPHITE,
        spaceBefore=8,
        spaceAfter=7,
    )

    style_normal = ParagraphStyle(
        "NormalCabek",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=DARK,
    )

    style_small = ParagraphStyle(
        "SmallCabek",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=GRAY,
    )

    style_kpi_value = ParagraphStyle(
        "KPIValue",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        alignment=TA_CENTER,
        textColor=GRAPHITE,
    )

    style_kpi_label = ParagraphStyle(
        "KPILabel",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        alignment=TA_CENTER,
        textColor=GRAY,
    )

    # --------------------------------------------------------------
    # HEADER / FOOTER
    # --------------------------------------------------------------

    def header_footer(canvas, doc):
        canvas.saveState()

        width, height = A4

        # Ligne supérieure
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.7)
        canvas.line(
            16 * mm,
            height - 14 * mm,
            width - 16 * mm,
            height - 14 * mm,
        )

        # Logo texte CABEK
        canvas.setFont("Helvetica-Bold", 10)
        canvas.setFillColor(GRAPHITE)

        canvas.drawString(
            16 * mm,
            height - 10.5 * mm,
            "CABEK"
        )

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GRAY)

        canvas.drawString(
            34 * mm,
            height - 10.5 * mm,
            "EXPERTISE AUTOMOBILE ASSISTÉE PAR IA"
        )

        # Footer
        canvas.setStrokeColor(BORDER)

        canvas.line(
            16 * mm,
            12 * mm,
            width - 16 * mm,
            12 * mm,
        )

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GRAY)

        canvas.drawString(
            16 * mm,
            7 * mm,
            "CABEK — Rapport d'expertise automobile"
        )

        canvas.drawRightString(
            width - 16 * mm,
            7 * mm,
            f"Page {doc.page}"
        )

        canvas.restoreState()

    # --------------------------------------------------------------
    # CONTENU
    # --------------------------------------------------------------

    story = []

    # ==============================================================
    # EN-TÊTE
    # ==============================================================

    story.append(Spacer(1, 5 * mm))

    story.append(
        Paragraph(
            "CABEK",
            style_titre
        )
    )

    story.append(
        Paragraph(
            "RAPPORT D'EXPERTISE AUTOMOBILE",
            style_titre
        )
    )

    story.append(
        Paragraph(
            "Analyse automatisée des dommages véhicule par intelligence artificielle",
            style_sous_titre
        )
    )

    story.append(Spacer(1, 7 * mm))

    # ==============================================================
    # INFORMATIONS GÉNÉRALES
    # ==============================================================

    info_data = [
        [
            "Catégorie",
            "C2",
            "Marque",
            marque,
        ],
        [
            "Photos analysées",
            str(n_photos),
            "Dommages détectés",
            str(len(instances_triees)),
        ],
    ]

    info_table = Table(
        info_data,
        colWidths=[
            30 * mm,
            50 * mm,
            35 * mm,
            50 * mm,
        ],
    )

    info_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
            ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),

            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),

            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTNAME", (3, 0), (3, -1), "Helvetica"),

            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, -1), DARK),

            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(info_table)

    story.append(Spacer(1, 8 * mm))

    # ==============================================================
    # KPI
    # ==============================================================

    kpi_data = [
        [
            Paragraph(
                str(len(instances_triees)),
                style_kpi_value,
            ),

            Paragraph(
                str(n_pieces_total),
                style_kpi_value,
            ),

            Paragraph(
                f"{agregats['cout_total']:,.0f} MAD",
                style_kpi_value,
            ),

            Paragraph(
                str(agregats["n_a_verifier"]),
                style_kpi_value,
            ),
        ],

        [
            Paragraph(
                "Dommages détectés",
                style_kpi_label,
            ),

            Paragraph(
                "Pièces identifiées",
                style_kpi_label,
            ),

            Paragraph(
                "Coût estimé",
                style_kpi_label,
            ),

            Paragraph(
                "À vérifier",
                style_kpi_label,
            ),
        ],
    ]

    kpi_table = Table(
        kpi_data,
        colWidths=[44 * mm] * 4,
    )

    kpi_table.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("BACKGROUND", (0, 0), (-1, -1), WHITE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(kpi_table)

    story.append(Spacer(1, 8 * mm))

    # ==============================================================
    # SYNTHÈSE
    # ==============================================================

    story.append(
        Paragraph(
            "SYNTHÈSE DE L'EXPERTISE",
            style_section,
        )
    )

    score_global = agregats.get("score_global", 0)

    if score_global < 0.35:
        gravite_globale = "Légère"
    elif score_global < 0.60:
        gravite_globale = "Modérée"
    else:
        gravite_globale = "Élevée"

    synthese = (
        f"L'analyse automatisée a identifié "
        f"<b>{len(instances_triees)}</b> dommage(s) "
        f"sur le véhicule de marque <b>{marque}</b>. "
        f"L'indice de gravité global est de "
        f"<b>{score_global:.2f}</b>, correspondant à une gravité "
        f"<b>{gravite_globale}</b>. "
        f"Le coût estimatif total des lignes automatiquement "
        f"chiffrables est de "
        f"<b>{agregats['cout_total']:,.0f} MAD</b>."
    )

    story.append(
        Paragraph(
            synthese,
            style_normal,
        )
    )

    story.append(Spacer(1, 7 * mm))

    # ==============================================================
    # DÉCOMPOSITION DU COÛT
    # ==============================================================

    story.append(
        Paragraph(
            "DÉCOMPOSITION DU COÛT",
            style_section,
        )
    )

    cout_data = [
        ["Poste", "Montant"],
        [
            "Main d'œuvre Tôlerie — MO Réparation",
            f"{agregats['total_mo_reparation']:,.0f} MAD",
        ],
        [
            "Main d'œuvre Peinture — MOP",
            f"{agregats['total_mo_peinture']:,.0f} MAD",
        ],
        [
            "Produit peinture — MET",
            f"{agregats['total_produit_peinture']:,.0f} MAD",
        ],
        [
            "Fourniture",
            f"{agregats['total_fourniture']:,.0f} MAD",
        ],
        [
            "TOTAL ESTIMÉ",
            f"{agregats['cout_total']:,.0f} MAD",
        ],
    ]

    cout_table = Table(
        cout_data,
        colWidths=[125 * mm, 40 * mm],
    )

    cout_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GRAPHITE),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("BACKGROUND", (0, -1), (-1, -1), LIGHT_GRAY),

            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),

            ("ALIGN", (1, 1), (1, -1), "RIGHT"),

            ("FONTSIZE", (0, 0), (-1, -1), 8),

            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(cout_table)

    story.append(Spacer(1, 7 * mm))

    story.append(
        Paragraph(
            f"Tarif MO Réparation : <b>{tarif_reparation} MAD/h</b> "
            f"• Tarif MOP : <b>{tarif_mop} MAD/h</b> "
            f"• Type de peinture : <b>MET</b>",
            style_small,
        )
    )

    # ==============================================================
    # PAGE DOMMAGES 
    # ==============================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "DÉTAIL DES DOMMAGES",
            style_section,
        )
    )

    damage_data = [
        [
            "#",
            "Dommage",
            "Pièce",
            "Gravité",
            "Confiance",
            "Coût",
        ]
    ]

    for index, item in enumerate(instances_triees, 1):

        if item.get("chiffrage_bloque") is True:
            cout = "Inclus remplacement"
        elif item.get("remplacement_requis") is True:
            cout = "Remplacement — à chiffrer"
        elif item.get("cout") is not None:
            cout = f"{item['cout']:,.0f} MAD"
        else:
            cout = "À vérifier"

        damage_data.append([
            str(index),
            item["type"],
            item["piece"],
            LABEL_GRAVITE[item["niveau"]],
            f"{item['confiance'] * 100:.0f}%",
            cout,
        ])

    damage_table = Table(
        damage_data,
        colWidths=[
            9 * mm,
            36 * mm,
            47 * mm,
            25 * mm,
            24 * mm,
            25 * mm,
        ],
        repeatRows=1,
    )

    damage_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GRAPHITE),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),

            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),

            ("FONTSIZE", (0, 0), (-1, -1), 7.5),

            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (4, 1), (5, -1), "RIGHT"),

            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(damage_table)

    # ==============================================================
    # DÉTAILS DES DOMMAGES
    # ==============================================================

    story.append(Spacer(1, 8 * mm))

    story.append(
        Paragraph(
            "DÉTAIL DU CHIFFRAGE",
            style_section,
        )
    )

    for index, item in enumerate(instances_triees, 1):

        detail = item.get("cout_detail", {})

        if item["cout"] is not None:

            heures_rep = detail.get("heures_reparation")
            heures_mop = detail.get("heures_mop")
            cout_met = detail.get("cout_met")

            texte = (
                f"<b>{index}. {item['type']} — {item['piece']}</b><br/>"
                f"Gravité : {LABEL_GRAVITE[item['niveau']]} "
                f"• Confiance : {item['confiance'] * 100:.0f}% "
                f"• Surface touchée : {item['surface_pct']:.1f}%<br/>"
                f"MO Réparation : {heures_rep if heures_rep is not None else '—'} h "
                f"× {tarif_reparation} DH "
                f"• MOP : {heures_mop if heures_mop is not None else '—'} h "
                f"× {tarif_mop} DH "
                f"• MET : {cout_met if cout_met is not None else '—'} MAD<br/>"
                f"<b>Coût : {item['cout']:,.0f} MAD</b>"
            )

        else:

            if item.get("chiffrage_bloque") is True:
                texte = (
                    f"<b>{index}. {item['type']} — {item['piece']}</b><br/>"
                    f"Gravité : {LABEL_GRAVITE[item['niveau']]} "
                    f"• Confiance : {item['confiance'] * 100:.0f}%<br/>"
                    f"<b>INCLUS DANS LE REMPLACEMENT</b> — "
                    f"La pièce présente un dommage nécessitant son remplacement complet. "
                    f"Ce dommage n'est pas chiffré séparément."
                )
            elif item.get("remplacement_requis") is True:
                avertissement = (
                    detail.get("avertissement")
                    or "Remplacement complet de la pièce."
                )
                texte = (
                    f"<b>{index}. {item['type']} — {item['piece']}</b><br/>"
                    f"Gravité : {LABEL_GRAVITE[item['niveau']]} "
                    f"• Confiance : {item['confiance'] * 100:.0f}%<br/>"
                    f"<b>REMPLACEMENT — À CHIFFRER MANUELLEMENT</b> — {avertissement}"
                )
            else:
                avertissement = (
                    detail.get("avertissement")
                    or detail.get("erreur")
                    or "Donnée manquante"
                )
                texte = (
                    f"<b>{index}. {item['type']} — {item['piece']}</b><br/>"
                    f"Gravité : {LABEL_GRAVITE[item['niveau']]} "
                    f"• Confiance : {item['confiance'] * 100:.0f}%<br/>"
                    f"<b>À CHIFFRER MANUELLEMENT</b> — {avertissement}"
                )

        story.append(
            Paragraph(
                texte,
                style_normal,
            )
        )

        if item.get("n_vues", 1) > 1:

            images = ", ".join(
                item.get("images_sources", [])
            )

            story.append(
                Paragraph(
                    f"Vu sur {item['n_vues']} photos : {images} "
                    f"— dommage fusionné et compté une seule fois.",
                    style_small,
                )
            )

        story.append(Spacer(1, 4 * mm))

    # ==============================================================
    # MÉTHODOLOGIE
    # ==============================================================

    story.append(
        Paragraph(
            "MÉTHODOLOGIE",
            style_section,
        )
    )

    story.append(
        Paragraph(
            "Le système analyse les photographies du véhicule à l'aide de modèles "
            "de détection IA. Il identifie les pièces, détecte les dommages, "
            "associe chaque dommage à la pièce correspondante et calcule un "
            "indice de gravité. Le coût est ensuite estimé selon le barème C2. "
            "Lorsque plusieurs détections correspondent à la même pièce et au "
            "même type de dommage, la détection présentant la confiance la plus "
            "élevée est conservée afin d'éviter un double comptage. "
            "Lorsqu'une pièce présente un dommage imposant son remplacement complet "
            "(par exemple pièce manquante, casse, crevaison ou fissure/bosse grave), "
            "les autres dommages de cette même pièce ne sont pas chiffrés séparément : "
            "ils sont considérés comme inclus dans le remplacement.",
            style_normal,
        )
    )

    story.append(Spacer(1, 6 * mm))

    # ==============================================================
    # AVERTISSEMENT
    # ==============================================================

    warning_data = [[
        Paragraph(
            "<b>⚠ AVERTISSEMENT</b><br/>"
            "Ce rapport est généré automatiquement par un système d'aide "
            "au diagnostic. Les résultats de détection, la gravité et les "
            "estimations financières doivent être contrôlés et validés par "
            "un expert automobile agréé avant toute décision de réparation "
            "ou d'indemnisation.",
            style_normal,
        )
    ]]

    warning_table = Table(
        warning_data,
        colWidths=[165 * mm],
    )

    warning_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7E8")),
            ("BOX", (0, 0), (-1, -1), 0.8, ACCENT),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )

    story.append(warning_table)

    # --------------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------------

    doc.build(
        story,
        onFirstPage=header_footer,
        onLaterPages=header_footer,
    )

    buffer.seek(0)

    return buffer.getvalue()


# ==============================================================
# GÉNÉRATION DU PDF
# ==============================================================

rapport_pdf = generer_rapport_pdf(
    instances_triees=instances_triees,
    agregats=agregats,
    marque=marque,
    n_photos=len(resultats_par_image),
    tarif_reparation=TARIF_REPARATION_DH,
    tarif_mop=TARIF_MOP_DH,
)


# ==============================================================
# BOUTON TÉLÉCHARGEMENT
# ==============================================================

st.download_button(
    label="📄 Télécharger le rapport d'expertise PDF",
    data=rapport_pdf,
    file_name="rapport_expertise_CABEK.pdf",
    mime="application/pdf",
    use_container_width=True,
)