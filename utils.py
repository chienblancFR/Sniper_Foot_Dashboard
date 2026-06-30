"""
utils.py — Fonctions utilitaires partagées par le dashboard Streamlit.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ──────────────────────────────────────────────────────────────
# 🔒  AUTHENTIFICATION
# ──────────────────────────────────────────────────────────────
def verifier_authentification():
    """
    Affiche un écran de connexion par mot de passe.
    Le mot de passe est stocké dans st.secrets["MOT_DE_PASSE"]
    ou dans la variable d'environnement DASHBOARD_PASSWORD.
    """
    import os
    mot_de_passe_attendu = None

    try:
        mot_de_passe_attendu = st.secrets["MOT_DE_PASSE"]
    except Exception:
        mot_de_passe_attendu = os.environ.get("DASHBOARD_PASSWORD", "sniper2025")

    if "authentifie" not in st.session_state:
        st.session_state["authentifie"] = False

    if not st.session_state["authentifie"]:
        st.markdown("## 🔒 Accès Restreint")
        mdp = st.text_input("Mot de passe :", type="password", key="mdp_input")
        if st.button("Connexion"):
            if mdp == mot_de_passe_attendu:
                st.session_state["authentifie"] = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
        st.stop()


# ──────────────────────────────────────────────────────────────
# 🧹  NETTOYAGE DES DONNÉES
# ──────────────────────────────────────────────────────────────
def nettoyer_colonnes_numeriques(df: pd.DataFrame, colonnes: list) -> pd.DataFrame:
    """Convertit les colonnes spécifiées en float, remplace les erreurs par NaN."""
    for col in colonnes:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def convertir_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte et convertit la colonne de date (Date, date, kickoff…) en datetime naive."""
    candidates = ['Date', 'date', 'Kickoff', 'kickoff', 'datetime']
    for col in candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Supprimer le fuseau horaire si présent pour uniformiser les comparaisons
            if hasattr(df[col].dt, 'tz') and df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
            if col != 'Date':
                df.rename(columns={col: 'Date'}, inplace=True)
            break
    return df


# ──────────────────────────────────────────────────────────────
# 🛑  ALERTES DE CHARGEMENT
# ──────────────────────────────────────────────────────────────
def afficher_alertes_chargement(statut: str, df: pd.DataFrame, msg_succes: str = ""):
    """
    Affiche un message selon le statut du chargement.
    Ne fait rien si statut == 'ok' et df non vide.
    """
    if statut == "error":
        st.error("⚠️ Impossible de charger les données. Vérifiez l'URL source ou la connexion.")
        st.stop()
    elif statut == "empty" or (statut == "ok" and df.empty):
        if msg_succes:
            st.info(msg_succes)
    # statut == "ok" et df non vide : on ne fait rien, la page s'affiche normalement


# ──────────────────────────────────────────────────────────────
# 🔍  FILTRES SIDEBAR
# ──────────────────────────────────────────────────────────────
def filtre_temporel_sidebar(df: pd.DataFrame, key_prefix: str = "live") -> pd.DataFrame:
    """
    Ajoute un filtre temporel (7j / 30j / 90j / Tout) dans la sidebar.
    key_prefix évite les collisions entre onglets Live / Back-test.
    """
    if df.empty or 'Date' not in df.columns:
        return df

    st.sidebar.subheader("📅 Période")
    periode = st.sidebar.radio(
        "Afficher les",
        options=["7 derniers jours", "30 derniers jours", "90 derniers jours", "Tout"],
        index=3,
        key=f"{key_prefix}_filtre_periode",
    )

    if periode != "Tout":
        jours = {"7 derniers jours": 7, "30 derniers jours": 30, "90 derniers jours": 90}[periode]
        date_min = pd.Timestamp.now() - pd.Timedelta(days=jours)
        df = df[df['Date'] >= date_min]

    return df


def filtre_ligue_sidebar(
    df: pd.DataFrame,
    key: str = "live_ligue",
    label_toutes: str = "Toutes les Ligues",
) -> pd.DataFrame:
    """Filtre par compétition dans la sidebar."""
    if df.empty or "Nom_Ligue" not in df.columns:
        return df
    ligues_dispo = sorted(df["Nom_Ligue"].unique().tolist())
    ligue_choisie = st.sidebar.selectbox(
        "🏆 Compétition :", [label_toutes] + ligues_dispo, key=key
    )
    if ligue_choisie != label_toutes:
        df = df[df["Nom_Ligue"] == ligue_choisie]
    return df


def filtre_marche_sidebar(
    df: pd.DataFrame,
    key: str = "live_marche",
    label_tous: str = "Tous les Marchés",
) -> pd.DataFrame:
    """Filtre par type de marché dans la sidebar."""
    if df.empty or "Type_Marche" not in df.columns:
        return df
    marches_dispo = sorted(df["Type_Marche"].unique().tolist())
    marche_choisi = st.sidebar.selectbox(
        f"📊 Marché ciblé :", [label_tous] + marches_dispo, key=key
    )
    if marche_choisi != label_tous:
        df = df[df["Type_Marche"] == marche_choisi]
    return df


# ──────────────────────────────────────────────────────────────
# 📉  CALCUL DU DRAWDOWN
# ──────────────────────────────────────────────────────────────
def calculer_max_drawdown(df: pd.DataFrame, col_profit: str, capital_initial: float):
    """
    Calcule le max drawdown en % et ajoute une colonne 'Bankroll' au DataFrame.
    Retourne (df_enrichi, max_drawdown_pct).
    """
    df = df.copy()
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)

    df['Bankroll'] = capital_initial + df[col_profit].cumsum()
    peak = df['Bankroll'].cummax()
    drawdown = (peak - df['Bankroll']) / peak
    max_dd_pct = drawdown.max() * 100 if not drawdown.empty else 0.0
    return df, max_dd_pct


def preparer_df_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise le CSV back-test : Profit_Unites = resultat × mise, tri chronologique.
    Colonne date_utc optionnelle (CSV regénéré après --report).
    """
    if df.empty:
        return df
    df = df.copy()
    cols_num = ['resultat', 'mise', 'clv', 'ev_modele', 'kelly', 'cote_h24', 'cote_cloture', 'h_val']
    df = nettoyer_colonnes_numeriques(df, cols_num)
    df['Profit_Unites'] = df['resultat'] * df['mise']
    if 'date_utc' in df.columns:
        df['Date'] = pd.to_datetime(df['date_utc'], errors='coerce', utc=True)
        if hasattr(df['Date'].dt, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        df = df.sort_values('Date', na_position='last').reset_index(drop=True)
    else:
        df = df.sort_values('fixture_id').reset_index(drop=True)
    return df


def calculer_drawdown_serie(profits, capital_initial: float = 100.0) -> float:
    """Max drawdown % — même logique que generer_rapport() dans backtest_football.py."""
    bankroll = capital_initial
    peak = bankroll
    max_dd = 0.0
    for p in profits:
        bankroll += float(p)
        if bankroll > peak:
            peak = bankroll
        if peak > 0:
            max_dd = max(max_dd, (peak - bankroll) / peak)
    return max_dd * 100


# ──────────────────────────────────────────────────────────────
# 📈  GRAPHIQUES PLOTLY
# ──────────────────────────────────────────────────────────────
def appliquer_theme_dark(fig: go.Figure):
    """Applique le thème sombre cohérent à toutes les figures Plotly."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF', size=12),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.08)',
            linecolor='rgba(255,255,255,0.2)',
            tickcolor='rgba(255,255,255,0.3)',
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.08)',
            linecolor='rgba(255,255,255,0.2)',
            tickcolor='rgba(255,255,255,0.3)',
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )


def creer_graphique_bankroll(
    df: pd.DataFrame,
    hover_data: list = None,
    couleur: str = '#00FF00',
    unite: str = 'Unités'
) -> go.Figure:
    """
    Courbe de croissance de la bankroll (valeur cumulée du P&L).
    Attend une colonne 'Bankroll' dans df (générée par calculer_max_drawdown).
    """
    fig = go.Figure()

    x_vals = df['Date'] if 'Date' in df.columns else df.index

    # Zone de remplissage sous la courbe
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=df['Bankroll'],
        mode='lines',
        fill='tozeroy',
        line=dict(color=couleur, width=2.5),
        fillcolor=f'rgba({_hex_to_rgb(couleur)},0.10)',
        name=f'Capital ({unite})',
        hovertemplate=(
            '<b>%{x|%d/%m/%Y}</b><br>'
            f'Capital : %{{y:.2f}} {unite}<extra></extra>'
        )
    ))

    fig.update_layout(
        yaxis_title=f"Capital ({unite})",
        xaxis_title="",
        showlegend=False,
        hovermode='x unified',
    )
    appliquer_theme_dark(fig)
    return fig


def creer_graphique_pl_marche(
    df: pd.DataFrame,
    col_x: str,
    titre_x: str = "Catégorie",
    titre_y: str = "P&L"
) -> go.Figure:
    """
    Graphique à barres : P&L par catégorie (marché, ligue…).
    Attend les colonnes P_and_L et Volume dans df.
    """
    couleurs = ['#00FF00' if v >= 0 else '#FF4500' for v in df['P_and_L']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[col_x],
        y=df['P_and_L'],
        marker_color=couleurs,
        text=[f"{v:+.2f} u<br>{n} paris" for v, n in zip(df['P_and_L'], df['Volume'])],
        textposition='outside',
        name='P&L'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="#FFFFFF", opacity=0.4)
    fig.update_layout(
        xaxis_title=titre_x,
        yaxis_title=titre_y,
        showlegend=False,
    )
    appliquer_theme_dark(fig)
    return fig


def creer_graphique_clv_cumule(
    df_clv: pd.DataFrame,
    col_marche: str = "Type_Marche",
) -> go.Figure:
    """CLV moyen cumulé par marché — axe X = Date si disponible."""
    fig = go.Figure()
    couleurs = {"Totals (Buts)": "#FF4500", "Handicap Asiatique": "#00BFFF"}
    a_dates = "Date" in df_clv.columns and df_clv["Date"].notna().any()

    for marche in df_clv[col_marche].unique():
        df_cat = df_clv[df_clv[col_marche] == marche].sort_values("Date").reset_index(drop=True)
        df_cat["CLV_Pct"] = df_cat["CLV"] * 100
        df_cat["CLV_Moy_Cumulee"] = df_cat["CLV_Pct"].expanding().mean()
        axe_x = df_cat["Date"] if a_dates else df_cat.index
        fig.add_trace(go.Scatter(
            x=axe_x,
            y=df_cat["CLV_Moy_Cumulee"],
            mode="lines",
            name=marche,
            line=dict(color=couleurs.get(marche, "#FFFFFF"), width=2.5),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#FFFFFF", opacity=0.5)
    fig.update_layout(
        yaxis_title="Beat The Close moyen (%)",
        xaxis_title="Date du pari" if a_dates else "Volume de paris",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    appliquer_theme_dark(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# 🔧  HELPERS INTERNES
# ──────────────────────────────────────────────────────────────
def _hex_to_rgb(hex_color: str) -> str:
    """Convertit '#RRGGBB' en 'R,G,B' pour les rgba() Plotly."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r},{g},{b}"
