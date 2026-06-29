"""
backtest_football.py — Back-test complet du modèle Dixon-Coles sur 2 saisons
=============================================================================
Usage :
  python backtest_football.py --collect    # Phase 1 : télécharge les données
  python backtest_football.py --simulate   # Phase 2 : simule les paris
  python backtest_football.py --report     # Phase 3 : génère le rapport
  python backtest_football.py              # Les 3 phases d'un coup

Résultats dans backtest_results.csv et imprimés dans la console.
"""

import argparse
import asyncio
import aiohttp
import aiosqlite
import numpy as np
import csv
import os
import sys
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from thefuzz import process
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
API_ODDS_KEY     = os.getenv("API_ODDS_KEY")

URL_FOOTBALL = "https://v3.football.api-sports.io"
HEADERS_FB   = {"x-apisports-key": API_FOOTBALL_KEY, "v": "3"}
DB_PATH      = "backtest_data.db"

# ─────────────────────────────────────────────────────────────
# ⚙️  CONFIGURATION
# ─────────────────────────────────────────────────────────────
SAISONS_BACKTEST = [2023, 2024]  # 2023 = saison 2023-24 pour ligues hivernales

CHAMPIONNATS = [
    {"nom": "La Liga",          "id": 140, "key": "soccer_spain_la_liga",            "c1": 4,  "rel": 18, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Bundesliga",       "id": 78,  "key": "soccer_germany_bundesliga",        "c1": 4,  "rel": 16, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Eredivisie",       "id": 88,  "key": "soccer_netherlands_eredivisie",    "c1": 2,  "rel": 16, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Serie A",          "id": 135, "key": "soccer_italy_serie_a",             "c1": 4,  "rel": 18, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Primeira Liga",    "id": 94,  "key": "soccer_portugal_primeira_liga",    "c1": 2,  "rel": 16, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Süper Lig",        "id": 203, "key": "soccer_turkey_super_league",       "c1": 2,  "rel": 17, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Allsvenskan",      "id": 113, "key": "soccer_sweden_allsvenskan",        "c1": 3,  "rel": 14, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Série A Brésil",   "id": 71,  "key": "soccer_brazil_campeonato",         "c1": 6,  "rel": 17, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Ligue 1",          "id": 61,  "key": "soccer_france_ligue_one",          "c1": 4,  "rel": 16, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "LaLiga 2",         "id": 141, "key": "soccer_spain_segunda_division",    "c1": 2,  "rel": 19, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Premier League",   "id": 39,  "key": "soccer_epl",                       "c1": 4,  "rel": 18, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Championship",     "id": 40,  "key": "soccer_england_championship",      "c1": 2,  "rel": 22, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "MLS",              "id": 253, "key": "soccer_usa_mls",                   "c1": 7,  "rel": 99, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Eliteserien",      "id": 103, "key": "soccer_norway_eliteserien",        "c1": 2,  "rel": 14, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Jupiler Pro",      "id": 144, "key": "soccer_belgium_first_div",         "c1": 6,  "rel": 13, "ev_min": 0.05, "ev_max": 0.15},
    {"nom": "Serie B",          "id": 136, "key": "soccer_italy_serie_b",             "c1": 2,  "rel": 16, "ev_min": 0.05, "ev_max": 0.15},
]

RHO_PAR_LIGUE = {
    78: -0.16, 88: -0.15, 39: -0.13, 40: -0.12, 61: -0.12,
    141: -0.12, 136: -0.11, 140: -0.10, 94: -0.10, 135: -0.09,
    203: -0.08, 71: -0.08, 113: -0.11, 103: -0.10, 144: -0.12, 253: -0.09,
}
RHO_DEFAULT  = -0.12
KELLY_FRAC      = 0.05   # fraction Kelly (5% — aligné avec le bot)
MIN_COTE        = 1.70   # ignorer les handicaps trop courts (< 1.70)
N_PRIOR         = 8      # matchs-équivalents shrinkage bayésien
# Poids du modèle dans le blend EV à H-24 : poids_dyn = 0.15 + 0.15*(24/168) ≈ 0.171
# Réplique la pondération dynamique du bot pour les paris pris environ 24h avant le coup d'envoi.
POIDS_DYN_H24   = 0.171

NAME_MAPPING = {
    # 🇫🇷 LIGUE 1
    "Paris Saint-Germain": "Paris Saint Germain",
    "Olympique Lyonnais": "Lyon", "Olympique de Marseille": "Marseille",
    "Stade Rennais FC": "Rennes", "Stade Rennais": "Rennes",
    "Stade de Reims": "Reims", "AS Monaco": "Monaco", "OGC Nice": "Nice",
    "RC Lens": "Lens", "Lille OSC": "Lille", "FC Nantes": "Nantes",
    "RC Strasbourg Alsace": "Strasbourg", "RC Strasbourg": "Strasbourg",
    "Montpellier HSC": "Montpellier", "Stade Brestois 29": "Brest",
    "FC Lorient": "Lorient", "AJ Auxerre": "Auxerre", "Le Havre AC": "Le Havre",
    "Toulouse FC": "Toulouse", "Angers SCO": "Angers",
    "AS Saint-Étienne": "Saint-Etienne", "AS Saint-Etienne": "Saint-Etienne",
    "Girondins de Bordeaux": "Bordeaux",
    # 🇪🇸 LA LIGA
    "Athletic Club": "Athletic Bilbao", "Athletic Club de Bilbao": "Athletic Bilbao",
    "Atlético Madrid": "Atletico Madrid", "Deportivo Alavés": "Alaves",
    "Cádiz CF": "Cadiz", "RC Celta": "Celta Vigo", "RCD Espanyol": "Espanyol",
    "RCD Mallorca": "Mallorca", "Getafe CF": "Getafe", "CA Osasuna": "Osasuna",
    "Real Betis Balompié": "Real Betis", "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia", "Real Valladolid": "Valladolid",
    "Girona FC": "Girona", "UD Las Palmas": "Las Palmas",
    "CD Leganés": "Leganes", "Villarreal CF": "Villarreal",
    "UD Almería": "Almeria", "Granada CF": "Granada",
    # 🇪🇸 LALIGA 2
    "SD Huesca": "Huesca", "Real Oviedo": "Oviedo",
    "Sporting Gijón": "Sporting Gijon", "Sporting de Gijón": "Sporting Gijon",
    "Real Zaragoza": "Zaragoza", "SD Eibar": "Eibar", "Málaga CF": "Malaga",
    "Racing Club de Santander": "Racing Santander", "Burgos CF": "Burgos",
    "Elche CF": "Elche", "Levante UD": "Levante",
    "Albacete Balompié": "Albacete", "FC Cartagena": "Cartagena",
    "CD Tenerife": "Tenerife", "Córdoba CF": "Cordoba", "CD Eldense": "Eldense",
    # 🇩🇪 BUNDESLIGA
    "FC Bayern München": "Bayern Munich", "Bayern München": "Bayern Munich",
    "1. FC Köln": "FC Koeln", "Borussia Mönchengladbach": "Borussia Monchengladbach",
    "TSG Hoffenheim": "Hoffenheim", "TSG 1899 Hoffenheim": "Hoffenheim",
    "SC Freiburg": "Freiburg", "VfB Stuttgart": "Stuttgart",
    "1. FSV Mainz 05": "Mainz", "FC Augsburg": "Augsburg",
    "SV Werder Bremen": "Werder Bremen", "VfL Wolfsburg": "Wolfsburg",
    "VfL Bochum 1848": "Bochum", "Hertha BSC": "Hertha Berlin",
    "1. FC Union Berlin": "Union Berlin", "FC Union Berlin": "Union Berlin",
    "1. FC Heidenheim 1846": "Heidenheim", "1. FC Heidenheim": "Heidenheim",
    "FC St. Pauli": "St. Pauli", "SV Darmstadt 98": "Darmstadt 98",
    "Holstein Kiel": "Holstein Kiel",
    # 🇮🇹 SERIE A & SERIE B
    "Inter": "Inter Milan", "AC Milan": "AC Milan", "AS Roma": "Roma",
    "SS Lazio": "Lazio", "ACF Fiorentina": "Fiorentina",
    "Atalanta BC": "Atalanta", "Hellas Verona": "Verona",
    "Torino FC": "Torino", "Bologna FC 1909": "Bologna",
    "Genoa CFC": "Genoa", "US Sassuolo Calcio": "Sassuolo",
    "Udinese Calcio": "Udinese", "Cagliari Calcio": "Cagliari",
    "Empoli FC": "Empoli", "US Lecce": "Lecce",
    "Parma Calcio 1913": "Parma", "Como 1907": "Como",
    "Venezia FC": "Venezia", "US Salernitana 1919": "Salernitana",
    "Frosinone Calcio": "Frosinone", "US Cremonese": "Cremonese",
    "AC Pisa 1909": "Pisa", "Brescia Calcio": "Brescia",
    "Spezia Calcio": "Spezia", "SSC Bari": "Bari",
    "UC Sampdoria": "Sampdoria", "Modena FC 2018": "Modena",
    # 🏴󠁧󠁢󠁥󠁮󠁧󠁿 PREMIER LEAGUE & CHAMPIONSHIP
    "Manchester United": "Manchester United", "Manchester City": "Manchester City",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "Wolverhampton Wanderers": "Wolverhampton",
    "Brighton & Hove Albion": "Brighton", "West Ham United": "West Ham",
    "Newcastle United": "Newcastle", "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich", "Southampton": "Southampton",
    "Sheffield United": "Sheffield United", "Sheff Utd": "Sheffield United",
    "Sheffield Wednesday": "Sheffield Wednesday", "Sheff Wed": "Sheffield Wednesday",
    "Queens Park Rangers": "QPR", "Leeds United": "Leeds",
    "West Bromwich Albion": "West Brom", "Swansea City": "Swansea",
    "Luton Town": "Luton", "Hull City": "Hull", "Middlesbrough": "Middlesbrough",
    "Coventry City": "Coventry", "Sunderland": "Sunderland",
    "Plymouth Argyle": "Plymouth", "Bristol City": "Bristol City",
    "Watford": "Watford", "Norwich City": "Norwich", "Cardiff City": "Cardiff",
    "Stoke City": "Stoke", "Blackburn Rovers": "Blackburn",
    "Preston North End": "Preston", "Burnley FC": "Burnley",
    "Burnley": "Burnley", "Millwall": "Millwall",
    "Huddersfield Town": "Huddersfield", "Birmingham City": "Birmingham",
    "Rotherham United": "Rotherham", "Derby County": "Derby",
    "Portsmouth": "Portsmouth", "Oxford United": "Oxford Utd",
    "Nottingham Forest": "Nottingham Forest", "Nottm Forest": "Nottingham Forest",
    # 🇳🇱 EREDIVISIE
    "AFC Ajax": "Ajax", "PSV Eindhoven": "PSV", "AZ Alkmaar": "AZ",
    "FC Utrecht": "Utrecht", "FC Twente": "Twente",
    "SC Heerenveen": "Heerenveen", "PEC Zwolle": "PEC Zwolle",
    "Almere City FC": "Almere City", "FC Groningen": "Groningen",
    "Sparta Rotterdam": "Sparta Rotterdam", "RKC Waalwijk": "RKC Waalwijk",
    "NEC Nijmegen": "NEC",
    # 🇵🇹 PRIMEIRA LIGA
    "SL Benfica": "Benfica", "FC Porto": "Porto", "Sporting CP": "Sporting CP",
    "SC Braga": "Braga", "Vitória SC": "Vitoria Guimaraes",
    "Vitoria SC": "Vitoria Guimaraes", "Gil Vicente FC": "Gil Vicente",
    "Boavista FC": "Boavista", "Moreirense FC": "Moreirense",
    "GD Estoril Praia": "Estoril", "GD Chaves": "Chaves",
    "Casa Pia AC": "Casa Pia", "CD Famalicão": "Famalicao",
    "Rio Ave FC": "Rio Ave", "SC Farense": "Farense",
    "CF Arouca": "Arouca", "CD Nacional": "Nacional",
    # 🇹🇷 SÜPER LIG
    "Galatasaray SK": "Galatasaray", "Fenerbahçe SK": "Fenerbahce",
    "Fenerbahce SK": "Fenerbahce", "Beşiktaş JK": "Besiktas",
    "Besiktas JK": "Besiktas", "Kasımpaşa SK": "Kasimpasa",
    "İstanbul Başakşehir FK": "Basaksehir", "Istanbul Basaksehir FK": "Basaksehir",
    "Göztepe SK": "Goztepe", "Yılport Samsunspor": "Samsunspor",
    "Fatih Karagümrük SK": "Karagumruk", "Sivasspor": "Sivasspor",
    "Alanyaspor": "Alanyaspor", "Konyaspor": "Konyaspor",
    "Antalyaspor": "Antalyaspor", "Kayserispor": "Kayserispor",
    "Gaziantep FK": "Gaziantep", "MKE Ankaragücü": "Ankaragucu",
    "Adana Demirspor": "Adana Demirspor",
    # 🇧🇷 SÉRIE A BRÉSIL
    "Athletico Paranaense": "Athletico-PR", "Atlético Paranaense": "Athletico-PR",
    "Atlético-MG": "Atletico Mineiro", "Atlético Mineiro": "Atletico Mineiro",
    "Bragantino": "Red Bull Bragantino", "Grêmio": "Gremio",
    "São Paulo FC": "Sao Paulo", "Sao Paulo": "Sao Paulo",
    "Sport Club Corinthians Paulista": "Corinthians",
    "Sociedade Esportiva Palmeiras": "Palmeiras",
    "Club de Regatas do Flamengo": "Flamengo",
    "Fluminense FC": "Fluminense", "Botafogo FR": "Botafogo",
    "CR Vasco da Gama": "Vasco da Gama", "EC Bahia": "Bahia",
    "Fortaleza EC": "Fortaleza", "Sport Club Internacional": "Internacional",
    "Cruzeiro EC": "Cruzeiro", "EC Juventude": "Juventude",
    "Criciúma EC": "Criciuma", "Santos FC": "Santos",
    "Ceará SC": "Ceara", "Coritiba FC": "Coritiba",
    # 🇸🇪 ALLSVENSKAN
    "AIK Fotboll": "AIK", "Malmö FF": "Malmo FF",
    "Djurgårdens IF": "Djurgarden", "IFK Göteborg": "IFK Goteborg",
    "BK Häcken": "BK Hacken", "IFK Norrköping": "IFK Norrkoping",
    "Mjällby AIF": "Mjallby", "Halmstads BK": "Halmstad",
    "IK Sirius FK": "Sirius", "Kalmar FF": "Kalmar",
    "Degerfors IF": "Degerfors", "GIF Sundsvall": "Sundsvall",
    # 🇳🇴 ELITESERIEN
    "FK Bodø/Glimt": "Bodo/Glimt", "Bodø/Glimt": "Bodo/Glimt",
    "Molde FK": "Molde", "Rosenborg BK": "Rosenborg", "SK Brann": "Brann",
    "Viking FK": "Viking", "Tromsø IL": "Tromso", "IL Tromso": "Tromso",
    "Stabæk Fotball": "Stabek", "Strømsgodset IF": "Stromsgodset",
    "FK Haugesund": "Haugesund", "Odd BK": "Odd",
    "Sandefjord Fotball": "Sandefjord", "Lillestrøm SK": "Lillestrom",
    # 🇧🇪 JUPILER PRO LEAGUE
    "RSC Anderlecht": "Anderlecht", "Club Brugge KV": "Club Brugge",
    "KAA Gent": "Gent", "Standard Liège": "Standard Liege",
    "Standard de Liège": "Standard Liege", "KRC Genk": "Genk",
    "Royal Antwerp FC": "Antwerp",
    "Royale Union Saint-Gilloise": "Union Saint Gilloise",
    "R. Charleroi SC": "Charleroi", "Cercle Brugge KSV": "Cercle Brugge",
    "Sint-Truidense VV": "Sint-Truiden", "KV Mechelen": "Mechelen",
    "KV Kortrijk": "Kortrijk", "K. Beerschot VA": "Beerschot",
    "KAS Eupen": "Eupen", "OH Leuven": "OHL Leuven",
    "Westerlo": "Westerlo", "RWDM Brussels FC": "RWDM",
    # 🇺🇸 MLS
    "Inter Miami CF": "Inter Miami", "LA Galaxy": "Los Angeles Galaxy",
    "LAFC": "Los Angeles FC", "New York Red Bulls": "NY Red Bulls",
    "New York City FC": "New York City FC", "Seattle Sounders FC": "Seattle Sounders",
    "Atlanta United FC": "Atlanta United", "D.C. United": "DC United",
    "Colorado Rapids": "Colorado Rapids", "Houston Dynamo FC": "Houston Dynamo",
    "Minnesota United FC": "Minnesota United", "CF Montréal": "CF Montreal",
    "Chicago Fire FC": "Chicago Fire", "Vancouver Whitecaps FC": "Vancouver Whitecaps",
    "St. Louis City SC": "St. Louis City", "Austin FC": "Austin FC",
}

# ─────────────────────────────────────────────────────────────
# 🗄️  BASE DE DONNÉES
# ─────────────────────────────────────────────────────────────
async def init_db(conn):
    await conn.executescript("""
        CREATE TABLE IF NOT EXISTS bt_fixtures (
            id          INTEGER PRIMARY KEY,
            ligue_id    INTEGER,
            saison      INTEGER,
            date_utc    TEXT,
            home_id     INTEGER,
            away_id     INTEGER,
            home_name   TEXT,
            away_name   TEXT,
            gh          INTEGER,
            ga          INTEGER
        );
        CREATE TABLE IF NOT EXISTS bt_xg (
            fixture_id  INTEGER,
            team_id     INTEGER,
            xg_p        REAL,
            xg_c        REAL,
            PRIMARY KEY (fixture_id, team_id)
        );
        CREATE TABLE IF NOT EXISTS bt_odds_h24 (
            fixture_id  INTEGER,
            market      TEXT,
            outcome     TEXT,
            h_val       REAL,
            cote        REAL,
            PRIMARY KEY (fixture_id, market, outcome, h_val)
        );
        CREATE TABLE IF NOT EXISTS bt_odds_cloture (
            fixture_id  INTEGER,
            market      TEXT,
            outcome     TEXT,
            h_val       REAL,
            cote        REAL,
            PRIMARY KEY (fixture_id, market, outcome, h_val)
        );
        CREATE TABLE IF NOT EXISTS bt_signaux (
            fixture_id  INTEGER,
            ligue_id    INTEGER,
            saison      INTEGER,
            market      TEXT,
            outcome     TEXT,
            h_val       REAL,
            cote_h24    REAL,
            cote_cloture REAL,
            ev_modele   REAL,
            kelly       REAL,
            mise        REAL,
            gh          INTEGER,
            ga          INTEGER,
            resultat    REAL,
            clv         REAL,
            PRIMARY KEY (fixture_id, market, outcome, h_val)
        );
    """)
    await conn.commit()


# ─────────────────────────────────────────────────────────────
# 🌐  HTTP HELPERS
# ─────────────────────────────────────────────────────────────
semaphore = asyncio.Semaphore(3)

async def fetch(session, url, headers=None, params=None):
    async with semaphore:
        try:
            async with session.get(url, headers=headers, params=params, timeout=20) as r:
                if r.status == 200:
                    return await r.json()
                if r.status == 429:
                    print("⏳ Rate limit — pause 30s")
                    await asyncio.sleep(30)
                return None
        except Exception as e:
            print(f"  ⚠️ fetch error: {e}")
            return None


def saison_pour_ligue(ligue_id, annee):
    """Ligues estivales : saison = année civile. Ligues hivernales : saison = année de début."""
    estivales = {71, 113, 253, 103}
    return annee if ligue_id in estivales else annee


# ─────────────────────────────────────────────────────────────
# 📐  MODÈLE MATHÉMATIQUE (copie des fonctions du bot principal)
# ─────────────────────────────────────────────────────────────
def generer_matrice(l_dom, l_ext, rho=-0.12):
    p_d = [poisson.pmf(i, l_dom) for i in range(10)]
    p_e = [poisson.pmf(i, l_ext) for i in range(10)]
    m = np.outer(p_d, p_e).astype(float)
    m[0, 0] *= max(0, 1 - l_dom * l_ext * rho)
    m[1, 0] *= max(0, 1 + l_ext * rho)
    m[0, 1] *= max(0, 1 + l_dom * rho)
    m[1, 1] *= max(0, 1 - rho)
    return m / np.sum(m)


_EPS = 1e-6   # tolérance pour comparaisons float (quarts de handicap)

def _payout_ah(res_net, cote):
    """5 issues Asian Handicap : full win / half win / push / half loss / full loss."""
    if res_net > 0.25 + _EPS:           return cote            # full win
    if abs(res_net - 0.25) < _EPS:      return 1.0 + (cote - 1.0) / 2  # half win
    if abs(res_net) < _EPS:             return 1.0             # push
    if abs(res_net + 0.25) < _EPS:      return 0.5             # half loss
    return 0.0                                                  # full loss

def _x_kelly_ah(res_net, cote):
    """Gain net pour Kelly mean-variance."""
    if res_net > 0.25 + _EPS:           return cote - 1.0
    if abs(res_net - 0.25) < _EPS:      return (cote - 1.0) / 2.0
    if abs(res_net) < _EPS:             return 0.0
    if abs(res_net + 0.25) < _EPS:      return -0.5
    return -1.0

def _payout_total(res_net, cote):
    """5 issues Asian Total : même logique que AH."""
    return _payout_ah(res_net, cote)

def _x_kelly_total(res_net, cote):
    return _x_kelly_ah(res_net, cote)


def ev_ah(mat, h, is_home, cote):
    """EV Asian Handicap — signe et 5 issues identiques au bot principal."""
    esp = 0.0
    for i in range(10):
        for j in range(10):
            diff = (i - j) if is_home else (j - i)
            res_net = diff + h          # ← signe correct (même convention que le bot)
            esp += mat[i, j] * _payout_ah(res_net, cote)
    return esp - 1.0


def ev_total(mat, h, is_over, cote):
    """EV Total Asiatique — 5 issues."""
    esp = 0.0
    for i in range(10):
        for j in range(10):
            tot = i + j
            res_net = (tot - h) if is_over else (h - tot)
            esp += mat[i, j] * _payout_total(res_net, cote)
    return esp - 1.0


def kelly_ah(mat, h, is_home, cote):
    """Kelly mean-variance AH — 5 issues, signe correct."""
    e1, e2 = 0.0, 0.0
    for i in range(10):
        for j in range(10):
            diff = (i - j) if is_home else (j - i)
            res_net = diff + h
            x = _x_kelly_ah(res_net, cote)
            e1 += mat[i, j] * x
            e2 += mat[i, j] * x * x
    return (e1 / e2) if e2 > 1e-9 else 0.0


def kelly_total(mat, h, is_over, cote):
    """Kelly mean-variance Total — 5 issues."""
    e1, e2 = 0.0, 0.0
    for i in range(10):
        for j in range(10):
            tot = i + j
            res_net = (tot - h) if is_over else (h - tot)
            x = _x_kelly_total(res_net, cote)
            e1 += mat[i, j] * x
            e2 += mat[i, j] * x * x
    return (e1 / e2) if e2 > 1e-9 else 0.0


def resultat_ah(gh, ga, h, is_home):
    """Résultat réel AH — même convention de signe que le bot."""
    diff = (gh - ga) if is_home else (ga - gh)
    res_net = diff + h
    if res_net > 0.25 + _EPS:           return 1.0      # full win
    if abs(res_net - 0.25) < _EPS:      return 0.5      # half win
    if abs(res_net) < _EPS:             return 0.0      # push
    if abs(res_net + 0.25) < _EPS:      return -0.5     # half loss
    return -1.0                                          # full loss


def resultat_total(gh, ga, h, is_over):
    """Résultat réel Total Asiatique."""
    tot = gh + ga
    res_net = (tot - h) if is_over else (h - tot)
    if res_net > 0.25 + _EPS:           return 1.0
    if abs(res_net - 0.25) < _EPS:      return 0.5
    if abs(res_net) < _EPS:             return 0.0
    if abs(res_net + 0.25) < _EPS:      return -0.5
    return -1.0


# ─────────────────────────────────────────────────────────────
# 📥  PHASE 1 — COLLECTE DES DONNÉES
# ─────────────────────────────────────────────────────────────
async def collecter_fixtures(conn, session, ligue, saison):
    """Télécharge tous les matchs terminés d'une ligue/saison."""
    url = f"{URL_FOOTBALL}/fixtures?league={ligue['id']}&season={saison}&status=FT"
    data = await fetch(session, url, HEADERS_FB)
    if not data or not data.get('response'):
        return 0

    rows = []
    for f in data['response']:
        rows.append((
            f['fixture']['id'], ligue['id'], saison,
            f['fixture']['date'],
            f['teams']['home']['id'], f['teams']['away']['id'],
            f['teams']['home']['name'], f['teams']['away']['name'],
            f['goals']['home'], f['goals']['away']
        ))

    await conn.executemany(
        "INSERT OR IGNORE INTO bt_fixtures VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    await conn.commit()
    print(f"  ✅ {ligue['nom']} {saison} : {len(rows)} matchs")
    return len(rows)


async def collecter_xg(conn, session, fixture_id, home_id, away_id):
    """Télécharge les xG réels d'un match (ou fait fallback sur les buts)."""
    # Vérifier le cache d'abord
    async with conn.execute(
        "SELECT 1 FROM bt_xg WHERE fixture_id=? AND team_id=?", (fixture_id, home_id)
    ) as cur:
        if await cur.fetchone():
            return

    url = f"{URL_FOOTBALL}/fixtures/statistics?fixture={fixture_id}"
    data = await fetch(session, url, HEADERS_FB)

    xg = {home_id: None, away_id: None}
    if data and data.get('response'):
        for team_stat in data['response']:
            t_id = team_stat['team']['id']
            raw = next(
                (s['value'] for s in team_stat['statistics'] if s['type'] == 'expected_goals'),
                None
            )
            try:
                xg[t_id] = float(raw) if raw not in (None, 'null', '') else None
            except (TypeError, ValueError):
                xg[t_id] = None

    # Fallback sur buts si xG indisponible
    async with conn.execute(
        "SELECT home_id, away_id, gh, ga FROM bt_fixtures WHERE id=?", (fixture_id,)
    ) as cur:
        row = await cur.fetchone()

    if row:
        h_id, a_id, gh, ga = row
        if xg.get(h_id) is None: xg[h_id] = float(gh or 0)
        if xg.get(a_id) is None: xg[a_id] = float(ga or 0)
        # xg_c = xG encaissé = xG de l'adversaire
        rows = [
            (fixture_id, h_id, xg.get(h_id, 0), xg.get(a_id, 0)),
            (fixture_id, a_id, xg.get(a_id, 0), xg.get(h_id, 0)),
        ]
        await conn.executemany("INSERT OR IGNORE INTO bt_xg VALUES (?,?,?,?)", rows)
        await conn.commit()


async def collecter_odds_historiques(conn, session, ligue, date_utc, table, fixture_ids_date):
    """
    Télécharge les cotes Pinnacle à un instant donné pour une ligue entière.
    date_utc : datetime UTC (ex: fixture_date - 24h pour H-24, ou - 5min pour closing)
    Un seul appel API couvre tous les matchs de la ligue à cette date.
    """
    date_str = date_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    url = (f"https://api.the-odds-api.com/v4/historical/sports/{ligue['key']}/odds"
           f"?apiKey={API_ODDS_KEY}&regions=eu&markets=spreads,totals"
           f"&oddsFormat=decimal&bookmakers=pinnacle&date={date_str}")

    raw = await fetch(session, url)
    if not raw:
        return
    # L'endpoint /v4/historical/ enveloppe les résultats dans {"timestamp":..., "data":[...]}
    data = raw.get('data', raw) if isinstance(raw, dict) else raw
    if not isinstance(data, list) or not data:
        return

    for event in data:
        # Trouver le fixture correspondant par fuzzy-match sur les noms d'équipes
        match = trouver_fixture(event['home_team'], event['away_team'],
                                event['commence_time'], fixture_ids_date)
        if not match:
            continue

        fixture_id = match
        pinnacle = next((b for b in event.get('bookmakers', []) if b['key'] == 'pinnacle'), None)
        if not pinnacle:
            continue

        rows = []
        for market in pinnacle['markets']:
            for out in market['outcomes']:
                rows.append((
                    fixture_id, market['key'], out['name'],
                    float(out.get('point', 0)), float(out['price'])
                ))

        if rows:
            await conn.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?,?,?,?,?)", rows)
    await conn.commit()


def trouver_fixture(home_odds, away_odds, commence_time, fixture_map):
    """
    Trouve l'ID du fixture API-Football correspondant à un event Odds API.
    fixture_map : {(home_name, away_name, date_str): fixture_id}
    """
    best_id, best_score = None, 0
    for (h_name, a_name), fid in fixture_map.items():
        h_mapped = NAME_MAPPING.get(h_name, h_name)
        a_mapped = NAME_MAPPING.get(a_name, a_name)
        score_h = process.extractOne(home_odds, [h_mapped])[1]
        score_a = process.extractOne(away_odds, [a_mapped])[1]
        score = (score_h + score_a) / 2
        if score > best_score and score > 75:
            best_score = score
            best_id = fid
    return best_id


async def phase_collecte(conn, session):
    print("\n" + "="*60)
    print("📥  PHASE 1 — COLLECTE DES DONNÉES HISTORIQUES")
    print("="*60)

    for ligue in CHAMPIONNATS:
        for saison in SAISONS_BACKTEST:
            print(f"\n🔄 {ligue['nom']} — Saison {saison}")

            # 1. Fixtures
            n = await collecter_fixtures(conn, session, ligue, saison)
            if n == 0:
                # Essayer saison - 1 pour ligues estivales indexées différemment
                await collecter_fixtures(conn, session, ligue, saison - 1)

            # 2. xG par fixture
            async with conn.execute(
                "SELECT id, home_id, away_id, date_utc FROM bt_fixtures WHERE ligue_id=? AND saison=?",
                (ligue['id'], saison)
            ) as cur:
                fixtures = await cur.fetchall()

            print(f"  📊 Collecte xG pour {len(fixtures)} matchs...")
            # Grouper les fixtures par date pour les appels odds (1 appel/date/ligue)
            par_date = defaultdict(dict)
            for fid, hid, aid, date_str in fixtures:
                await collecter_xg(conn, session, fid, hid, aid)
                try:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    date_key = dt.strftime('%Y-%m-%d')
                    par_date[date_key][(fid, hid, aid)] = dt
                except Exception:
                    pass

            # 3. Odds historiques H-24 et clôture H-5min
            print(f"  📈 Collecte cotes historiques ({len(par_date)} journées)...")
            fixture_name_map = {}
            for fid, hid, aid, date_str in fixtures:
                async with conn.execute(
                    "SELECT home_name, away_name FROM bt_fixtures WHERE id=?", (fid,)
                ) as cur:
                    row = await cur.fetchone()
                if row:
                    fixture_name_map[(row[0], row[1])] = fid

            for date_key, fixtures_du_jour in par_date.items():
                # Prendre la première date du jour
                sample_dt = next(iter(fixtures_du_jour.values()))
                dt_h24    = sample_dt - timedelta(hours=24)
                dt_close  = sample_dt - timedelta(minutes=5)

                await collecter_odds_historiques(
                    conn, session, ligue, dt_h24, "bt_odds_h24", fixture_name_map
                )
                await collecter_odds_historiques(
                    conn, session, ligue, dt_close, "bt_odds_cloture", fixture_name_map
                )
                await asyncio.sleep(0.5)  # Respecter les rate limits

    print("\n✅ Phase 1 terminée.")


# ─────────────────────────────────────────────────────────────
# 🔬  PHASE 2 — SIMULATION DU MODÈLE
# ─────────────────────────────────────────────────────────────
async def calculer_ligue_avg(conn, ligue_id, saison, avant_date):
    """Moyenne de buts par équipe par match dans la ligue/saison AVANT avant_date."""
    async with conn.execute("""
        SELECT AVG((CAST(gh AS REAL) + CAST(ga AS REAL)) / 2.0)
        FROM bt_fixtures
        WHERE ligue_id=? AND saison=? AND date_utc < ?
          AND gh IS NOT NULL AND ga IS NOT NULL
    """, (ligue_id, saison, avant_date)) as cur:
        row = await cur.fetchone()
    return max(0.8, row[0]) if row and row[0] else 1.3


async def reconstruire_xg_equipe(conn, team_id, ligue_id, avant_date, saison, venue='all', ligue_avg=1.3):
    """
    Calcule le xG moyen de l'équipe en utilisant UNIQUEMENT les matchs
    joués AVANT avant_date. Réplique la logique du bot principal :
    - split home/away (venue='home'|'away'|'all')
    - decay exponentiel (demi-vie 46j)
    - shrinkage bayésien
    - fallback saison précédente si < 10 matchs
    Retourne (xg_off, xg_def, n_matchs).
    """
    if venue == 'home':
        venue_filter = "AND f.home_id = ?"
        params = (team_id, ligue_id, saison, saison - 1, avant_date, team_id)
    elif venue == 'away':
        venue_filter = "AND f.away_id = ?"
        params = (team_id, ligue_id, saison, saison - 1, avant_date, team_id)
    else:
        venue_filter = ""
        params = (team_id, ligue_id, saison, saison - 1, avant_date)

    async with conn.execute(f"""
        SELECT f.id, f.date_utc, x.xg_p, x.xg_c, f.home_id, f.away_id, f.saison
        FROM bt_fixtures f
        JOIN bt_xg x ON x.fixture_id = f.id AND x.team_id = ?
        WHERE f.ligue_id = ?
          AND f.saison IN (?, ?)
          AND f.date_utc < ?
          AND f.gh IS NOT NULL
          {venue_filter}
        ORDER BY f.date_utc DESC
        LIMIT 15
    """, params) as cur:
        rows = await cur.fetchall()

    # Fallback sur toutes les venues si < 5 matchs dans le venue demandé
    if len(rows) < 5 and venue != 'all':
        return await reconstruire_xg_equipe(conn, team_id, ligue_id, avant_date, saison, venue='all')

    if len(rows) < 5:
        return 1.3, 1.1, len(rows)  # Promu / données insuffisantes

    now = datetime.fromisoformat(avant_date.replace('Z', '+00:00'))
    tp = tc = tw = 0.0
    for fid, date_str, xg_p, xg_c, home_id, away_id, saison_m in rows:
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            jours = max(0, (now - dt).days)
        except Exception:
            jours = 30
        w = np.exp(-0.015 * jours)
        if saison_m < saison:
            w *= 0.80
        tp += xg_p * w
        tc += xg_c * w
        tw += w

    xg_off_brut = tp / tw
    xg_def_brut = tc / tw

    n = len(rows)
    w_eq = n / (n + N_PRIOR)
    xg_off = w_eq * xg_off_brut + (1 - w_eq) * ligue_avg
    xg_def = w_eq * xg_def_brut + (1 - w_eq) * ligue_avg
    return xg_off, xg_def, n


async def simuler_paris(conn):
    print("\n" + "="*60)
    print("🔬  PHASE 2 — SIMULATION DU MODÈLE")
    print("="*60)

    # Diagnostics rapides pour détecter les problèmes de collecte
    async with conn.execute("SELECT COUNT(*) FROM bt_fixtures WHERE gh IS NOT NULL") as cur:
        n_fix = (await cur.fetchone())[0]
    async with conn.execute("SELECT COUNT(*) FROM bt_odds_h24") as cur:
        n_odds = (await cur.fetchone())[0]
    async with conn.execute("SELECT COUNT(*) FROM bt_xg") as cur:
        n_xg = (await cur.fetchone())[0]
    print(f"  📋 Fixtures avec résultat : {n_fix}")
    print(f"  📊 Cotes H-24 en base     : {n_odds}")
    print(f"  🎯 Entrées xG en base     : {n_xg}")

    if n_odds == 0:
        print("\n  ⚠️  AUCUNE cote H-24 trouvée — la collecte d'odds a échoué.")
        print("  💡 Vérifiez API_ODDS_KEY et que votre plan inclut l'endpoint /v4/historical/")
        print("\n✅ Phase 2 terminée (0 signaux).")
        return

    for ligue in CHAMPIONNATS:
        async with conn.execute(
            "SELECT id, saison, date_utc, home_id, away_id, home_name, away_name, gh, ga "
            "FROM bt_fixtures WHERE ligue_id=? ORDER BY date_utc",
            (ligue['id'],)
        ) as cur:
            fixtures = await cur.fetchall()

        signaux = 0
        for fid, saison, date_utc, h_id, a_id, h_name, a_name, gh, ga in fixtures:
            if gh is None or ga is None:
                continue

            # Récupérer les cotes H-24 disponibles
            async with conn.execute(
                "SELECT market, outcome, h_val, cote FROM bt_odds_h24 WHERE fixture_id=?",
                (fid,)
            ) as cur:
                odds_h24 = await cur.fetchall()

            if not odds_h24:
                continue

            # Moyenne de buts de la ligue calculée dynamiquement (remplace 1.3 hardcodé)
            avg_ligue = await calculer_ligue_avg(conn, ligue['id'], saison, date_utc)

            # Reconstituer xG AVANT ce match — split home/away + global comme le bot principal
            xg_off_d_sp, xg_def_d_sp, n_d = await reconstruire_xg_equipe(
                conn, h_id, ligue['id'], date_utc, saison, venue='home', ligue_avg=avg_ligue
            )
            xg_off_e_sp, xg_def_e_sp, n_e = await reconstruire_xg_equipe(
                conn, a_id, ligue['id'], date_utc, saison, venue='away', ligue_avg=avg_ligue
            )

            # Filtre : ignorer si l'une des équipes manque d'historique suffisant
            if n_d < 8 or n_e < 8:
                continue

            # Venue blending adaptatif : réplique w_venue() du bot
            # Moins de matchs venue-spécifiques → on se fie davantage aux stats globales
            xg_off_d_gl, xg_def_d_gl, _ = await reconstruire_xg_equipe(
                conn, h_id, ligue['id'], date_utc, saison, venue='all', ligue_avg=avg_ligue
            )
            xg_off_e_gl, xg_def_e_gl, _ = await reconstruire_xg_equipe(
                conn, a_id, ligue['id'], date_utc, saison, venue='all', ligue_avg=avg_ligue
            )

            def w_venue(n_spec, max_w=0.80):
                return min(max_w, (n_spec / 10.0) * max_w)

            wd = w_venue(n_d)
            we = w_venue(n_e)
            xg_off_d = xg_off_d_sp * wd + xg_off_d_gl * (1.0 - wd)
            xg_def_d = xg_def_d_sp * wd + xg_def_d_gl * (1.0 - wd)
            xg_off_e = xg_off_e_sp * we + xg_off_e_gl * (1.0 - we)
            xg_def_e = xg_def_e_sp * we + xg_def_e_gl * (1.0 - we)

            # Paramètres Poisson Dixon-Coles avec avantage domicile implicite via split venue
            L_A = max(0.4, (xg_off_d + xg_def_e) / 2)
            L_B = max(0.4, (xg_off_e + xg_def_d) / 2)

            rho = RHO_PAR_LIGUE.get(ligue['id'], RHO_DEFAULT)
            mat = generer_matrice(L_A, L_B, rho)

            # Parcourir les marchés — collecter tous les signaux valides pour ce fixture
            home_name_odds = NAME_MAPPING.get(h_name, h_name)
            candidats = []  # (ev_final, market, outcome, h_val, cote_h24, k, mise, is_home)

            ev_min_l = ligue.get('ev_min', 0.05)
            ev_max_l = ligue.get('ev_max', 0.15)

            # Index (market, h_val) → cotes des deux côtés pour calcul no-vig
            # Le partenaire d'un outcome (spreads, h) est l'outcome (spreads, -h)
            partner_cote: dict = {}
            for mk, out, hv, c in odds_h24:
                if mk == 'spreads':
                    partner_cote[(mk, hv)] = c

            for market, outcome, h_val, cote_h24 in odds_h24:
                # Handicap Asiatique uniquement — les Totaux n'ont pas d'edge démontré
                if market != 'spreads':
                    continue

                if cote_h24 < MIN_COTE:
                    continue

                is_home = (outcome == home_name_odds) or (
                    process.extractOne(outcome, [home_name_odds])[1] > 85
                )

                ev_modele = ev_ah(mat, h_val, is_home, cote_h24)
                k = kelly_ah(mat, h_val, is_home, cote_h24)

                # Calcul EV Pinnacle no-vig (réplique du blend du bot)
                # La cote partenaire est stockée sous la clé (-h_val)
                cote_partner = partner_cote.get((market, -h_val))
                if cote_partner and cote_partner > 1.0:
                    ovr = (1.0 / cote_h24) + (1.0 / cote_partner)
                    cote_novig = cote_h24 / ovr
                    ev_pinnacle = ev_ah(mat, h_val, is_home, cote_novig)
                else:
                    ev_pinnacle = ev_modele  # fallback si partenaire absent

                ev_final = ev_modele * POIDS_DYN_H24 + ev_pinnacle * (1.0 - POIDS_DYN_H24)

                if not (ev_min_l <= ev_final <= ev_max_l):
                    continue

                mise = min(round(k * 100 * KELLY_FRAC, 2), 5.0)
                if mise < 0.1:
                    continue

                candidats.append((ev_final, market, outcome, h_val, cote_h24, k, mise, is_home))

            if not candidats:
                continue

            # 🔒 FILTRE 1 PARI/MATCH : garder uniquement le signal avec le meilleur EV final
            candidats.sort(key=lambda x: x[0], reverse=True)
            ev_final, market, outcome, h_val, cote_h24, k, mise, flag = candidats[0]

            # Cote de clôture
            async with conn.execute(
                "SELECT cote FROM bt_odds_cloture WHERE fixture_id=? AND market=? AND outcome=? AND h_val=?",
                (fid, market, outcome, h_val)
            ) as cur:
                row = await cur.fetchone()
            cote_cloture = row[0] if row else None

            # Résultat réel
            if market == 'spreads':
                res = resultat_ah(gh, ga, h_val, flag)
            else:
                res = resultat_total(gh, ga, h_val, flag)

            clv = round((cote_h24 / cote_cloture) - 1, 4) if cote_cloture else None

            await conn.execute(
                "INSERT OR REPLACE INTO bt_signaux VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (fid, ligue['id'], saison, market, outcome, h_val,
                 cote_h24, cote_cloture, round(ev_final, 4), round(k, 4),
                 mise, gh, ga, res, clv)
            )
            signaux += 1

        await conn.commit()
        print(f"  {ligue['nom']} : {signaux} signaux générés")

    print("\n✅ Phase 2 terminée.")


# ─────────────────────────────────────────────────────────────
# 📊  PHASE 3 — RAPPORT D'ANALYSE
# ─────────────────────────────────────────────────────────────
async def generer_rapport(conn):
    print("\n" + "="*60)
    print("📊  PHASE 3 — RAPPORT D'ANALYSE")
    print("="*60)

    nom_par_id = {c['id']: c['nom'] for c in CHAMPIONNATS}

    async with conn.execute(
        "SELECT * FROM bt_signaux WHERE resultat IS NOT NULL ORDER BY ligue_id, saison"
    ) as cur:
        rows = await cur.fetchall()

    # Ajouter le nom de la ligue en fin de tuple (compatible avec toutes versions SQLite)
    signaux = [(*r, nom_par_id.get(r[1], str(r[1]))) for r in rows]

    if not signaux:
        print("⚠️  Aucun signal trouvé. Lancez d'abord --collect et --simulate.")
        return

    # ── Rapport global ──────────────────────────────────────
    total = len(signaux)
    clv_vals  = [s[14] for s in signaux if s[14] is not None]
    res_vals  = [(s[13], s[10]) for s in signaux if s[13] is not None]  # (resultat, mise)
    pnl_total = sum(r * m for r, m in res_vals)
    mises_tot = sum(m for _, m in res_vals)
    clv_moy   = np.mean(clv_vals) if clv_vals else 0
    win_rate  = sum(1 for r, _ in res_vals if r > 0) / len(res_vals) if res_vals else 0

    print(f"\n{'─'*50}")
    print(f"  RÉSULTATS GLOBAUX ({total} signaux)")
    print(f"{'─'*50}")
    print(f"  CLV moyen           : {clv_moy:+.2%}")
    print(f"  P&L total           : {pnl_total:+.1f} u")
    print(f"  Mises totales       : {mises_tot:.1f} u")
    print(f"  ROI                 : {pnl_total/mises_tot:+.2%}" if mises_tot else "  ROI : N/A")
    print(f"  Win rate            : {win_rate:.1%}")
    print(f"  Signaux avec CLV    : {len(clv_vals)}/{total}")

    # Drawdown maximum
    bankroll = 100.0
    peak = bankroll
    max_dd = 0.0
    for r, m in res_vals:
        bankroll += r * m
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak
        if dd > max_dd:
            max_dd = dd
    print(f"  Drawdown max        : {max_dd:.1%}")

    # ── Rapport par saison ──────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  PAR SAISON")
    print(f"{'─'*50}")
    print(f"  {'Saison':<8} {'N':>5} {'CLV':>8} {'ROI':>8} {'P&L':>8} {'Drawdown':>10}")
    print(f"  {'─'*8} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

    par_saison = defaultdict(list)
    for s in signaux:
        par_saison[s[2]].append(s)  # s[2] = saison

    for saison, rows_s in sorted(par_saison.items()):
        clv_s  = [r[14] for r in rows_s if r[14] is not None]
        res_s  = [(r[13], r[10]) for r in rows_s if r[13] is not None]
        pnl_s  = sum(r * m for r, m in res_s)
        mis_s  = sum(m for _, m in res_s)
        roi_s  = pnl_s / mis_s if mis_s else 0
        clv_s_moy = np.mean(clv_s) if clv_s else 0
        # Drawdown par saison
        bk, pk, dd_s = 100.0, 100.0, 0.0
        for r, m in res_s:
            bk += r * m
            if bk > pk: pk = bk
            dd_s = max(dd_s, (pk - bk) / pk)
        marker = "✅" if roi_s > 0 else "❌"
        print(f"  {marker} {saison:<6} {len(rows_s):>5} {clv_s_moy:>+7.1%} {roi_s:>+7.1%} {pnl_s:>+7.1f}u {dd_s:>9.1%}")

    # ── Rapport par ligue ───────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  PAR LIGUE")
    print(f"{'─'*50}")
    print(f"  {'Ligue':<20} {'N':>5} {'CLV':>8} {'ROI':>8} {'P&L':>8}")
    print(f"  {'─'*20} {'─'*5} {'─'*8} {'─'*8} {'─'*8}")

    par_ligue = defaultdict(list)
    for s in signaux:
        par_ligue[s[-1]].append(s)  # s[-1] = ligue_nom

    for nom, rows in sorted(par_ligue.items()):
        n = len(rows)
        clv_l = [r[14] for r in rows if r[14] is not None]
        res_l = [(r[13], r[10]) for r in rows if r[13] is not None]
        pnl_l = sum(r * m for r, m in res_l)
        mis_l = sum(m for _, m in res_l)
        clv_l_moy = np.mean(clv_l) if clv_l else 0
        roi_l = pnl_l / mis_l if mis_l else 0
        marker = "✅" if roi_l > 0 else "❌"
        print(f"  {marker} {nom:<18} {n:>5} {clv_l_moy:>+7.1%} {roi_l:>+7.1%} {pnl_l:>+7.1f}u")

    # ── Rapport par marché ──────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  PAR MARCHÉ")
    print(f"{'─'*50}")
    for market in ['spreads', 'totals']:
        rows_m = [s for s in signaux if s[3] == market and s[13] is not None]
        if not rows_m:
            continue
        pnl_m = sum(r[13] * r[10] for r in rows_m)
        mis_m = sum(r[10] for r in rows_m)
        roi_m = pnl_m / mis_m if mis_m else 0
        label = "Handicap Asiatique" if market == 'spreads' else "Totaux"
        print(f"  {label:<22} {len(rows_m):>5} signaux → ROI {roi_m:+.2%} | P&L {pnl_m:+.1f}u")

    # ── Courbe de calibration ───────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  CALIBRATION DU MODÈLE (P_modèle vs fréquence réelle)")
    print(f"{'─'*50}")
    # Discrétiser les EV en tranches
    tranches = [(0.05, 0.07), (0.07, 0.09), (0.09, 0.12), (0.12, 0.20)]
    for lo, hi in tranches:
        rows_t = [s for s in signaux if lo <= s[8] < hi and s[13] is not None]
        if not rows_t:
            continue
        win = sum(1 for s in rows_t if s[13] > 0)
        push = sum(1 for s in rows_t if s[13] == 0)
        n_t = len(rows_t)
        wr = win / (n_t - push) if (n_t - push) > 0 else 0
        ev_moy = np.mean([s[8] for s in rows_t])
        print(f"  EV [{lo:.0%}-{hi:.0%}]  n={n_t:>4}  Win={wr:.1%}  EV_moy={ev_moy:+.2%}")

    # ── Export CSV ──────────────────────────────────────────
    csv_path = "backtest_results.csv"
    async with conn.execute(
        "SELECT * FROM bt_signaux WHERE resultat IS NOT NULL ORDER BY ligue_id, saison"
    ) as cur:
        rows_csv = await cur.fetchall()

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['fixture_id', 'ligue_id', 'saison', 'market', 'outcome',
                    'h_val', 'cote_h24', 'cote_cloture', 'ev_modele', 'kelly',
                    'mise', 'gh', 'ga', 'resultat', 'clv'])
        w.writerows(rows_csv)

    print(f"\n📄 Résultats détaillés exportés dans {csv_path}")
    print("\n✅ Phase 3 terminée.")


# ─────────────────────────────────────────────────────────────
# 🚀  POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Back-test Dixon-Coles Football")
    parser.add_argument('--collect',  action='store_true', help='Phase 1 : collecte données')
    parser.add_argument('--simulate', action='store_true', help='Phase 2 : simulation')
    parser.add_argument('--report',   action='store_true', help='Phase 3 : rapport')
    args = parser.parse_args()

    all_phases = not (args.collect or args.simulate or args.report)

    async with aiosqlite.connect(DB_PATH) as conn:
        await init_db(conn)

        async with aiohttp.ClientSession() as session:
            if args.collect or all_phases:
                await phase_collecte(conn, session)
            if args.simulate or all_phases:
                await simuler_paris(conn)
            if args.report or all_phases:
                await generer_rapport(conn)


if __name__ == "__main__":
    asyncio.run(main())
