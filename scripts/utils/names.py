import re, unicodedata

TEAM_MAP = {
  "KAN":"KC","KCC":"KC","KANSAS CITY CHIEFS":"KC","CHIEFS":"KC",
  "GNB":"GB","GBP":"GB","PACKERS":"GB",
  "SFO":"SF","SF 49ERS":"SF","49ERS":"SF",
  "NWE":"NE","PATRIOTS":"NE","NEP":"NE",
  "NOR":"NO","NOS":"NO","SAINTS":"NO",
  "TAM":"TB","TBB":"TB","BUCCANEERS":"TB",
  "LAR":"LAR","RAMS":"LAR",
  "LAC":"LAC","CHARGERS":"LAC",
  "NYJ":"NYJ","JETS":"NYJ","NYG":"NYG","GIANTS":"NYG",
  # … (extend as needed)
}

def normalize_team(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode().upper().strip()
    s = re.sub(r"[^A-Z ]+","", s)
    s = TEAM_MAP.get(s, TEAM_MAP.get(s.replace(" ",""), s))
    return s

def normalize_player(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode()
    s = re.sub(r"\s+"," ", s).strip()
    return s
