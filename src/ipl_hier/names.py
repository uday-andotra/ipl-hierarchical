"""Franchise name continuity. Defunct sides stay distinct."""

ALIASES = {
    "delhi daredevils": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "kings xi punjab": "Punjab Kings",
    "punjab kings": "Punjab Kings",
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "royal challengers bengaluru": "Royal Challengers Bengaluru",
    "rising pune supergiants": "Rising Pune Supergiant",
    "rising pune supergiant": "Rising Pune Supergiant",
}

# Cities that map cleanly to one living franchise. Playoffs / UAE / others are neutral.
CITY_HOME = {
    "mumbai": "Mumbai Indians",
    "chennai": "Chennai Super Kings",
    "kolkata": "Kolkata Knight Riders",
    "jaipur": "Rajasthan Royals",
    "hyderabad": "Sunrisers Hyderabad",
    "bengaluru": "Royal Challengers Bengaluru",
    "bangalore": "Royal Challengers Bengaluru",
    "delhi": "Delhi Capitals",
    "dharamsala": "Punjab Kings",
    "chandigarh": "Punjab Kings",
    "mohali": "Punjab Kings",
    "ahmedabad": "Gujarat Titans",
    "lucknow": "Lucknow Super Giants",
}


def canon_team(name: str) -> str:
    if not name:
        return ""
    key = " ".join(name.strip().lower().split())
    return ALIASES.get(key, name.strip())
