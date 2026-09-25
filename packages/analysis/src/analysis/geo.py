"""Country / US-state recognition, so a column of place names can be drawn on a real map.

Plotly's choropleth resolves locations most reliably from ISO-3 codes, but real datasets spell
countries every way imaginable ("Korea, Rep.", "Viet Nam", "US", "CIV"). This module normalizes
any of those spellings to ISO-3 (or a USPS state code) so the chart engine can always hand Plotly
an unambiguous code, and so schema detection can tell "a column of countries" from "a column of
arbitrary categories" by how much of it actually matches.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

# iso3 | iso2 | canonical name | aliases (semicolon separated)
_COUNTRY_TABLE = """
AFG|AF|Afghanistan|
ALB|AL|Albania|
DZA|DZ|Algeria|
AND|AD|Andorra|
AGO|AO|Angola|
ATG|AG|Antigua and Barbuda|Antigua & Barbuda
ARG|AR|Argentina|
ARM|AM|Armenia|
AUS|AU|Australia|
AUT|AT|Austria|
AZE|AZ|Azerbaijan|
BHS|BS|Bahamas|The Bahamas
BHR|BH|Bahrain|
BGD|BD|Bangladesh|
BRB|BB|Barbados|
BLR|BY|Belarus|
BEL|BE|Belgium|
BLZ|BZ|Belize|
BEN|BJ|Benin|
BTN|BT|Bhutan|
BOL|BO|Bolivia|Bolivia (Plurinational State of)
BIH|BA|Bosnia and Herzegovina|Bosnia & Herzegovina;Bosnia
BWA|BW|Botswana|
BRA|BR|Brazil|
BRN|BN|Brunei|Brunei Darussalam
BGR|BG|Bulgaria|
BFA|BF|Burkina Faso|
BDI|BI|Burundi|
CPV|CV|Cabo Verde|Cape Verde
KHM|KH|Cambodia|
CMR|CM|Cameroon|
CAN|CA|Canada|
CAF|CF|Central African Republic|
TCD|TD|Chad|
CHL|CL|Chile|
CHN|CN|China|People's Republic of China;PRC;Mainland China
COL|CO|Colombia|
COM|KM|Comoros|
COG|CG|Congo|Republic of the Congo;Congo (Brazzaville);Congo, Rep.;Congo Republic
COD|CD|Democratic Republic of the Congo|DR Congo;DRC;Congo (Kinshasa);Congo, Dem. Rep.;Congo, Democratic Republic of the;Congo-Kinshasa
CRI|CR|Costa Rica|
CIV|CI|Cote d'Ivoire|Ivory Coast;Côte d'Ivoire
HRV|HR|Croatia|
CUB|CU|Cuba|
CYP|CY|Cyprus|
CZE|CZ|Czechia|Czech Republic
DNK|DK|Denmark|
DJI|DJ|Djibouti|
DMA|DM|Dominica|
DOM|DO|Dominican Republic|
ECU|EC|Ecuador|
EGY|EG|Egypt|Egypt, Arab Rep.;Arab Republic of Egypt
SLV|SV|El Salvador|
GNQ|GQ|Equatorial Guinea|
ERI|ER|Eritrea|
EST|EE|Estonia|
SWZ|SZ|Eswatini|Swaziland
ETH|ET|Ethiopia|
FJI|FJ|Fiji|
FIN|FI|Finland|
FRA|FR|France|
GAB|GA|Gabon|
GMB|GM|Gambia|The Gambia;Gambia, The
GEO|GE|Georgia|
DEU|DE|Germany|
GHA|GH|Ghana|
GRC|GR|Greece|
GRD|GD|Grenada|
GTM|GT|Guatemala|
GIN|GN|Guinea|
GNB|GW|Guinea-Bissau|Guinea Bissau
GUY|GY|Guyana|
HTI|HT|Haiti|
HND|HN|Honduras|
HKG|HK|Hong Kong|Hong Kong SAR;Hong Kong SAR, China;Hong Kong, China
HUN|HU|Hungary|
ISL|IS|Iceland|
IND|IN|India|
IDN|ID|Indonesia|
IRN|IR|Iran|Iran, Islamic Rep.;Islamic Republic of Iran;Iran (Islamic Republic of)
IRQ|IQ|Iraq|
IRL|IE|Ireland|Republic of Ireland
ISR|IL|Israel|
ITA|IT|Italy|
JAM|JM|Jamaica|
JPN|JP|Japan|
JOR|JO|Jordan|
KAZ|KZ|Kazakhstan|
KEN|KE|Kenya|
KIR|KI|Kiribati|
XKX|XK|Kosovo|
KWT|KW|Kuwait|
KGZ|KG|Kyrgyzstan|Kyrgyz Republic
LAO|LA|Laos|Lao PDR;Lao People's Democratic Republic
LVA|LV|Latvia|
LBN|LB|Lebanon|
LSO|LS|Lesotho|
LBR|LR|Liberia|
LBY|LY|Libya|
LIE|LI|Liechtenstein|
LTU|LT|Lithuania|
LUX|LU|Luxembourg|
MAC|MO|Macao|Macau;Macao SAR, China
MDG|MG|Madagascar|
MWI|MW|Malawi|
MYS|MY|Malaysia|
MDV|MV|Maldives|
MLI|ML|Mali|
MLT|MT|Malta|
MHL|MH|Marshall Islands|
MRT|MR|Mauritania|
MUS|MU|Mauritius|
MEX|MX|Mexico|
FSM|FM|Micronesia|Federated States of Micronesia;Micronesia, Fed. Sts.
MDA|MD|Moldova|Republic of Moldova
MCO|MC|Monaco|
MNG|MN|Mongolia|
MNE|ME|Montenegro|
MAR|MA|Morocco|
MOZ|MZ|Mozambique|
MMR|MM|Myanmar|Burma
NAM|NA|Namibia|
NRU|NR|Nauru|
NPL|NP|Nepal|
NLD|NL|Netherlands|The Netherlands;Holland
NZL|NZ|New Zealand|
NIC|NI|Nicaragua|
NER|NE|Niger|
NGA|NG|Nigeria|
PRK|KP|North Korea|Korea, Dem. People's Rep.;Democratic People's Republic of Korea;Korea, North;DPRK;Korea (North)
MKD|MK|North Macedonia|Macedonia;Republic of North Macedonia
NOR|NO|Norway|
OMN|OM|Oman|
PAK|PK|Pakistan|
PLW|PW|Palau|
PSE|PS|Palestine|State of Palestine;Palestinian Territories;West Bank and Gaza;Palestinian Territory
PAN|PA|Panama|
PNG|PG|Papua New Guinea|
PRY|PY|Paraguay|
PER|PE|Peru|
PHL|PH|Philippines|
POL|PL|Poland|
PRT|PT|Portugal|
PRI|PR|Puerto Rico|
QAT|QA|Qatar|
ROU|RO|Romania|
RUS|RU|Russia|Russian Federation
RWA|RW|Rwanda|
KNA|KN|Saint Kitts and Nevis|St. Kitts and Nevis;St Kitts and Nevis
LCA|LC|Saint Lucia|St. Lucia;St Lucia
VCT|VC|Saint Vincent and the Grenadines|St. Vincent and the Grenadines
WSM|WS|Samoa|
SMR|SM|San Marino|
STP|ST|Sao Tome and Principe|São Tomé and Príncipe
SAU|SA|Saudi Arabia|
SEN|SN|Senegal|
SRB|RS|Serbia|
SYC|SC|Seychelles|
SLE|SL|Sierra Leone|
SGP|SG|Singapore|
SVK|SK|Slovakia|Slovak Republic
SVN|SI|Slovenia|
SLB|SB|Solomon Islands|
SOM|SO|Somalia|Somaliland region
ZAF|ZA|South Africa|
KOR|KR|South Korea|Korea, Rep.;Republic of Korea;Korea, South;Korea (South);Korea
SSD|SS|South Sudan|
ESP|ES|Spain|
LKA|LK|Sri Lanka|
SDN|SD|Sudan|
SUR|SR|Suriname|
SWE|SE|Sweden|
CHE|CH|Switzerland|
SYR|SY|Syria|Syrian Arab Republic
TWN|TW|Taiwan|Taiwan Province of China
TJK|TJ|Tajikistan|
TZA|TZ|Tanzania|United Republic of Tanzania
THA|TH|Thailand|
TLS|TL|Timor-Leste|East Timor
TGO|TG|Togo|
TON|TO|Tonga|
TTO|TT|Trinidad and Tobago|Trinidad & Tobago
TUN|TN|Tunisia|
TUR|TR|Turkey|Turkiye;Türkiye
TKM|TM|Turkmenistan|
TUV|TV|Tuvalu|
UGA|UG|Uganda|
UKR|UA|Ukraine|
ARE|AE|United Arab Emirates|UAE
GBR|GB|United Kingdom|UK;Great Britain;Britain;United Kingdom of Great Britain and Northern Ireland;England
USA|US|United States|United States of America;U.S.;U.S.A.;America;US of A
URY|UY|Uruguay|
UZB|UZ|Uzbekistan|
VUT|VU|Vanuatu|
VAT|VA|Vatican City|Holy See;Vatican
VEN|VE|Venezuela|Venezuela, RB;Venezuela (Bolivarian Republic of)
VNM|VN|Vietnam|Viet Nam
YEM|YE|Yemen|Yemen, Rep.
ZMB|ZM|Zambia|
ZWE|ZW|Zimbabwe|
GRL|GL|Greenland|
BMU|BM|Bermuda|
""".strip()

# USPS code | name
_US_STATE_TABLE = """
AL|Alabama
AK|Alaska
AZ|Arizona
AR|Arkansas
CA|California
CO|Colorado
CT|Connecticut
DE|Delaware
DC|District of Columbia|Washington DC;Washington, D.C.
FL|Florida
GA|Georgia
HI|Hawaii
ID|Idaho
IL|Illinois
IN|Indiana
IA|Iowa
KS|Kansas
KY|Kentucky
LA|Louisiana
ME|Maine
MD|Maryland
MA|Massachusetts
MI|Michigan
MN|Minnesota
MS|Mississippi
MO|Missouri
MT|Montana
NE|Nebraska
NV|Nevada
NH|New Hampshire
NJ|New Jersey
NM|New Mexico
NY|New York
NC|North Carolina
ND|North Dakota
OH|Ohio
OK|Oklahoma
OR|Oregon
PA|Pennsylvania
RI|Rhode Island
SC|South Carolina
SD|South Dakota
TN|Tennessee
TX|Texas
UT|Utah
VT|Vermont
VA|Virginia
WA|Washington
WV|West Virginia
WI|Wisconsin
WY|Wyoming
""".strip()


def _norm(text: str) -> str:
    """Case/punctuation/accent-insensitive key: 'Côte d'Ivoire' and 'cote d ivoire' collide."""
    import unicodedata

    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _build_country_index() -> tuple[dict[str, str], dict[str, str], set[str]]:
    names: dict[str, str] = {}
    canonical: dict[str, str] = {}
    iso2_only: set[str] = set()
    for line in _COUNTRY_TABLE.splitlines():
        iso3, iso2, name, aliases = (line.split("|") + [""])[:4]
        canonical[iso3] = name
        names[_norm(name)] = iso3
        for alias in filter(None, aliases.split(";")):
            names[_norm(alias)] = iso3
        names.setdefault(_norm(iso3), iso3)
        iso2_only.add(iso2)
        names.setdefault("iso2:" + iso2.lower(), iso3)
    return names, canonical, iso2_only


_COUNTRY_INDEX, COUNTRY_NAMES, _ISO2_CODES = _build_country_index()
_ISO3_CODES = set(COUNTRY_NAMES)


def _build_state_index() -> tuple[dict[str, str], dict[str, str]]:
    names: dict[str, str] = {}
    canonical: dict[str, str] = {}
    for line in _US_STATE_TABLE.splitlines():
        code, name, aliases = (line.split("|") + [""])[:3]
        canonical[code] = name
        names[_norm(name)] = code
        for alias in filter(None, aliases.split(";")):
            names[_norm(alias)] = code
        names.setdefault("code:" + code.lower(), code)
    return names, canonical


_STATE_INDEX, STATE_NAMES = _build_state_index()


def country_to_iso3(value: object) -> str | None:
    """Any spelling/code of a country -> ISO-3, or None if it isn't one."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    if not text:
        return None
    key = _norm(text)
    if key in _COUNTRY_INDEX:
        return _COUNTRY_INDEX[key]
    if len(text) == 2 and text.isalpha():
        return _COUNTRY_INDEX.get("iso2:" + text.lower())
    return None


def state_to_code(value: object) -> str | None:
    """US state name or USPS code -> 2-letter code, or None."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    if not text:
        return None
    key = _norm(text)
    if key in _STATE_INDEX:
        return _STATE_INDEX[key]
    if len(text) == 2 and text.isalpha():
        return _STATE_INDEX.get("code:" + text.lower())
    return None


def _match_rate(series: pd.Series, resolver, max_unique: int = 400) -> tuple[float, dict[str, str]]:
    """Fraction of *distinct non-null values* that resolve, and the value->code mapping."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return 0.0, {}
    uniques = pd.unique(values)
    if len(uniques) > max_unique:
        uniques = uniques[:max_unique]
    mapping: dict[str, str] = {}
    for value in uniques:
        code = resolver(value)
        if code:
            mapping[value] = code
    return len(mapping) / len(uniques), mapping


def _code_kind(series: pd.Series) -> str:
    """Whether a matched column is written as ISO-3 codes, ISO-2 codes, or names."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "name"
    lengths = values.str.len()
    if (lengths == 3).mean() > 0.9 and values.str.isupper().mean() > 0.9:
        return "iso3"
    if (lengths == 2).mean() > 0.9 and values.str.isupper().mean() > 0.9:
        return "iso2"
    return "name"


_COUNTRY_NAME_HINT = re.compile(r"(^|[^a-z])(country|countries|nation|iso3?|cntry|country_?code|countrycode|location|economy)([^a-z]|$)", re.I)
_STATE_NAME_HINT = re.compile(r"(^|[^a-z])(state|states|us_?state|province)([^a-z]|$)", re.I)
_LAT_HINT = re.compile(r"^(lat|latitude|y_?coord(inate)?|lat_?deg)$", re.I)
_LON_HINT = re.compile(r"^(lon|lng|long|longitude|x_?coord(inate)?|lon_?deg)$", re.I)


def detect_geo(df: pd.DataFrame) -> dict[str, object]:
    """Find columns that can be drawn on a map.

    A column counts as a country/state column when enough of its distinct values genuinely
    resolve against the tables above - a name hint alone ("state" may mean "status") is not
    enough, and value matches alone need a high hit rate so a categorical whose few values
    happen to collide with a country name isn't misread as geography.
    """
    result: dict[str, object] = {"country": None, "us_state": None, "lat_lon": None}
    text_columns = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]

    best_country: tuple[float, str, dict[str, str], float] | None = None
    best_state: tuple[float, str, dict[str, str], float] | None = None
    for column in text_columns:
        series = df[column]
        unique_count = series.nunique(dropna=True)
        if unique_count < 3:
            continue
        hinted_country = bool(_COUNTRY_NAME_HINT.search(str(column)))
        hinted_state = bool(_STATE_NAME_HINT.search(str(column)))

        c_rate, c_map = _match_rate(series, country_to_iso3)
        s_rate, s_map = _match_rate(series, state_to_code)
        c_threshold = 0.5 if hinted_country else 0.85
        s_threshold = 0.6 if hinted_state else 0.9
        # Two-letter codes are ambiguous ("CA" is Canada and California): let whichever table
        # explains more of the column win, and break ties toward the column's name.
        country_ok = c_rate >= c_threshold and len(c_map) >= 3
        state_ok = s_rate >= s_threshold and len(s_map) >= 3
        if country_ok and state_ok:
            if s_rate > c_rate or (s_rate == c_rate and hinted_state and not hinted_country):
                country_ok = False
            else:
                state_ok = False
        # a country-named column beats a near-equal code column (names read better on a map)
        c_score = c_rate + (0.03 if hinted_country else 0.0) + (0.02 if _code_kind(series) == "name" else 0.0)
        if country_ok and (best_country is None or c_score > best_country[0]):
            best_country = (c_score, column, c_map, c_rate)
        s_score = s_rate + (0.03 if hinted_state else 0.0)
        if state_ok and (best_state is None or s_score > best_state[0]):
            best_state = (s_score, column, s_map, s_rate)

    if best_country is not None:
        _, column, mapping, rate = best_country
        result["country"] = {"column": column, "kind": _code_kind(df[column]), "match_rate": round(rate, 3), "matched": len(mapping)}
    if best_state is not None:
        _, column, mapping, rate = best_state
        result["us_state"] = {"column": column, "kind": _code_kind(df[column]), "match_rate": round(rate, 3), "matched": len(mapping)}

    lat_col = lon_col = None
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column]):
            continue
        values = df[column].dropna()
        if values.empty:
            continue
        if lat_col is None and _LAT_HINT.match(str(column)) and values.between(-90, 90).mean() > 0.98:
            lat_col = column
        elif lon_col is None and _LON_HINT.match(str(column)) and values.between(-180, 180).mean() > 0.98:
            lon_col = column
    if lat_col is not None and lon_col is not None:
        result["lat_lon"] = {"lat": lat_col, "lon": lon_col}
    return result


def map_locations(series: pd.Series, kind: str) -> tuple[pd.Series, int]:
    """Resolve a series to Plotly location codes. Returns (codes, unresolved_count)."""
    resolver = state_to_code if kind == "us_state" else country_to_iso3
    cache: dict[str, str | None] = {}

    def resolve(value: object) -> str | None:
        key = str(value)
        if key not in cache:
            cache[key] = resolver(value)
        return cache[key]

    codes = series.map(resolve)
    unresolved = int(series.notna().sum() - codes.notna().sum())
    return codes, unresolved


def canonical_country_name(iso3: str) -> str:
    return COUNTRY_NAMES.get(iso3, iso3)


def known_iso3(values: Iterable[str]) -> list[str]:
    return [v for v in values if v in _ISO3_CODES]
