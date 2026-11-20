"""News taxonomy classifier for financial & crypto alpha modeling."""

import re
from typing import Dict, List, Tuple

CATEGORY_PATTERNS = {
    "Crypto_Blockchain": [
        r"\bbitcoin\b", r"\bbtc\b", r"\bcryptocurrenc(y|ies)\b", r"\bcrypto\b",
        r"\bblockchain\b", r"\bethereum\b", r"\beth\b", r"\bsolana\b", r"\bbinance\b",
        r"\bcoinbase\b", r"\bmicrostrategy\b", r"\bsatoshi\b", r"\bdefi\b", r"\bweb3\b",
        r"\btokenomics\b", r"\bstablecoin\b", r"\btether\b", r"\busdt\b", r"\busdc\b",
        r"\bmining pool\b", r"\bhash rate\b", r"\bhalving\b", r"\bcrypto exchange\b",
        r"\bdigital asset\b", r"\bdigital currency\b", r"\bcardano\b", r"\bripple\b",
        r"\bxrp\b", r"\bblackrock ibit\b"
    ],
    "Monetary_Policy": [
        r"\bfederal reserve\b", r"\bjerome powell\b", r"\bfomc\b", r"\binterest rates?\b",
        r"\brate cut\b", r"\brate hike\b", r"\bquantitative easing\b", r"\bquantitative tightening\b",
        r"\bcentral bank\b", r"\binflation\b", r"\bcpi\b", r"\bpce\b", r"\bcore pce\b",
        r"\bmonetary policy\b", r"\bliquidity\b", r"\byield curve\b", r"\becb\b",
        r"\bbank of japan\b", r"\bbank of england\b", r"\bdiscount rate\b",
        r"\becon[_ ,-]*inflation\b", r"\bepu[_ ,-]*policy\b", r"\bmonetary\b"
    ],
    "Regulation_Legal": [
        r"\bsec\b", r"\bgensler\b", r"\bcftc\b", r"\bregulat(or|ory|ion|ions)\b",
        r"\blawsuit\b", r"\bsubpoena\b", r"\bsanction(s)?\b", r"\banti-money laundering\b",
        r"\baml\b", r"\bkyc\b", r"\bsecurities and exchange commission\b",
        r"\bdepartment of justice\b", r"\bdoj\b", r"\betf approval\b", r"\betf denial\b",
        r"\bfinra\b", r"\binsider trading\b", r"\bclass action\b",
        r"\bwb[_ ,-]*justice\b", r"\btax[_ ,-]*fncact\b"
    ],
    "Macroeconomics": [
        r"\bgdp\b", r"\brecession\b", r"\bunemployment\b", r"\bnon-farm payroll(s)?\b",
        r"\bdebt ceiling\b", r"\btreasury yield(s)?\b", r"\bfiscal policy\b",
        r"\bconsumer sentiment\b", r"\bstagflation\b", r"\beconomic growth\b",
        r"\bjobless claims\b", r"\bconsumer spending\b", r"\bdeficit\b"
    ],
    "Geopolitics_Conflict": [
        r"\bwar\b", r"\bmilitary\b", r"\bmissile(s)?\b", r"\bnato\b", r"\brussia\b",
        r"\bukraine\b", r"\btaiwan\b", r"\bmiddle east\b", r"\bceasefire\b",
        r"\bgeopolitical\b", r"\btrade war\b", r"\btariff(s)?\b", r"\binvasion\b",
        r"\barmed conflict\b"
    ],
    "Tech_Cybersecurity": [
        r"\bcyberattack\b", r"\bhack(ed|er|ing|s)?\b", r"\bransomware\b", r"\bexploit\b",
        r"\bdata breach\b", r"\bartificial intelligence\b", r"\bsemiconductor(s)?\b",
        r"\bquantum computing\b", r"\bcloud outage\b", r"\bmalware\b", r"\bzero-day\b",
        r"\bcyber\b", r"\bcybersecurity\b"
    ],
    "Energy_Commodities": [
        r"\bcrude oil\b", r"\bbrent\b", r"\bwti\b", r"\bopec\b", r"\bnatural gas\b",
        r"\bgold price\b", r"\bsilver price\b", r"\belectricity grid\b", r"\bpower grid\b",
        r"\benergy crisis\b", r"\bpetroleum\b"
    ],
    "Equities_FinancialMarkets": [
        r"\bs&p 500\b", r"\bnasdaq\b", r"\bdow jones\b", r"\bwall street\b",
        r"\bearnings report\b", r"\bstock market\b", r"\bbear market\b", r"\bbull market\b",
        r"\bbanking crisis\b", r"\bcredit default swap\b", r"\bhedge fund\b",
        r"\bprivate equity\b"
    ]
}

# Pre-compile regexes for high-throughput batch execution
COMPILED_RULES: Dict[str, List[re.Pattern]] = {
    category: [re.compile(pat, re.IGNORECASE) for pat in patterns]
    for category, patterns in CATEGORY_PATTERNS.items()
}


def classify_article(
    url: str,
    title: str = "",
    themes: str = "",
    orgs: str = ""
) -> Tuple[str, List[str]]:
    """
    Classify a news article into a primary category and secondary tags.
    Crypto is evaluated first to ensure crypto-regulatory/macro news is prioritized for BTC modeling.
    """
    # GDELT theme codes are underscore/comma-delimited (e.g. ECON_INFLATION,EPU_POLICY).
    # A tokenized copy lets word-boundary rules match the individual code words too.
    theme_tokens = re.sub(r"[,_\-\s]+", " ", themes or "")
    search_text = f"{url} {title or ''} {orgs or ''} {themes or ''} {theme_tokens}".lower()
    
    matched_categories = []
    for category, patterns in COMPILED_RULES.items():
        for pat in patterns:
            if pat.search(search_text):
                matched_categories.append(category)
                break

    if not matched_categories:
        return "General_News", []

    # Primary category is the first matched (prioritizing Crypto_Blockchain, Monetary_Policy, etc.)
    primary_category = matched_categories[0]
    secondary_tags = matched_categories[1:]

    return primary_category, secondary_tags
