"""
watchlist.py
Turns one model's forecast-year scores into the watch list.
Used by analysis.py (Baseline C) and gnn.py (the version 2 main model), so the
status, member rank, tier, and context rules are the same for every model.
The settings are in config.yaml under watch_list.
"""


def build_watch_list(scores, cfg):
    """
    scores: a DataFrame with one row per member and at least the columns iso3,
            risk (the model's score from 0 to 1), and rank, sorted by rank.
    Returns the same rows with status, member_rank, tier, context, and a 0 to 100 score.
    """
    watch = cfg["watch_list"]
    exited = cfg["exited_2025"]

    def member_status(iso3):
        """Left ECOWAS, suspended, or a member in good standing."""
        if iso3 in exited:
            return "Left ECOWAS (2025)"
        if iso3 in watch["suspended"]:
            return watch["suspended"][iso3]
        return "Member"

    def tier(risk, status):
        """One tier rule for every output and the risk map."""
        if status == "Left ECOWAS (2025)":
            return "Left ECOWAS"
        if risk >= watch["tiers"]["high"]:
            return "High"
        if risk >= watch["tiers"]["elevated"]:
            return "Elevated"
        return "Lower"

    now = scores.copy()
    now["status"] = [member_status(iso) for iso in now.iso3]
    # Rank only the current members (the three that left are scored but not ranked)
    members = now[now.status != "Left ECOWAS (2025)"].copy()
    members["member_rank"] = members["risk"].rank(ascending=False, method="first").astype(int)
    now = now.merge(members[["iso3", "member_rank"]], on="iso3", how="left")
    now["tier"] = [tier(r, st) for r, st in zip(now.risk, now.status)]
    now["context"] = now.iso3.map(watch["context"]).fillna("")
    now["risk_score_0_100"] = (now.risk * 100).round(0).astype(int)
    return now
