"""config.yaml is the single source of settings, so check that it is consistent."""


def test_fifteen_members(cfg):
    assert len(cfg["countries"]) == 15
    assert len({c["iso3"] for c in cfg["countries"]}) == 15


def test_every_member_has_a_border_entry(cfg):
    members = {c["iso3"] for c in cfg["countries"]}
    assert set(cfg["borders"]) == members


def test_neighbors_are_members_and_borders_are_two_way(cfg):
    members = set(cfg["borders"])
    for country, neighbors in cfg["borders"].items():
        for n in neighbors:
            assert n in members, f"{n} is not a member"
            assert country in cfg["borders"][n], f"{country}-{n} is listed only one way"


def test_cabo_verde_has_no_land_borders(cfg):
    assert cfg["borders"]["CPV"] == []


def test_exited_members_are_members(cfg):
    assert set(cfg["exited_2025"]) <= set(cfg["borders"])
