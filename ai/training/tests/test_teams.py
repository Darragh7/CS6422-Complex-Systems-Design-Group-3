import pytest

from ai.training import teams


@pytest.mark.parametrize("variants, canonical", [
    (["Man United", "Manchester United", "Manchester United FC"], "Manchester United"),
    (["Ath Madrid", "Atletico Madrid", "Club Atlético de Madrid"], "Atletico Madrid"),
    (["Paris SG", "Paris Saint-Germain", "Paris Saint-Germain FC"], "Paris Saint-Germain"),
    (["Inter", "Inter Milan", "FC Internazionale Milano"], "Inter Milan"),
    (["Brighton", "Brighton & Hove Albion FC", "Brighton and Hove Albion"], "Brighton and Hove Albion"),
    (["Bayern Munich", "FC Bayern München"], "Bayern Munich"),
    (["Dortmund", "Borussia Dortmund"], "Borussia Dortmund"),
    (["M'gladbach", "Borussia Mönchengladbach"], "Borussia Monchengladbach"),
    (["Nott'm Forest", "Nottingham Forest FC"], "Nottingham Forest"),
    (["Wolves", "Wolverhampton Wanderers FC"], "Wolverhampton Wanderers"),
])
def test_spellings_resolve_to_one_canonical_name(variants, canonical):
    assert {teams.canonical_team(v) for v in variants} == {canonical}


def test_normalisation_ignores_case_accents_and_punctuation():
    assert teams.canonical_team("  man   UNITED ") == "Manchester United"
    assert teams.canonical_team("Atlético Madrid") == "Atletico Madrid"
    assert teams.canonical_team("St. Pauli") == "St Pauli"


def test_every_alias_points_at_a_known_club():
    for aliases in (teams.SOURCE_ALIASES, teams.API_ALIASES):
        assert set(aliases.values()) <= set(teams.CANONICAL_TEAMS)


def test_canonical_names_are_unique_and_do_not_clash_after_normalising():
    keys = [teams.normalise_key(t) for t in teams.CANONICAL_TEAMS]
    assert len(keys) == len(set(keys)) == len(teams.CANONICAL_TEAMS)


def test_unknown_name_is_returned_cleaned_or_raises_when_strict():
    assert teams.canonical_team("Some  Unknown FC") == "Some Unknown FC"
    with pytest.raises(teams.UnknownTeamError):
        teams.canonical_team("Some Unknown FC", strict=True)


def test_unmatched_lists_only_unresolved_names():
    assert teams.unmatched(["Man United", "Zzz Rovers", "Zzz Rovers", "Arsenal FC"]) == ["Zzz Rovers"]


def test_check_cli_reports_unmatched_names(tmp_path, capsys):
    good = tmp_path / "PL_2024.csv"
    good.write_text("id,home_team,away_team\n1,Arsenal FC,Chelsea FC\n", encoding="utf-8")
    bad = tmp_path / "XX_2024.csv"
    bad.write_text("id,home_team,away_team\n1,Arsenal FC,Mystery Town\n", encoding="utf-8")
    assert teams.main(["--check", str(good)]) == 0
    assert teams.main(["--check", str(bad)]) == 1
    assert "Mystery Town" in capsys.readouterr().out
