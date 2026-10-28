from memetagger.characters import CharacterDB, default_db, franchise_tag, split_danbooru


def test_franchise_tag():
    assert franchise_tag("k-on!") == "k-on"
    assert franchise_tag("re:zero_kara_hajimeru_isekai_seikatsu") == "re_zero"
    assert franchise_tag("kirby_(series)") == "kirby"
    assert franchise_tag("fate/grand_order") == "fate"
    assert franchise_tag("some_new_show") == "some_new_show"
    assert split_danbooru("rem_(re:zero)") == ("rem", "re:zero") and split_danbooru("cirno") == ("cirno", None)


def test_bundled_db_matching():
    db = default_db()
    assert len(db.by_tag) > 20000
    assert db.match("Yui Hirasawa") == ("hirasawa_yui", "k-on")
    assert db.match("Reimu Hakurei") == ("hakurei_reimu", "touhou")
    assert db.match("Rem", "Re:Zero") == ("rem", "re_zero")
    assert db.match("Rem (Re:Zero)") == ("rem", "re_zero")
    assert db.match("Asuka", "Evangelion") == ("souryuu_asuka_langley", "evangelion")   # franchise is never contradicted
    assert db.match("Goku", "Dragon Ball Z") == ("son_goku", "dragon_ball")
    assert db.match("Yui") is None and db.match("Miku") is None                        # ambiguous one-word names
    assert db.match("Some Random Guy", "Unknown Show") is None
    assert db.franchise_of("hirasawa_yui") == "k-on" and db.known_franchise("Re:Zero") == "re_zero"


def test_small_db_rules():
    db = CharacterDB([("aqua_(konosuba)", "kono_subarashii_sekai_ni_shukufuku_wo!", 4000),
                      ("aqua_(kingdom_hearts)", "kingdom_hearts", 900)])
    assert db.match("Aqua") is None                           # two popular candidates
    assert db.match("Aqua", "KonoSuba") == ("aqua", "konosuba")
    assert db.match("Aqua", "Kingdom Hearts") == ("aqua", "kingdom_hearts")
