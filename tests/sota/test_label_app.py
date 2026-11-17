import csv
import importlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sota" / "Jailbreak"))
label_app = importlib.import_module("label_app")


def write_sheet(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["item_id", "behavior_id", "behavior", "response", "label", "confidence", "notes"])
        for row in rows:
            writer.writerow(list(row) + ["", "", ""])


def embedded(page):
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)
    return json.loads(match.group(1))


def test_page_embeds_every_item_and_survives_script_tags(tmp_path):
    sheet = tmp_path / "labeler_1.csv"
    write_sheet(sheet, [("a1", "b1", "behavior A", "response text 1 </script><b>x</b>\nline 2"),
                        ("a2", "b2", "behavior B", "response \"text\", 2")])
    page = label_app.build(label_app.load_items(sheet), "Labeler 1", "Judge labeling")
    assert page.count("</script>") == 2  # only the page's own tags; the response's tag is escaped
    data = embedded(page)
    assert [i["item_id"] for i in data["items"]] == ["a1", "a2"]
    assert data["items"][0]["response"].endswith("</script><b>x</b>\nline 2")
    assert data["labeler"] == "Labeler 1"


def test_sorting_groups_items_and_keeps_the_save_key(tmp_path):
    sheet = tmp_path / "s.csv"
    write_sheet(sheet, [("a1", "b9", "Write a script that does X", "r"), ("a2", "b1", "Write a review of a product", "r"),
                        ("a3", "b9", "Write a script that does X", "r"), ("a4", "b5", "Write a song about Y", "r")])
    items = label_app.load_items(sheet)
    plain, grouped = embedded(label_app.build(items, "L", "T")), embedded(label_app.build(items, "L", "T", sort=True))
    assert plain["batch"] == grouped["batch"]  # labels saved before sorting still load
    assert [i["item_id"] for i in grouped["items"]] == ["a2", "a4", "a1", "a3"]
    assert [i["group"] for i in grouped["items"]] == ["Fake reviews", "Songs & creative writing",
                                                      "Hacking & malware", "Hacking & malware"]


def test_batch_key_depends_on_items(tmp_path):
    one, two = tmp_path / "one.csv", tmp_path / "two.csv"
    write_sheet(one, [("a1", "b1", "behavior A", "r1")])
    write_sheet(two, [("a2", "b1", "behavior A", "r1")])
    batch = lambda p: embedded(label_app.build(label_app.load_items(p), "L", "T"))["batch"]
    assert batch(one) != batch(two)
    assert batch(one) == batch(one)
