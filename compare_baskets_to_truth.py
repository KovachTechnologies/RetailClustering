#!/usr/bin/env python3
"""
Compare basket CSVs to a customer truth set.

Basket files (customer_record_id, item_description):
    df_intersect_true.csv
    df_intersect_false.csv
    df_non_intersect_true.csv
    df_non_intersect_false.csv

Truth set (u_customer_record_id, item_description):
    df_customer_filtered.csv

For each basket file and each customer_record_id:
    - look up that customer in the truth set
    - print each basket item; items also in the truth set are printed in red
    - print |basket ∩ truth| vs |truth items for that customer|

Also writes compare_baskets_report.html (red = match) next to this script.
If openpyxl is installed, also writes compare_baskets_report.xlsx with red font.

Match rule: item_description equal after strip, internal-whitespace collapse,
and case-fold. Customer ids are compared as stripped strings.

Usage:
    python compare_baskets_to_truth.py
    python compare_baskets_to_truth.py --data-dir /path/to/csvs
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
from collections import defaultdict
from pathlib import Path

BASKET_FILES = [
    "df_intersect_true.csv",
    "df_intersect_false.csv",
    "df_non_intersect_true.csv",
    "df_non_intersect_false.csv",
]
TRUTH_FILE = "df_customer_filtered.csv"

RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


def enable_windows_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def norm_item(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def norm_id(value: str) -> str:
    return (value or "").strip()


def find_column(fieldnames: list[str] | None, candidates: list[str]) -> str:
    if not fieldnames:
        raise ValueError("CSV has no header row")
    lookup = {name.strip().casefold(): name for name in fieldnames}
    for candidate in candidates:
        hit = lookup.get(candidate.casefold())
        if hit:
            return hit
    raise ValueError(
        f"Could not find any of {candidates} in columns {list(fieldnames)}"
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def load_truth(path: Path) -> dict[str, list[str]]:
    """customer id -> truth item descriptions, original order, blanks dropped."""
    fieldnames, rows = read_csv(path)
    id_col = find_column(
        fieldnames, ["u_customer_record_id", "customer_record_id"]
    )
    item_col = find_column(fieldnames, ["item_description", "item"])
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        customer_id = norm_id(row.get(id_col, ""))
        item = (row.get(item_col) or "").strip()
        if customer_id and item:
            grouped[customer_id].append(item)
    return grouped


def load_baskets(path: Path) -> dict[str, list[str]]:
    fieldnames, rows = read_csv(path)
    id_col = find_column(fieldnames, ["customer_record_id", "u_customer_record_id"])
    item_col = find_column(fieldnames, ["item_description", "item"])
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        customer_id = norm_id(row.get(id_col, ""))
        item = (row.get(item_col) or "").strip()
        if customer_id and item:
            grouped[customer_id].append(item)
    return grouped


def unique_norm(items: list[str]) -> set[str]:
    return {norm_item(item) for item in items if norm_item(item)}


def compare_customer(
    customer_id: str, basket_items: list[str], truth_items: list[str]
) -> dict:
    truth_norms = unique_norm(truth_items)
    basket_norms = unique_norm(basket_items)
    matched_norms = basket_norms & truth_norms
    return {
        "customer_id": customer_id,
        "basket_items": basket_items,
        "truth_items": truth_items,
        "n_basket_rows": len(basket_items),
        "n_truth_rows": len(truth_items),
        "n_basket_unique": len(basket_norms),
        "n_truth_unique": len(truth_norms),
        "n_matches": len(matched_norms),
        "matched_norms": matched_norms,
        "in_truth": truth_norms,
    }


def console_report(file_name: str, results: list[dict]) -> None:
    print()
    print(f"{BOLD}===== {file_name} ====={RESET}")
    if not results:
        print("  (no customer rows)")
        return

    file_matches = 0
    file_truth = 0
    for result in results:
        customer_id = result["customer_id"]
        n_matches = result["n_matches"]
        n_truth = result["n_truth_unique"]
        file_matches += n_matches
        file_truth += n_truth
        ratio = (n_matches / n_truth) if n_truth else 0.0

        print(f"\ncustomer_record_id: {customer_id}")
        if not result["truth_items"]:
            print("  no truth-set rows for this customer")
        for item in result["basket_items"]:
            if norm_item(item) in result["in_truth"]:
                print(f"  {RED}{item}{RESET}")
            else:
                print(f"  {item}")
        print(
            f"  matches: {n_matches} / {n_truth} truth items ({ratio:.1%})"
        )

    file_ratio = (file_matches / file_truth) if file_truth else 0.0
    print(
        f"\nFILE {file_name}: {file_matches} matches / {file_truth} truth items "
        f"across {len(results)} customers ({file_ratio:.1%})"
    )


def html_section(file_name: str, results: list[dict]) -> str:
    parts = [f"<h2>{html.escape(file_name)}</h2>"]
    if not results:
        parts.append("<p>(no customer rows)</p>")
        return "\n".join(parts)

    file_matches = 0
    file_truth = 0
    for result in results:
        n_matches = result["n_matches"]
        n_truth = result["n_truth_unique"]
        file_matches += n_matches
        file_truth += n_truth
        ratio = (n_matches / n_truth) if n_truth else 0.0
        parts.append(
            f"<h3>customer_record_id: {html.escape(result['customer_id'])}</h3>"
        )
        parts.append("<ul>")
        if not result["truth_items"]:
            parts.append("<li><em>no truth-set rows for this customer</em></li>")
        for item in result["basket_items"]:
            escaped = html.escape(item)
            if norm_item(item) in result["in_truth"]:
                parts.append(f'<li class="match">{escaped}</li>')
            else:
                parts.append(f"<li>{escaped}</li>")
        parts.append("</ul>")
        parts.append(
            f"<p><strong>matches: {n_matches} / {n_truth} truth items "
            f"({ratio:.1%})</strong></p>"
        )
    file_ratio = (file_matches / file_truth) if file_truth else 0.0
    parts.append(
        f"<p class='file-total'>FILE total: {file_matches} matches / "
        f"{file_truth} truth items across {len(results)} customers "
        f"({file_ratio:.1%})</p>"
    )
    return "\n".join(parts)


def write_html(path: Path, sections: list[str]) -> None:
    body = "\n".join(sections)
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Basket vs truth set</title>
  <style>
    body {{ font-family: Consolas, Menlo, monospace; margin: 24px; }}
    li.match {{ color: #c40000; font-weight: 700; }}
    .file-total {{ background: #f4f4f4; padding: 8px; }}
  </style>
</head>
<body>
  <h1>Basket vs truth set</h1>
  <p>Red items are basket purchases whose item_description is in that customer's truth set.</p>
  {body}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def write_xlsx(path: Path, all_results: list[tuple[str, list[dict]]]) -> bool:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        return False

    red_font = Font(color="FF0000", bold=True)
    header_fill = PatternFill("solid", fgColor="D9E2F3")
    workbook = Workbook()
    summary = workbook.active
    summary.title = "summary"
    summary.append(
        [
            "file",
            "customer_record_id",
            "basket_rows",
            "truth_rows",
            "unique_basket_items",
            "unique_truth_items",
            "matches",
            "match_rate",
        ]
    )
    for cell in summary[1]:
        cell.fill = header_fill
        cell.font = Font(bold=True)

    for file_name, results in all_results:
        sheet = workbook.create_sheet(file_name.replace(".csv", "")[:31])
        sheet.append(["customer_record_id", "item_description", "in_truth_set"])
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = Font(bold=True)
        for result in results:
            rate = (
                result["n_matches"] / result["n_truth_unique"]
                if result["n_truth_unique"]
                else 0.0
            )
            summary.append(
                [
                    file_name,
                    result["customer_id"],
                    result["n_basket_rows"],
                    result["n_truth_rows"],
                    result["n_basket_unique"],
                    result["n_truth_unique"],
                    result["n_matches"],
                    rate,
                ]
            )
            for item in result["basket_items"]:
                matched = norm_item(item) in result["in_truth"]
                sheet.append([result["customer_id"], item, "Y" if matched else "N"])
                if matched:
                    sheet.cell(sheet.max_row, 2).font = red_font
                    sheet.cell(sheet.max_row, 3).font = red_font
            sheet.append(
                [
                    result["customer_id"],
                    (
                        f"matches: {result['n_matches']} / "
                        f"{result['n_truth_unique']} truth items"
                    ),
                    "",
                ]
            )
            sheet.cell(sheet.max_row, 2).font = Font(bold=True)
            sheet.append([])
        sheet.column_dimensions["A"].width = 24
        sheet.column_dimensions["B"].width = 60
        sheet.column_dimensions["C"].width = 16

    summary.column_dimensions["A"].width = 32
    summary.column_dimensions["B"].width = 24
    workbook.save(path)
    return True


def main() -> int:
    enable_windows_ansi()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Folder containing the five CSVs (default: current directory)",
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    truth_path = data_dir / TRUTH_FILE
    if not truth_path.exists():
        print(f"Missing truth set: {truth_path}", file=sys.stderr)
        return 1

    truth = load_truth(truth_path)
    print(f"Truth set: {truth_path} ({len(truth)} customers)")

    html_sections: list[str] = []
    all_results: list[tuple[str, list[dict]]] = []

    for file_name in BASKET_FILES:
        basket_path = data_dir / file_name
        if not basket_path.exists():
            print(f"Missing basket file: {basket_path}", file=sys.stderr)
            continue
        baskets = load_baskets(basket_path)
        results = [
            compare_customer(customer_id, items, truth.get(customer_id, []))
            for customer_id, items in baskets.items()
        ]
        console_report(file_name, results)
        html_sections.append(html_section(file_name, results))
        all_results.append((file_name, results))

    html_path = data_dir / "compare_baskets_report.html"
    write_html(html_path, html_sections)
    print(f"\nWrote {html_path}")

    xlsx_path = data_dir / "compare_baskets_report.xlsx"
    if write_xlsx(xlsx_path, all_results):
        print(f"Wrote {xlsx_path}")
    else:
        print("openpyxl not installed; skipped xlsx. pip install openpyxl to enable it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
