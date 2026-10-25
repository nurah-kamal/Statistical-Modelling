# Reads Stats SA's CPI History (CPIHistory.pdf, publication P0141) and writes the
# monthly headline series to cpi_headline.csv.
#   Table B1: headline index, Dec 2024 = 100, from 1980
#   Table B2: official year-on-year inflation rate (%), monthly from 1969
import re

import pandas as pd
import pdfplumber

MONTHS = 12
row = re.compile(r"^(\d{4})((?:\s+-?\d+,\d)+)\s*$")


def read_rows(pages):
    """Year rows from the given pages: {year: [values]}, with the decimal comma read."""
    rows = {}
    for page in pages:
        for line in (page.extract_text() or "").splitlines():
            m = row.match(line.strip())
            if m:
                rows[int(m.group(1))] = [float(v.replace(",", ".")) for v in m.group(2).split()]
    return rows


def monthly(rows, first_year):
    """Monthly values from year rows. A full year has 12 months and an annual figure;
    the current year has only the months published so far."""
    out = []
    for year, values in sorted(rows.items()):
        if year < first_year:
            continue
        months = values[:MONTHS] if len(values) == MONTHS + 1 else values
        if len(values) == MONTHS:
            raise ValueError(f"{year}: 12 values, cannot tell if the year is complete")
        for m, v in enumerate(months, start=1):
            out.append((pd.Timestamp(year, m, 1), v))
    return pd.Series(dict(out))


with pdfplumber.open("CPIHistory.pdf") as pdf:
    first, rest = pdf.pages[0], pdf.pages[1:]
    assert "Table B1" in first.extract_text(), "page 1 should hold Table B1"
    assert "Table B2" in rest[0].extract_text(), "page 2 should start Table B2"
    b1 = read_rows([first])
    b2 = read_rows(rest)

# Years before 1969 in Table B2 have an annual rate only
index = monthly(b1, 1980)
rate = monthly(b2, 1969)
cpi = pd.DataFrame({"index_dec2024_100": index, "yoy_rate_pct": rate}).rename_axis("month")

# Check: Table B1's annual averages against the mean of its months
for year, values in b1.items():
    if len(values) == MONTHS + 1:
        assert abs(sum(values[:MONTHS]) / MONTHS - values[MONTHS]) <= 0.1, f"B1 average off in {year}"

# Check: the official rates against the rates implied by the rounded index. Start in
# 2010: before that the index has too few digits (3.2 in 1981), and the official 2009
# rates compare against the old basket that was replaced in January 2009.
implied = (cpi["index_dec2024_100"] / cpi["index_dec2024_100"].shift(12) - 1) * 100
gap = (implied - cpi["yoy_rate_pct"])["2010":].abs()
print(f"Months: {cpi['yoy_rate_pct'].notna().sum()} of official rates "
      f"({rate.index[0]:%b %Y} to {rate.index[-1]:%b %Y}), "
      f"{cpi['index_dec2024_100'].notna().sum()} of index values")
print(f"Official rate against the rate implied by the rounded index, 2010 on: "
      f"median gap {gap.median():.2f} points, largest {gap.max():.2f}")

cpi.to_csv("cpi_headline.csv", date_format="%Y-%m")
print("Wrote cpi_headline.csv")
