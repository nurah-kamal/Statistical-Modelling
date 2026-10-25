"""South African public holidays and working days.

A Python port of src/holidays.js, src/declared.js and src/workdays.js from
https://github.com/nurah-kamal/reporting-and-bi/tree/main/sa-turnaround. The rules and sources are in that
repo's docs/HOLIDAY_RULES.md. Declared days are checked for 2009-2026 only.
"""
from datetime import date, timedelta

# Schedule 1 of the Public Holidays Act 36 of 1994, in force from 1995.
FIXED = [
    (1, 1, "New Year's Day"),
    (3, 21, "Human Rights Day"),
    (4, 27, "Freedom Day"),
    (5, 1, "Workers' Day"),
    (6, 16, "Youth Day"),
    (8, 9, "National Women's Day"),
    (9, 24, "Heritage Day"),
    (12, 16, "Day of Reconciliation"),
    (12, 25, "Christmas Day"),
    (12, 26, "Day of Goodwill"),
]

# Once-off days declared under section 2A of the Act.
DECLARED = {
    date(2009, 4, 22): "National and provincial elections",
    date(2011, 5, 18): "Local government elections",
    date(2011, 12, 27): "Declared holiday (Christmas fell on a Sunday)",
    date(2014, 5, 7): "National and provincial elections",
    date(2016, 8, 3): "Local government elections",
    date(2016, 12, 27): "Declared holiday (Christmas fell on a Sunday)",
    date(2019, 5, 8): "National and provincial elections",
    date(2021, 11, 1): "Local government elections",
    date(2022, 12, 27): "Declared holiday (Christmas fell on a Sunday)",
    date(2023, 12, 15): "Rugby World Cup celebration",
    date(2024, 5, 29): "National and provincial elections",
    date(2026, 11, 4): "Local government elections",
}


def easter_sunday(year):
    # Anonymous Gregorian algorithm.
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


_cache = {}


def holidays_for(year):
    """Every public holiday in a year as {date: name}."""
    if year < 1995:
        raise ValueError("The Public Holidays Act applies from 1995")
    if year in _cache:
        return _cache[year]
    easter = easter_sunday(year)
    listed = [(date(year, m, d), name) for m, d, name in FIXED]
    listed += [(easter - timedelta(days=2), "Good Friday"), (easter + timedelta(days=1), "Family Day")]
    out = {}
    for d, name in listed:
        out[d] = f"{out[d]} and {name}" if d in out else name
    # Section 2(1): a holiday on a Sunday makes the Monday a holiday, unless that
    # Monday is already one.
    for d, name in sorted(out.items()):
        if d.weekday() == 6 and d + timedelta(days=1) not in out:
            out[d + timedelta(days=1)] = f"{name} (observed)"
    for d, name in DECLARED.items():
        if d.year == year and d not in out:
            out[d] = name
    _cache[year] = dict(sorted(out.items()))
    return _cache[year]


def is_holiday(d):
    return d in holidays_for(d.year)


def is_working_day(d):
    return d.weekday() < 5 and not is_holiday(d)


def working_days_between(start, end):
    """Working days after start, up to and including end (same day is 0)."""
    count, d = 0, start
    while d < end:
        d += timedelta(days=1)
        count += is_working_day(d)
    return count
