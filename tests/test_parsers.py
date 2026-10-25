import math

from fifa_value.data.clean import (
    parse_position_rating,
    parse_work_rate,
    parse_year,
)


def test_parse_position_rating():
    assert parse_position_rating("85+3") == 88.0
    assert parse_position_rating("74-2") == 72.0
    assert parse_position_rating("81") == 81.0
    assert parse_position_rating(" 80 + 2 ") == 82.0
    assert math.isnan(parse_position_rating(None))
    assert math.isnan(parse_position_rating("N/A"))


def test_parse_work_rate():
    att, defense = parse_work_rate("High/Medium")
    assert att == 3.0
    assert defense == 2.0

    att, defense = parse_work_rate("Medium/Low")
    assert att == 2.0
    assert defense == 1.0

    att, defense = parse_work_rate(None)
    assert math.isnan(att) and math.isnan(defense)


def test_parse_year():
    assert parse_year("2024") == 2024.0
    assert parse_year("2021.0") == 2021.0
    assert parse_year("June 30, 2022") == 2022.0
    assert parse_year("2014-07-01") == 2014.0
    assert math.isnan(parse_year(None))
