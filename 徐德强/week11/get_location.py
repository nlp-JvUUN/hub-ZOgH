"""
get_location.py - 根据城市名查询经纬度

示例：
  python get_location.py --city 宁德
"""

import argparse
import json

import httpx


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


def get_location(city: str) -> dict | None:
    """
    查询城市对应的经纬度。

    Args:
        city: 城市名称，例如 "宁德"、"北京"、"上海"

    Returns:
        找到时返回包含城市名、国家、省份、经纬度的 dict；找不到时返回 None。
    """
    with httpx.Client(timeout=10.0) as client:
        def _geocode(name: str) -> list[dict]:
            resp = client.get(GEOCODING_URL, params={
                "name": name,
                "count": 10,
                "language": "zh",
                "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _geocode(city)
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))

        if is_low_admin and not has_suffix:
            retry = _geocode(city + "市")
            if retry:
                results = retry

        if not results:
            return None

        def _rank(result: dict) -> tuple[int, int]:
            feature_code = str(result.get("feature_code", ""))
            admin_priority = 1 if feature_code.startswith("PPLA") or feature_code.startswith("ADM") else 0
            population = result.get("population") or 0
            return admin_priority, population

        loc = max(results, key=_rank)
        return {
            "city": loc.get("name", city),
            "country": loc.get("country", ""),
            "admin1": loc.get("admin1", ""),
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="根据城市名查询经纬度")
    parser.add_argument("--city", required=True, help="城市中文名，如 宁德")
    parser.add_argument("--json", action="store_true", help="输出 JSON，便于传给下一步")
    args = parser.parse_args()

    try:
        location = get_location(args.city)
    except httpx.RequestError as exc:
        print(f"经纬度获取失败：{exc}")
        return

    if location is None:
        print(f"未找到城市 '{args.city}'，请尝试其他写法（如'宁德市'改'宁德'）")
        return

    if args.json:
        print(json.dumps(location, ensure_ascii=False))
        return

    location_str = f"{location['country']} {location['admin1']} {location['city']}".strip()
    print(f"城市：{location_str}")
    print(f"纬度：{location['latitude']}")
    print(f"经度：{location['longitude']}")


if __name__ == "__main__":
    main()
