"""
geoip_lookup.py

Wraps GeoIP resolution so the rest of the pipeline doesn't care whether it's
backed by a real MaxMind GeoLite2 database or a synthetic fallback table.

--- To use REAL GeoLite2 data for your actual submission ---
1. Create a free MaxMind account: https://www.maxmind.com/en/geolite2/signup
2. Under "Manage License Keys", generate a key, then download:
     GeoLite2-City.mmdb
     GeoLite2-ASN.mmdb
3. Place both files in ./geoip_data/ (same folder as this script)
4. pip install geoip2 --break-system-packages

This sandbox has no network access to MaxMind's servers, so this module
automatically falls back to a synthetic-but-deterministic IP->geo table when
the real .mmdb files aren't found, so the pipeline still runs end-to-end
offline. Swap in the real files later and NOTHING else in your pipeline
needs to change — that's the point of wrapping it behind get_geo().
"""

import hashlib
import os

GEOIP_DIR = "geoip_data"
CITY_DB = os.path.join(GEOIP_DIR, "GeoLite2-City.mmdb")
ASN_DB = os.path.join(GEOIP_DIR, "GeoLite2-ASN.mmdb")

_city_reader = None
_asn_reader = None
_real_geoip_available = False

try:
    import geoip2.database  # noqa: F401

    if os.path.exists(CITY_DB) and os.path.exists(ASN_DB):
        _city_reader = geoip2.database.Reader(CITY_DB)
        _asn_reader = geoip2.database.Reader(ASN_DB)
        _real_geoip_available = True
except ImportError:
    pass  # geoip2 package not installed — fine, fallback handles it

_FALLBACK_COUNTRIES = ["US", "DE", "IN", "SG", "NL", "RU", "BR", "JP", "GB", "FR"]
# Approximate country centroids, used only by the synthetic fallback so geo-velocity
# math (haversine distance) still has something plausible to work with offline.
_FALLBACK_CENTROIDS = {
    "US": (39.8, -98.6), "DE": (51.2, 10.4), "IN": (22.4, 78.6), "SG": (1.35, 103.8),
    "NL": (52.1, 5.3), "RU": (61.5, 105.3), "BR": (-10.3, -53.2), "JP": (36.2, 138.3),
    "GB": (54.0, -2.9), "FR": (46.6, 2.5),
}


_lookup_stats = {"real": 0, "fallback": 0}


def _synthetic_geo(ip: str) -> dict:
    """Deterministic fake geo/asn/lat/lon derived from the IP itself, so the same IP
    always resolves the same way even without a real database. Lat/lon here are the
    country's centroid plus small deterministic jitter, NOT a real location."""
    _lookup_stats["fallback"] += 1
    h = int(hashlib.md5(ip.encode()).hexdigest(), 16)
    country = _FALLBACK_COUNTRIES[h % len(_FALLBACK_COUNTRIES)]
    asn = f"AS{1000 + (h % 64000)}"
    base_lat, base_lon = _FALLBACK_CENTROIDS[country]
    jitter = ((h % 1000) / 1000.0 - 0.5) * 4  # +/- 2 degrees, deterministic per IP
    return {"geo_country": country, "asn": asn, "lat": base_lat + jitter, "lon": base_lon + jitter}


def get_geo(ip: str) -> dict:
    """Single entry point the rest of the pipeline should call. Returns
    {"geo_country":..., "asn":..., "lat":..., "lon":...} regardless of backend."""
    if _real_geoip_available:
        try:
            city_resp = _city_reader.city(ip)
            asn_resp = _asn_reader.asn(ip)
            _lookup_stats["real"] += 1
            return {
                "geo_country": city_resp.country.iso_code or "UNKNOWN",
                "asn": f"AS{asn_resp.autonomous_system_number}",
                "lat": city_resp.location.latitude,
                "lon": city_resp.location.longitude,
            }
        except Exception:
            # e.g. private/reserved IP ranges aren't in the DB — fall back cleanly,
            # but COUNT it so you know how much of your dataset is actually fake geo data
            return _synthetic_geo(ip)
    return _synthetic_geo(ip)


def get_stats() -> dict:
    """Returns {"real": n, "fallback": n} — report this in your write-up so
    you're not overstating how much of the dataset used real GeoLite2 data."""
    return dict(_lookup_stats)


def using_real_geoip() -> bool:
    return _real_geoip_available


if __name__ == "__main__":
    test_ip = "8.8.8.8"
    print(f"Real GeoLite2 active: {using_real_geoip()}")
    print(f"get_geo({test_ip}) -> {get_geo(test_ip)}")
