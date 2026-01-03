import math
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


class UKBuoyData:
    def __init__(self, station_id=None):
        self.api_key = os.getenv("METOFFICE_API_KEY")
        self.base_url = "http://datapoint.metoffice.gov.uk/public/data/val/wxobs/all/json"
        self.station_id = station_id

        # Column names as they appear in the DataFrame (with units)
        self.met_features = [
            "WindSpeed (knots)",
            "AirTemperature (degrees_C)",
            "AtmosphericPressure (millibars)",
        ]
        self.wave_features = [
            "WaveHeight (meters)",
            "Hmax (meters)",
            "Tp (seconds)",
        ]

    def list_stations(self):
        """Returns available station metadata (id, name, lat, lon)."""
        params = {"res": "hourly"}
        if self.api_key:
            params["key"] = self.api_key

        try:
            resp = requests.get(
                f"{self.base_url}/sitelist", params=params, timeout=15
            )
        except requests.RequestException as exc:
            raise RuntimeError("Met Office station list request failed") from exc

        if resp.status_code == 401:
            raise RuntimeError(
                "Met Office API key missing or invalid. Set METOFFICE_API_KEY."
            )
        if not resp.ok:
            raise RuntimeError(
                f"Met Office station list request failed with status {resp.status_code}"
            )

        try:
            data = resp.json()
            locations = data["Locations"]["Location"]
        except (ValueError, KeyError) as exc:
            raise RuntimeError(
                "Met Office station list payload missing expected fields"
            ) from exc

        if isinstance(locations, dict):
            locations = [locations]

        stations = []
        for loc in locations:
            stations.append(
                {
                    "id": str(loc.get("id") or loc.get("i")),
                    "name": loc.get("name") or loc.get("n"),
                    "latitude": self._to_float(loc.get("latitude") or loc.get("lat")),
                    "longitude": self._to_float(loc.get("longitude") or loc.get("lon")),
                }
            )
        return stations

    def fetch_data(self, days_back=30):
        """Fetches the full dataset from the Met Office for the chosen station."""
        stations = self.list_stations()
        station_ids = [s["id"] for s in stations if s.get("id")]
        if not station_ids:
            raise RuntimeError("Met Office station list was empty.")

        target_station = self.station_id or station_ids[0]
        if target_station not in station_ids:
            raise ValueError(
                f"Invalid station_id '{self.station_id}'. Valid IDs: {', '.join(station_ids)}"
            )
        self.station_id = target_station

        params = {"res": "hourly"}
        if self.api_key:
            params["key"] = self.api_key

        try:
            resp = requests.get(
                f"{self.base_url}/{self.station_id}", params=params, timeout=15
            )
        except requests.RequestException as exc:
            raise RuntimeError("Met Office observations request failed") from exc

        if resp.status_code == 401:
            raise RuntimeError(
                "Met Office API key missing or invalid. Set METOFFICE_API_KEY."
            )
        if not resp.ok:
            raise RuntimeError(
                f"Met Office observations request failed with status {resp.status_code}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise RuntimeError("Met Office observations response is not valid JSON") from exc

        df = self._parse_observations(payload)
        if df.empty:
            raise RuntimeError("Met Office returned no data for the requested station.")

        cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=days_back)
        df = df[df.index >= cutoff]

        for col in self.wave_features:
            if col not in df.columns:
                df[col] = math.nan

        ordered_cols = ["station_id"] + self.met_features + self.wave_features
        df = df.reindex(columns=ordered_cols)
        df.index.name = "time (UTC)"
        return df

    def _parse_observations(self, payload):
        """Transforms Met Office JSON into the Irish-style DataFrame schema."""
        try:
            location = payload["SiteRep"]["DV"]["Location"]
        except KeyError as exc:
            raise RuntimeError(
                "Met Office observations payload missing expected fields"
            ) from exc

        periods = location.get("Period", [])
        if isinstance(periods, dict):
            periods = [periods]

        rows = []
        for period in periods:
            base_date_str = period.get("value")
            base_date = self._parse_base_date(base_date_str)
            reps = period.get("Rep", [])
            if isinstance(reps, dict):
                reps = [reps]

            for rep in reps:
                ts = self._timestamp_from_rep(base_date, rep.get("$"))
                if ts is None:
                    continue
                row = {
                    "time (UTC)": pd.Timestamp(ts),
                    "station_id": str(
                        location.get("i")
                        or location.get("id")
                        or self.station_id
                        or ""
                    ),
                    "WindSpeed (knots)": self._to_knots(rep.get("S")),
                    "AirTemperature (degrees_C)": self._to_float(rep.get("T")),
                    "AtmosphericPressure (millibars)": self._to_float(rep.get("P")),
                    "WaveHeight (meters)": math.nan,
                    "Hmax (meters)": math.nan,
                    "Tp (seconds)": math.nan,
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.set_index("time (UTC)").sort_index()
        df = df.dropna(subset=self.met_features, how="all")
        return df

    @staticmethod
    def _parse_base_date(base_date_str):
        if not base_date_str:
            return None
        try:
            return datetime.strptime(base_date_str, "%Y-%m-%dZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return None

    @staticmethod
    def _timestamp_from_rep(base_date, offset_minutes):
        if base_date is None:
            return None
        try:
            minutes = int(offset_minutes)
        except (TypeError, ValueError):
            return None
        return base_date + timedelta(minutes=minutes)

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    def _to_knots(self, value):
        speed = self._to_float(value)
        if math.isnan(speed):
            return math.nan
        return speed * 0.868976  # mph to knots


if __name__ == "__main__":
    client = UKBuoyData()
    df = client.fetch_data(days_back=7)
    print(df.head())
    print("Columns:", list(df.columns))
