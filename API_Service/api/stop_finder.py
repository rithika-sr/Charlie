# api/stop_finder.py
"""
MBTA Stop Finder with Fuzzy Matching

Loads ALL MBTA stops at startup (subway, bus, commuter rail, ferry, etc.)
and provides intelligent fuzzy matching for messy user input.
"""

import logging
import os
from typing import List, Dict, Any, Optional

import httpx
from rapidfuzz import fuzz, process

logger = logging.getLogger("stop_finder")

MBTA_BASE = "https://api-v3.mbta.com"
MBTA_API_KEY = os.getenv("MBTA_API_KEY", "")


class StopFinder:
    """
    Loads all MBTA stops once at startup and provides fuzzy matching.
    
    Supports:
    - Subway (Red, Orange, Blue, Green lines)
    - Light Rail (Green Line branches, Mattapan)
    - Commuter Rail (all lines)
    - Bus (all routes)
    - Ferry
    - Silver Line
    """

    ROUTE_TYPES = {
        0: "Light Rail (Green Line, Mattapan)",
        1: "Heavy Rail (Red, Orange, Blue)",
        2: "Commuter Rail",
        3: "Bus",
        4: "Ferry",
    }

    def __init__(self) -> None:
        self.stops: List[Dict[str, Any]] = []
        self.stops_by_id: Dict[str, Dict[str, Any]] = {}
        self._load_stops()

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.api+json"}
        if MBTA_API_KEY:
            headers["x-api-key"] = MBTA_API_KEY
        return headers

    def _load_stops(self) -> None:
        """Load ALL stops from MBTA API."""
        logger.info("Loading all MBTA stops...")
        
        all_stops: Dict[str, Dict[str, Any]] = {}
        
        try:
            url = f"{MBTA_BASE}/stops"
            params = {"page[limit]": 10000, "sort": "name"}
            
            response = httpx.get(
                url, 
                params=params, 
                headers=self._get_headers(),
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("data", []):
                stop = self._parse_stop(item)
                if stop:
                    all_stops[stop["id"]] = stop
                    
            logger.info(f"Loaded {len(all_stops)} total stops from MBTA API")
            
        except Exception as e:
            logger.error(f"Error loading stops: {e}")
            self._load_stops_by_route_type(all_stops)
        
        self.stops = list(all_stops.values())
        self.stops_by_id = all_stops
        logger.info(f"StopFinder initialized with {len(self.stops)} stops")

    def _load_stops_by_route_type(self, all_stops: Dict[str, Dict[str, Any]]) -> None:
        """Fallback: load stops by route type."""
        for route_type in [0, 1, 2, 3, 4]:
            try:
                url = f"{MBTA_BASE}/stops"
                params = {"filter[route_type]": route_type, "page[limit]": 5000}
                
                response = httpx.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                for item in data.get("data", []):
                    stop = self._parse_stop(item)
                    if stop and stop["id"] not in all_stops:
                        all_stops[stop["id"]] = stop
                        
            except Exception as e:
                logger.warning(f"Error loading route_type {route_type}: {e}")

    def _parse_stop(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a stop from MBTA API response."""
        try:
            attrs = item.get("attributes", {})
            stop_id = item.get("id")
            name = attrs.get("name")
            
            if not stop_id or not name:
                return None
            
            municipality = attrs.get("municipality") or ""
            description = attrs.get("description") or ""
            latitude = attrs.get("latitude")
            longitude = attrs.get("longitude")
            wheelchair = attrs.get("wheelchair_boarding")
            
            full = f"{name} ({municipality})" if municipality else name
            aliases = self._build_aliases(name, municipality, description)
            
            return {
                "id": stop_id,
                "name": name,
                "municipality": municipality,
                "description": description,
                "full": full,
                "aliases": aliases,
                "latitude": latitude,
                "longitude": longitude,
                "wheelchair_accessible": wheelchair == 1,
            }
            
        except Exception:
            return None

    def _build_aliases(self, name: str, municipality: str, description: str) -> List[str]:
        """Build search aliases for a stop."""
        aliases = [name.lower()]
        
        if municipality:
            aliases.append(f"{name} {municipality}".lower())
        if description:
            aliases.append(description.lower())
        
        name_lower = name.lower()
        
        if "station" in name_lower:
            aliases.append(name_lower.replace(" station", ""))
            
        # Common abbreviations
        abbreviations = {
            "jfk/umass": ["jfk", "umass", "jfk umass"],
            "hynes convention center": ["hynes", "hynes convention"],
            "boston college": ["bc", "boston college"],
            "north station": ["north sta", "north"],
            "south station": ["south sta", "south"],
            "park street": ["park st", "park"],
            "downtown crossing": ["dtx", "downtown"],
            "government center": ["gov center", "govt center"],
            "massachusetts avenue": ["mass ave", "mass avenue"],
            "harvard": ["harvard square"],
            "central": ["central square"],
            "kenmore": ["kenmore square"],
        }
        
        for key, values in abbreviations.items():
            if key in name_lower:
                aliases.extend(values)
        
        return list(set(aliases))

    def best_match(self, query: str, min_score: int = 50) -> Optional[Dict[str, Any]]:
        """Find the best matching stop for a query."""
        if not query or not self.stops:
            return None
        
        query = query.strip()
        query_lower = query.lower()
        
        # 1. Exact stop_id match
        if query in self.stops_by_id:
            stop = self.stops_by_id[query]
            return {**stop, "score": 100, "match_type": "exact_id"}
        
        # 2. Exact name match
        for stop in self.stops:
            if stop["name"].lower() == query_lower:
                return {**stop, "score": 100, "match_type": "exact_name"}
        
        # 3. Alias match
        for stop in self.stops:
            if query_lower in stop["aliases"]:
                return {**stop, "score": 95, "match_type": "alias"}
        
        # 4. Fuzzy match
        choices = [s["full"] for s in self.stops]
        
        result = process.extractOne(
            query, 
            choices, 
            scorer=fuzz.WRatio,
            score_cutoff=min_score
        )
        
        if result is None:
            result = process.extractOne(
                query,
                choices,
                scorer=fuzz.partial_ratio,
                score_cutoff=min_score
            )
        
        if result is None:
            return None
        
        match_str, score, idx = result
        stop = self.stops[idx]
        
        logger.info(f"StopFinder: '{query}' -> '{stop['name']}' (score: {score})")
        return {**stop, "score": score, "match_type": "fuzzy"}

    def search(self, query: str, limit: int = 10, min_score: int = 40) -> List[Dict[str, Any]]:
        """Return multiple matching stops sorted by relevance."""
        if not query or not self.stops:
            return []
        
        choices = [s["full"] for s in self.stops]
        
        results = process.extract(
            query.strip(),
            choices,
            scorer=fuzz.WRatio,
            limit=limit,
            score_cutoff=min_score
        )
        
        matches = []
        for match_str, score, idx in results:
            stop = self.stops[idx]
            matches.append({**stop, "score": score})
        
        return matches

    def get_stop_by_id(self, stop_id: str) -> Optional[Dict[str, Any]]:
        """Get a stop by its exact ID."""
        return self.stops_by_id.get(stop_id)

    def get_stops_count(self) -> int:
        """Return total number of loaded stops."""
        return len(self.stops)


# Global singleton instance
stop_finder = StopFinder()