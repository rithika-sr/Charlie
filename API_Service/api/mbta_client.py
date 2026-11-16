# api/mbta_client.py
"""
MBTA API Client

Complete client for interacting with the MBTA V3 API.
Supports all transit types: Subway, Light Rail, Commuter Rail, Bus, Ferry.
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import httpx

logger = logging.getLogger("mbta_client")

MBTA_BASE = "https://api-v3.mbta.com"
MBTA_API_KEY = os.getenv("MBTA_API_KEY", "")


class MBTAClient:
    """Async client for MBTA V3 API."""
    
    ROUTE_TYPES = {
        0: "Light Rail",
        1: "Heavy Rail",
        2: "Commuter Rail",
        3: "Bus",
        4: "Ferry",
    }

    def __init__(self) -> None:
        self.base_url = MBTA_BASE
        self.api_key = MBTA_API_KEY
        
    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def _get(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Make async GET request to MBTA API."""
        url = f"{self.base_url}{endpoint}"
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params, headers=self._get_headers())
            response.raise_for_status()
            return response.json()

    async def get_predictions(
        self,
        stop_id: str,
        route_id: Optional[str] = None,
        direction_id: Optional[int] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Get real-time predictions for a stop."""
        params: Dict[str, Any] = {
            "filter[stop]": stop_id,
            "include": "trip,route,stop,vehicle",
            "sort": "arrival_time",
            "page[limit]": min(limit * 2, 50),
        }
        
        if route_id:
            params["filter[route]"] = route_id
        if direction_id is not None:
            params["filter[direction_id]"] = direction_id
            
        return await self._get("/predictions", params)

    async def get_next_arrivals(
        self,
        stop_id: str,
        route_id: Optional[str] = None,
        destination: Optional[str] = None,
        direction_id: Optional[int] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get formatted next arrivals for a stop."""
        raw = await self.get_predictions(
            stop_id=stop_id,
            route_id=route_id,
            direction_id=direction_id,
            limit=limit * 2,
        )
        
        # Build lookup tables
        trips: Dict[str, Dict[str, Any]] = {}
        routes: Dict[str, Dict[str, Any]] = {}
        vehicles: Dict[str, Dict[str, Any]] = {}
        
        for item in raw.get("included", []):
            item_type = item.get("type")
            item_id = item.get("id")
            attrs = item.get("attributes", {})
            
            if item_type == "trip":
                trips[item_id] = attrs
            elif item_type == "route":
                routes[item_id] = attrs
            elif item_type == "vehicle":
                vehicles[item_id] = attrs
        
        now = datetime.now(timezone.utc)
        arrivals: List[Dict[str, Any]] = []
        
        for pred in raw.get("data", []):
            attrs = pred.get("attributes", {})
            rels = pred.get("relationships", {})
            
            trip_id = (rels.get("trip") or {}).get("data", {}).get("id")
            route_id_val = (rels.get("route") or {}).get("data", {}).get("id")
            vehicle_id = (rels.get("vehicle") or {}).get("data", {}).get("id")
            
            trip_attrs = trips.get(trip_id, {})
            route_attrs = routes.get(route_id_val, {})
            
            arrival_time_str = attrs.get("arrival_time") or attrs.get("departure_time")
            
            if not arrival_time_str:
                continue
            
            try:
                arrival_dt = datetime.fromisoformat(arrival_time_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            
            if arrival_dt < now - timedelta(minutes=1):
                continue
            
            minutes_away = (arrival_dt - now).total_seconds() / 60
            
            headsign = trip_attrs.get("headsign", "")
            route_name = route_attrs.get("long_name") or route_attrs.get("short_name") or route_id_val
            route_type = route_attrs.get("type")
            route_color = route_attrs.get("color", "")
            
            arrival = {
                "stop_id": stop_id,
                "route_id": route_id_val,
                "route_name": route_name,
                "route_type": route_type,
                "route_type_name": self.ROUTE_TYPES.get(route_type, "Unknown"),
                "route_color": f"#{route_color}" if route_color else None,
                "trip_id": trip_id,
                "headsign": headsign,
                "destination": headsign,
                "arrival_time": arrival_time_str,
                "arrival_time_local": arrival_dt.astimezone().strftime("%I:%M %p"),
                "minutes_away": round(minutes_away, 1),
                "direction_id": attrs.get("direction_id"),
                "stop_sequence": attrs.get("stop_sequence"),
                "status": attrs.get("status"),
                "vehicle_id": vehicle_id,
            }
            
            arrivals.append(arrival)
        
        # Filter by destination if specified
        if destination:
            from rapidfuzz import fuzz
            dest_lower = destination.lower()
            filtered = []
            for arr in arrivals:
                hs = (arr.get("headsign") or "").lower()
                if fuzz.partial_ratio(dest_lower, hs) >= 60:
                    filtered.append(arr)
            arrivals = filtered
        
        arrivals.sort(key=lambda x: x.get("arrival_time") or "")
        return arrivals[:limit]

    async def get_routes(self, route_type: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all routes, optionally filtered by type."""
        params: Dict[str, Any] = {"page[limit]": 1000}
        
        if route_type is not None:
            params["filter[type]"] = route_type
            
        raw = await self._get("/routes", params)
        
        routes = []
        for item in raw.get("data", []):
            attrs = item.get("attributes", {})
            routes.append({
                "id": item.get("id"),
                "name": attrs.get("long_name") or attrs.get("short_name"),
                "short_name": attrs.get("short_name"),
                "long_name": attrs.get("long_name"),
                "type": attrs.get("type"),
                "type_name": self.ROUTE_TYPES.get(attrs.get("type"), "Unknown"),
                "color": f"#{attrs.get('color')}" if attrs.get("color") else None,
                "description": attrs.get("description"),
            })
        
        return routes

    async def get_subway_routes(self) -> List[Dict[str, Any]]:
        """Get all subway routes."""
        light_rail = await self.get_routes(route_type=0)
        heavy_rail = await self.get_routes(route_type=1)
        return light_rail + heavy_rail

    async def get_bus_routes(self) -> List[Dict[str, Any]]:
        return await self.get_routes(route_type=3)

    async def get_commuter_rail_routes(self) -> List[Dict[str, Any]]:
        return await self.get_routes(route_type=2)

    async def get_ferry_routes(self) -> List[Dict[str, Any]]:
        return await self.get_routes(route_type=4)

    async def get_alerts(
        self,
        route_id: Optional[str] = None,
        stop_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get service alerts."""
        params: Dict[str, Any] = {"page[limit]": 100}
        
        if route_id:
            params["filter[route]"] = route_id
        if stop_id:
            params["filter[stop]"] = stop_id
            
        raw = await self._get("/alerts", params)
        
        alerts = []
        for item in raw.get("data", []):
            attrs = item.get("attributes", {})
            alerts.append({
                "id": item.get("id"),
                "effect": attrs.get("effect"),
                "severity": attrs.get("severity"),
                "header": attrs.get("header"),
                "description": attrs.get("description"),
                "url": attrs.get("url"),
                "updated_at": attrs.get("updated_at"),
            })
        
        return alerts


# Global client instance
mbta_client = MBTAClient()