'use client'

import { useEffect, useState, useCallback } from 'react'
import { MapContainer, TileLayer, Marker, Popup, Circle, Polyline, useMap } from 'react-leaflet'
import L from 'leaflet'
import { X, Navigation, Train, Bus, MapPin, Clock, AlertCircle } from 'lucide-react'
import 'leaflet/dist/leaflet.css'

// Fix Leaflet default icon issue
if (typeof window !== 'undefined') {
  delete (L.Icon.Default.prototype as any)._getIconUrl
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  })
}

const API_URL = 'https://charlie-mbta-api-588293495748.us-east1.run.app'
const MBTA_API = 'https://api-v3.mbta.com'
const MBTA_KEY = '5e6979638b10499c8bf109ff2ec64da8'

interface TrainInfo {
  route: string
  minutes_away: number
  stop_id: string
  vehicle_id?: string
}

interface Vehicle {
  id: string
  latitude: number
  longitude: number
  bearing?: number
  speed?: number
  label: string
  route_id: string
  direction_id: number
  current_status?: string
  updated_at: string
}

interface Stop {
  id: string
  name: string
  latitude: number
  longitude: number
  municipality?: string
}

interface Prediction {
  route_id: string
  arrival_time: string
  minutes_away: number
  headsign: string
}

// Custom icons for different transit types
const createVehicleIcon = (routeId: string, bearing: number = 0) => {
  const colors: { [key: string]: string } = {
    'Red': '#DA291C',
    'Orange': '#ED8B00',
    'Blue': '#003DA5',
    'Green-B': '#00843D',
    'Green-C': '#00843D',
    'Green-D': '#00843D',
    'Green-E': '#00843D',
    'default': '#FFC72C'
  }
  
  const color = colors[routeId] || colors['default']
  const isGreen = routeId.startsWith('Green')
  const routeColor = isGreen ? colors['Green-B'] : color
  
  return L.divIcon({
    html: `
      <div style="
        transform: rotate(${bearing}deg);
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
      ">
        <div style="
          width: 28px;
          height: 28px;
          background: ${routeColor};
          border-radius: 50%;
          border: 3px solid white;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          color: white;
          font-size: 14px;
        ">
          ${routeId.includes('Green') ? 'G' : routeId.charAt(0)}
        </div>
      </div>
    `,
    className: 'vehicle-marker',
    iconSize: [32, 32],
    iconAnchor: [16, 16]
  })
}

const createStopIcon = () => {
  return L.divIcon({
    html: `
      <div style="
        width: 20px;
        height: 20px;
        background: white;
        border: 3px solid #2563eb;
        border-radius: 50%;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
      "></div>
    `,
    className: 'stop-marker',
    iconSize: [20, 20],
    iconAnchor: [10, 10]
  })
}

// Component to auto-center map on data
function MapController({ center, zoom }: { center: [number, number], zoom: number }) {
  const map = useMap()
  
  useEffect(() => {
    map.setView(center, zoom)
  }, [center, zoom, map])
  
  return null
}

interface LiveTransitMapProps {
  selectedTrain: TrainInfo | null
  userLocation: [number, number] | null
  onClose: () => void
}

export default function LiveTransitMap({ selectedTrain, userLocation, onClose }: LiveTransitMapProps) {
  const [vehicles, setVehicles] = useState<Vehicle[]>([])
  const [stops, setStops] = useState<Stop[]>([])
  const [selectedStop, setSelectedStop] = useState<Stop | null>(null)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)
  const [mapCenter, setMapCenter] = useState<[number, number]>([42.3601, -71.0589]) // Boston
  const [mapZoom, setMapZoom] = useState(13)

  // Fetch live vehicles
  const fetchVehicles = useCallback(async () => {
    try {
      const params = new URLSearchParams({
        'filter[route_type]': '0,1', // Subway and light rail
        'include': 'route,trip'
      })
      
      if (selectedTrain?.route) {
        params.set('filter[route]', selectedTrain.route)
      }

      const response = await fetch(
        `${MBTA_API}/vehicles?${params.toString()}`,
        { headers: { 'x-api-key': MBTA_KEY } }
      )

      if (!response.ok) throw new Error('Failed to fetch vehicles')

      const data = await response.json()
      const vehicleData: Vehicle[] = []

      for (const vehicle of data.data) {
        const attrs = vehicle.attributes
        const relationships = vehicle.relationships
        
        if (attrs.latitude && attrs.longitude) {
          vehicleData.push({
            id: vehicle.id,
            latitude: attrs.latitude,
            longitude: attrs.longitude,
            bearing: attrs.bearing || 0,
            speed: attrs.speed || 0,
            label: attrs.label || 'Unknown',
            route_id: relationships?.route?.data?.id || 'Unknown',
            direction_id: attrs.direction_id || 0,
            current_status: attrs.current_status,
            updated_at: attrs.updated_at
          })
        }
      }

      setVehicles(vehicleData)
      
      // Auto-center on first vehicle if available
      if (vehicleData.length > 0 && selectedTrain) {
        setMapCenter([vehicleData[0].latitude, vehicleData[0].longitude])
        setMapZoom(14)
      }
    } catch (error) {
      console.error('Error fetching vehicles:', error)
    }
  }, [selectedTrain])

  // Fetch stops near location or for route
  const fetchStops = useCallback(async () => {
    try {
      const params = new URLSearchParams({
        'page[limit]': '50'
      })

      if (selectedTrain?.stop_id) {
        params.set('filter[id]', selectedTrain.stop_id)
      } else if (userLocation) {
        params.set('filter[latitude]', userLocation[0].toString())
        params.set('filter[longitude]', userLocation[1].toString())
        params.set('filter[radius]', '0.01') // ~1 km
      }

      const response = await fetch(
        `${MBTA_API}/stops?${params.toString()}`,
        { headers: { 'x-api-key': MBTA_KEY } }
      )

      if (!response.ok) throw new Error('Failed to fetch stops')

      const data = await response.json()
      const stopData: Stop[] = []

      for (const stop of data.data) {
        const attrs = stop.attributes
        if (attrs.latitude && attrs.longitude) {
          stopData.push({
            id: stop.id,
            name: attrs.name,
            latitude: attrs.latitude,
            longitude: attrs.longitude,
            municipality: attrs.municipality
          })
        }
      }

      setStops(stopData)
      
      // Center on selected stop if available
      if (stopData.length > 0 && selectedTrain?.stop_id) {
        const targetStop = stopData[0]
        setMapCenter([targetStop.latitude, targetStop.longitude])
        setMapZoom(15)
      }
    } catch (error) {
      console.error('Error fetching stops:', error)
    }
  }, [selectedTrain, userLocation])

  // Fetch predictions for a stop
  const fetchPredictions = useCallback(async (stopId: string) => {
    try {
      const response = await fetch(
        `${MBTA_API}/predictions?filter[stop]=${stopId}&sort=arrival_time&page[limit]=5`,
        { headers: { 'x-api-key': MBTA_KEY } }
      )

      if (!response.ok) throw new Error('Failed to fetch predictions')

      const data = await response.json()
      const predData: Prediction[] = []

      for (const pred of data.data) {
        const attrs = pred.attributes
        const route = pred.relationships?.route?.data?.id
        
        if (attrs.arrival_time || attrs.departure_time) {
          const arrivalTime = attrs.arrival_time || attrs.departure_time
          const now = new Date()
          const arrival = new Date(arrivalTime)
          const minutesAway = Math.round((arrival.getTime() - now.getTime()) / 60000)
          
          predData.push({
            route_id: route || 'Unknown',
            arrival_time: arrivalTime,
            minutes_away: minutesAway,
            headsign: attrs.headsign || 'Unknown'
          })
        }
      }

      setPredictions(predData)
    } catch (error) {
      console.error('Error fetching predictions:', error)
    }
  }, [])

  // Initial data fetch
  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      await Promise.all([fetchVehicles(), fetchStops()])
      setLoading(false)
    }
    
    loadData()
  }, [fetchVehicles, fetchStops])

  // Refresh vehicles every 10 seconds
  useEffect(() => {
    const interval = setInterval(fetchVehicles, 10000)
    return () => clearInterval(interval)
  }, [fetchVehicles])

  // Handle stop click
  const handleStopClick = async (stop: Stop) => {
    setSelectedStop(stop)
    await fetchPredictions(stop.id)
  }

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Map Header */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-4 shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold flex items-center gap-2">
              <Train className="w-6 h-6" />
              Live Transit Map
            </h2>
            <p className="text-sm text-blue-100 mt-1">
              Real-time vehicle positions updated every 10 seconds
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/20 rounded-lg transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Map Stats */}
        <div className="mt-4 grid grid-cols-3 gap-3">
          <div className="bg-white/20 rounded-lg p-3">
            <div className="text-xs text-blue-100">Active Vehicles</div>
            <div className="text-2xl font-bold">{vehicles.length}</div>
          </div>
          <div className="bg-white/20 rounded-lg p-3">
            <div className="text-xs text-blue-100">Nearby Stops</div>
            <div className="text-2xl font-bold">{stops.length}</div>
          </div>
          <div className="bg-white/20 rounded-lg p-3">
            <div className="text-xs text-blue-100">Map Zoom</div>
            <div className="text-2xl font-bold">{mapZoom}</div>
          </div>
        </div>
      </div>

      {/* Map Container */}
      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 bg-white bg-opacity-90 z-50 flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto"></div>
              <p className="mt-4 text-gray-600 font-medium">Loading live transit data...</p>
            </div>
          </div>
        )}

        <MapContainer
          center={mapCenter}
          zoom={mapZoom}
          style={{ height: '100%', width: '100%' }}
          className="z-0"
        >
          <MapController center={mapCenter} zoom={mapZoom} />
          
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {/* User Location */}
          {userLocation && (
            <>
              <Circle
                center={userLocation}
                radius={100}
                pathOptions={{ color: '#3b82f6', fillColor: '#3b82f6', fillOpacity: 0.2 }}
              />
              <Marker
                position={userLocation}
                icon={L.divIcon({
                  html: `
                    <div style="
                      width: 16px;
                      height: 16px;
                      background: #3b82f6;
                      border: 3px solid white;
                      border-radius: 50%;
                      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                    "></div>
                  `,
                  className: 'user-location-marker',
                  iconSize: [16, 16],
                  iconAnchor: [8, 8]
                })}
              >
                <Popup>
                  <div className="text-sm">
                    <div className="font-semibold flex items-center gap-1">
                      <Navigation className="w-4 h-4" />
                      Your Location
                    </div>
                  </div>
                </Popup>
              </Marker>
            </>
          )}

          {/* Live Vehicles */}
          {vehicles.map((vehicle) => (
            <Marker
              key={vehicle.id}
              position={[vehicle.latitude, vehicle.longitude]}
              icon={createVehicleIcon(vehicle.route_id, vehicle.bearing)}
            >
              <Popup>
                <div className="text-sm min-w-[200px]">
                  <div className="font-bold text-lg mb-2 flex items-center gap-2">
                    <Train className="w-5 h-5" />
                    {vehicle.route_id} Line
                  </div>
                  <div className="space-y-1 text-gray-700">
                    <div className="flex justify-between">
                      <span className="font-medium">Vehicle:</span>
                      <span>{vehicle.label}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium">Status:</span>
                      <span className="capitalize">{vehicle.current_status || 'In Transit'}</span>
                    </div>
                    {vehicle.speed !== undefined && vehicle.speed > 0 && (
                      <div className="flex justify-between">
                        <span className="font-medium">Speed:</span>
                        <span>{Math.round(vehicle.speed * 2.237)} mph</span>
                      </div>
                    )}
                    <div className="flex justify-between text-xs text-gray-500 mt-2 pt-2 border-t">
                      <span>Updated:</span>
                      <span>{new Date(vehicle.updated_at).toLocaleTimeString()}</span>
                    </div>
                  </div>
                </div>
              </Popup>
            </Marker>
          ))}

          {/* Transit Stops */}
          {stops.map((stop) => (
            <Marker
              key={stop.id}
              position={[stop.latitude, stop.longitude]}
              icon={createStopIcon()}
              eventHandlers={{
                click: () => handleStopClick(stop)
              }}
            >
              <Popup>
                <div className="text-sm min-w-[250px]">
                  <div className="font-bold text-base mb-2 flex items-center gap-2">
                    <MapPin className="w-4 h-4" />
                    {stop.name}
                  </div>
                  {stop.municipality && (
                    <div className="text-gray-600 mb-2">{stop.municipality}</div>
                  )}
                  <button
                    onClick={() => handleStopClick(stop)}
                    className="w-full mt-2 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-xs font-medium"
                  >
                    View Arrivals
                  </button>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>

      {/* Stop Details Panel */}
      {selectedStop && (
        <div className="border-t border-gray-200 bg-white p-4 max-h-64 overflow-y-auto">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-bold text-lg flex items-center gap-2">
              <MapPin className="w-5 h-5 text-blue-600" />
              {selectedStop.name}
            </h3>
            <button
              onClick={() => setSelectedStop(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {predictions.length > 0 ? (
            <div className="space-y-2">
              <div className="text-sm font-semibold text-gray-700 mb-2">Next Arrivals:</div>
              {predictions.map((pred, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${
                      pred.route_id.includes('Red') ? 'bg-red-600' :
                      pred.route_id.includes('Orange') ? 'bg-orange-600' :
                      pred.route_id.includes('Blue') ? 'bg-blue-600' :
                      pred.route_id.includes('Green') ? 'bg-green-600' :
                      'bg-yellow-600'
                    }`}></div>
                    <div>
                      <div className="font-semibold text-sm">{pred.route_id}</div>
                      <div className="text-xs text-gray-600">{pred.headsign}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-blue-600">
                      {pred.minutes_away < 1 ? 'Arriving' : `${pred.minutes_away} min`}
                    </div>
                    <div className="text-xs text-gray-500">
                      <Clock className="w-3 h-3 inline mr-1" />
                      {new Date(pred.arrival_time).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-6 text-gray-500">
              <AlertCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No predictions available for this stop</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
