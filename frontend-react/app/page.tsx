// Save this as: frontend/page.tsx (or app/page.tsx in Next.js)
// Run with: npm run dev

"use client";

import React, { useState, useRef, useEffect } from "react";

// ============================================================
// CONFIGURATION - Change this to your backend URL
// ============================================================
const API_URL = "http://localhost:8000";

// ============================================================
// TYPES
// ============================================================
interface Message {
  role: "user" | "bot";
  content: string;
  timestamp: Date;
  data?: any;
}

interface Arrival {
  route_name: string;
  route_id: string;
  route_type: number;
  headsign: string;
  minutes_away: number;
  arrival_time: string;
  trip_id?: string;
}

interface StationRoute {
  route_id: string;
  route_name: string;
  route_type: number;
  arrivals: Arrival[];
}

// ============================================================
// MAIN COMPONENT
// ============================================================
export default function CharlieApp() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "bot",
      content: `Welcome to Charlie, your MBTA real-time transit assistant.

I can help you with:
- Real-time train and bus arrivals
- Route planning between stations
- All MBTA lines: Red, Orange, Blue, Green, Silver, Commuter Rail, Bus, Ferry

Try saying: "I'm at Ruggles, next train to Forest Hills"`,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarView, setSidebarView] = useState<"menu" | "search" | "arrivals">("menu");
  const [stationSearch, setStationSearch] = useState("");
  const [selectedStation, setSelectedStation] = useState<string | null>(null);
  const [selectedStationName, setSelectedStationName] = useState<string>("");
  const [stationArrivals, setStationArrivals] = useState<StationRoute[]>([]);
  const [arrivalsLoading, setArrivalsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);

  // Check API status on load
  useEffect(() => {
    checkApiStatus();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Initialize speech recognition
  useEffect(() => {
    if (typeof window !== "undefined" && ("SpeechRecognition" in window || "webkitSpeechRecognition" in window)) {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = "en-US";

      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setIsListening(false);
      };

      recognitionRef.current.onerror = () => {
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      if (response.ok) {
        setApiStatus("online");
      } else {
        setApiStatus("offline");
      }
    } catch {
      setApiStatus("offline");
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const query = input;
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query }),
      });

      const data = await response.json();

      const botMessage: Message = {
        role: "bot",
        content: data.reply || "Sorry, I couldn't process that request.",
        timestamp: new Date(),
        data: data.data,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: "Connection error. Please make sure the backend is running on " + API_URL,
          timestamp: new Date(),
        },
      ]);
    }

    setLoading(false);
  };

  const toggleVoice = () => {
    if (!recognitionRef.current) {
      alert("Voice recognition not supported in your browser. Use Chrome or Edge.");
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  const loadStationArrivals = async (stopId: string, stationName: string) => {
    setSelectedStation(stopId);
    setSelectedStationName(stationName);
    setSidebarView("arrivals");
    setArrivalsLoading(true);

    try {
      const response = await fetch(`${API_URL}/station/${stopId}`);
      const data = await response.json();
      setStationArrivals(data.routes || []);
    } catch {
      setStationArrivals([]);
    }

    setArrivalsLoading(false);
  };

  const getRouteColor = (routeName: string, routeType: number): string => {
    const name = routeName.toLowerCase();
    if (name.includes("red")) return "#DA291C";
    if (name.includes("orange")) return "#ED8B00";
    if (name.includes("blue")) return "#003DA5";
    if (name.includes("green")) return "#00843D";
    if (name.includes("silver")) return "#7C878E";
    if (name.includes("commuter") || routeType === 2) return "#80276C";
    if (name.includes("ferry") || routeType === 4) return "#008EAA";
    if (routeType === 3) return "#FFC72C"; // Bus
    return "#333333";
  };

  // Station list for search
  const stations = [
    { id: "place-rugg", name: "Ruggles" },
    { id: "place-pktrm", name: "Park Street" },
    { id: "place-dwnxg", name: "Downtown Crossing" },
    { id: "place-sstat", name: "South Station" },
    { id: "place-north", name: "North Station" },
    { id: "place-harsq", name: "Harvard" },
    { id: "place-kencl", name: "Kenmore" },
    { id: "place-coecl", name: "Copley" },
    { id: "place-gover", name: "Government Center" },
    { id: "place-state", name: "State" },
    { id: "place-forhl", name: "Forest Hills" },
    { id: "place-ogmnl", name: "Oak Grove" },
    { id: "place-alfcl", name: "Alewife" },
    { id: "place-asmnl", name: "Ashmont" },
    { id: "place-brntn", name: "Braintree" },
    { id: "place-wondl", name: "Wonderland" },
    { id: "place-bbsta", name: "Back Bay" },
    { id: "place-hymnl", name: "Hynes Convention Center" },
    { id: "place-lake", name: "Boston College" },
    { id: "place-river", name: "Riverside" },
  ];

  const filteredStations = stations.filter((s) =>
    s.name.toLowerCase().includes(stationSearch.toLowerCase())
  );

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif", background: "#F7F8FA" }}>
      {/* Sidebar Overlay */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.4)",
            zIndex: 999,
          }}
        />
      )}

      {/* Sidebar */}
      <aside
        style={{
          position: "fixed",
          left: 0,
          top: 0,
          height: "100%",
          width: "320px",
          background: "#FFFFFF",
          borderRight: "1px solid #E0E0E0",
          transform: sidebarOpen ? "translateX(0)" : "translateX(-100%)",
          transition: "transform 0.3s ease",
          zIndex: 1000,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Sidebar Header */}
        <div style={{ padding: "20px", borderBottom: "1px solid #E0E0E0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ fontSize: "18px", fontWeight: 600, margin: 0 }}>Menu</h2>
          <button
            onClick={() => setSidebarOpen(false)}
            style={{
              width: "36px",
              height: "36px",
              border: "none",
              background: "#F7F8FA",
              borderRadius: "8px",
              cursor: "pointer",
              fontSize: "20px",
            }}
          >
            ×
          </button>
        </div>

        {/* Sidebar Navigation */}
        {sidebarView === "menu" && (
          <nav style={{ padding: "16px", display: "flex", flexDirection: "column", gap: "8px" }}>
            <button
              onClick={() => { setSidebarOpen(false); }}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                padding: "14px 16px",
                border: "none",
                background: "#003DA5",
                color: "white",
                borderRadius: "10px",
                cursor: "pointer",
                fontSize: "15px",
                fontWeight: 500,
              }}
            >
              <span>💬</span> Chat
            </button>
            <button
              onClick={() => setSidebarView("search")}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                padding: "14px 16px",
                border: "none",
                background: "#F7F8FA",
                borderRadius: "10px",
                cursor: "pointer",
                fontSize: "15px",
                fontWeight: 500,
              }}
            >
              <span>🔍</span> Search Station
            </button>
            <button
              onClick={() => window.open("https://www.mbta.com/schedules/subway", "_blank")}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                padding: "14px 16px",
                border: "none",
                background: "#F7F8FA",
                borderRadius: "10px",
                cursor: "pointer",
                fontSize: "15px",
                fontWeight: 500,
              }}
            >
              <span>🗺️</span> Transit Map
            </button>
          </nav>
        )}

        {/* Station Search */}
        {sidebarView === "search" && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ padding: "16px", borderBottom: "1px solid #E0E0E0" }}>
              <button
                onClick={() => setSidebarView("menu")}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  padding: "8px 12px",
                  border: "none",
                  background: "#F7F8FA",
                  borderRadius: "8px",
                  cursor: "pointer",
                  fontSize: "14px",
                  marginBottom: "12px",
                }}
              >
                ← Back
              </button>
              <input
                type="text"
                placeholder="Search stations..."
                value={stationSearch}
                onChange={(e) => setStationSearch(e.target.value)}
                style={{
                  width: "100%",
                  padding: "12px 16px",
                  border: "2px solid #E0E0E0",
                  borderRadius: "10px",
                  fontSize: "15px",
                  outline: "none",
                }}
              />
            </div>
            <div style={{ flex: 1, overflowY: "auto", padding: "8px" }}>
              {filteredStations.map((station) => (
                <div
                  key={station.id}
                  onClick={() => loadStationArrivals(station.id, station.name)}
                  style={{
                    padding: "14px 16px",
                    borderRadius: "10px",
                    cursor: "pointer",
                    marginBottom: "4px",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "#F7F8FA")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <div style={{ fontWeight: 600, fontSize: "15px" }}>{station.name}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Station Arrivals */}
        {sidebarView === "arrivals" && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ padding: "16px", borderBottom: "1px solid #E0E0E0" }}>
              <button
                onClick={() => setSidebarView("search")}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  padding: "8px 12px",
                  border: "none",
                  background: "#F7F8FA",
                  borderRadius: "8px",
                  cursor: "pointer",
                  fontSize: "14px",
                  marginBottom: "12px",
                }}
              >
                ← Back
              </button>
              <h3 style={{ fontSize: "20px", fontWeight: 700, margin: 0 }}>{selectedStationName}</h3>
            </div>
            <div style={{ flex: 1, overflowY: "auto" }}>
              {arrivalsLoading ? (
                <div style={{ padding: "40px", textAlign: "center", color: "#8A8A8A" }}>Loading arrivals...</div>
              ) : stationArrivals.length === 0 ? (
                <div style={{ padding: "40px", textAlign: "center", color: "#8A8A8A" }}>No upcoming arrivals</div>
              ) : (
                stationArrivals.map((route) => (
                  <div key={route.route_id} style={{ borderBottom: "1px solid #E0E0E0" }}>
                    <div
                      style={{
                        padding: "12px 16px",
                        background: getRouteColor(route.route_name, route.route_type),
                        color: route.route_type === 3 ? "#000" : "#FFF",
                        fontWeight: 700,
                        fontSize: "14px",
                      }}
                    >
                      {route.route_name}
                    </div>
                    {route.arrivals.slice(0, 5).map((arr, idx) => (
                      <div
                        key={idx}
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          padding: "14px 16px",
                          borderBottom: "1px solid #F7F8FA",
                        }}
                      >
                        <div>
                          <div style={{ fontWeight: 600, fontSize: "15px" }}>{arr.headsign}</div>
                          <div style={{ fontSize: "13px", color: "#8A8A8A" }}>
                            {arr.minutes_away < 1 ? "Arriving" : `${arr.minutes_away} min`}
                          </div>
                        </div>
                        <div style={{ fontSize: "15px", fontWeight: 600 }}>{arr.arrival_time}</div>
                      </div>
                    ))}
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </aside>

      {/* Main Content */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", height: "100vh" }}>
        {/* Header */}
        <header
          style={{
            background: "#FFFFFF",
            borderBottom: "1px solid #E0E0E0",
            padding: "16px 24px",
            display: "flex",
            alignItems: "center",
            gap: "16px",
          }}
        >
          <button
            onClick={() => { setSidebarOpen(true); setSidebarView("menu"); }}
            style={{
              width: "44px",
              height: "44px",
              border: "none",
              background: "#F7F8FA",
              borderRadius: "10px",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "20px",
            }}
          >
            ☰
          </button>
          <div style={{ flex: 1 }}>
            <h1 style={{ fontSize: "22px", fontWeight: 700, margin: 0 }}>Charlie</h1>
            <p style={{ fontSize: "13px", color: "#5C5C5C", margin: 0 }}>MBTA Real-Time Transit Assistant</p>
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "8px 14px",
              background: apiStatus === "online" ? "#E8F5E9" : apiStatus === "offline" ? "#FFEBEE" : "#FFF3E0",
              borderRadius: "20px",
            }}
          >
            <div
              style={{
                width: "8px",
                height: "8px",
                background: apiStatus === "online" ? "#4CAF50" : apiStatus === "offline" ? "#F44336" : "#FF9800",
                borderRadius: "50%",
              }}
            />
            <span style={{ fontSize: "13px", fontWeight: 600, color: apiStatus === "online" ? "#2E7D32" : apiStatus === "offline" ? "#C62828" : "#E65100" }}>
              {apiStatus === "online" ? "Connected" : apiStatus === "offline" ? "Offline" : "Checking..."}
            </span>
          </div>
        </header>

        {/* Chat Messages */}
        <div style={{ flex: 1, overflowY: "auto", padding: "24px", display: "flex", flexDirection: "column", gap: "16px" }}>
          {messages.map((msg, idx) => (
            <div key={idx} style={{ display: "flex", flexDirection: "column", alignItems: msg.role === "user" ? "flex-end" : "flex-start" }}>
              <div
                style={{
                  maxWidth: "75%",
                  padding: "14px 18px",
                  borderRadius: "16px",
                  borderBottomRightRadius: msg.role === "user" ? "4px" : "16px",
                  borderBottomLeftRadius: msg.role === "bot" ? "4px" : "16px",
                  background: msg.role === "user" ? "#003DA5" : "#FFFFFF",
                  color: msg.role === "user" ? "#FFFFFF" : "#1A1A1A",
                  border: msg.role === "bot" ? "1px solid #E0E0E0" : "none",
                  whiteSpace: "pre-wrap",
                  fontSize: "15px",
                  lineHeight: 1.5,
                }}
              >
                {msg.content}
              </div>
              <div style={{ fontSize: "11px", color: "#8A8A8A", marginTop: "6px", padding: "0 4px" }}>
                {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </div>
            </div>
          ))}
          {loading && (
            <div style={{ display: "flex", alignItems: "flex-start" }}>
              <div
                style={{
                  padding: "14px 18px",
                  background: "#FFFFFF",
                  border: "1px solid #E0E0E0",
                  borderRadius: "16px",
                  borderBottomLeftRadius: "4px",
                }}
              >
                <div style={{ display: "flex", gap: "4px" }}>
                  <div style={{ width: "8px", height: "8px", background: "#8A8A8A", borderRadius: "50%", animation: "pulse 1.4s infinite" }} />
                  <div style={{ width: "8px", height: "8px", background: "#8A8A8A", borderRadius: "50%", animation: "pulse 1.4s infinite 0.2s" }} />
                  <div style={{ width: "8px", height: "8px", background: "#8A8A8A", borderRadius: "50%", animation: "pulse 1.4s infinite 0.4s" }} />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div style={{ background: "#FFFFFF", borderTop: "1px solid #E0E0E0", padding: "16px 24px" }}>
          <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && sendMessage()}
              placeholder="I'm at Ruggles, next train to Forest Hills..."
              style={{
                flex: 1,
                padding: "14px 18px",
                border: "2px solid #E0E0E0",
                borderRadius: "12px",
                fontSize: "15px",
                outline: "none",
              }}
            />
            <button
              onClick={toggleVoice}
              style={{
                width: "48px",
                height: "48px",
                border: "none",
                background: isListening ? "#DA291C" : "#F7F8FA",
                borderRadius: "12px",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "20px",
              }}
            >
              🎤
            </button>
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              style={{
                width: "48px",
                height: "48px",
                border: "none",
                background: loading || !input.trim() ? "#E0E0E0" : "#003DA5",
                borderRadius: "12px",
                cursor: loading || !input.trim() ? "not-allowed" : "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "20px",
                color: "#FFFFFF",
              }}
            >
              ➤
            </button>
          </div>
        </div>
      </main>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(0.8); }
        }
      `}</style>
    </div>
  );
}
