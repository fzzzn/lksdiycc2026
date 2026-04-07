package main

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"

	"github.com/joho/godotenv"
)

func main() {
	if err := godotenv.Load(".env"); err != nil {
		log.Printf("no .env file found, using system environment variables")
	}

	apiPrediction := mustGetEnv("API_PREDICTION")
	apiForecast := mustGetEnv("API_FORECAST")
	apiGatewayKey := mustGetEnv("API_GATEWAY_KEY")
	port := getEnvOrDefault("PORT", "3000")

	templatesDir := filepath.Join(".", "templates")

	predictionProxy := newAPIGatewayProxy(apiPrediction, apiGatewayKey)
	forecastProxy := newAPIGatewayProxy(apiForecast, apiGatewayKey)

	mux := http.NewServeMux()
	mux.HandleFunc("/", staticPageHandler("GET", filepath.Join(templatesDir, "index.html")))
	mux.HandleFunc("/prediction", staticPageHandler("GET", filepath.Join(templatesDir, "prediction.html")))
	mux.HandleFunc("/forecasting", staticPageHandler("GET", filepath.Join(templatesDir, "forecasting.html")))
	mux.HandleFunc("/api/predict", methodHandler("POST", predictionProxy))
	mux.HandleFunc("/api/forecast", methodHandler("POST", forecastProxy))

	addr := ":" + port
	log.Printf("lm-frontend listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

func staticPageHandler(method string, filePath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method {
			w.Header().Set("Allow", method)
			http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
			return
		}

		http.ServeFile(w, r, filePath)
	}
}

func methodHandler(method string, handler http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method {
			w.Header().Set("Allow", method)
			http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
			return
		}

		handler.ServeHTTP(w, r)
	}
}

func newAPIGatewayProxy(rawTarget string, apiKey string) http.Handler {
	targetURL, err := url.Parse(rawTarget)
	if err != nil {
		log.Fatalf("invalid target URL %q: %v", rawTarget, err)
	}

	proxy := httputil.NewSingleHostReverseProxy(targetURL)
	proxy.Director = func(req *http.Request) {
		req.URL.Scheme = targetURL.Scheme
		req.URL.Host = targetURL.Host
		req.URL.Path = targetURL.Path
		req.URL.RawPath = targetURL.RawPath
		req.URL.RawQuery = targetURL.RawQuery
		req.Host = targetURL.Host
		req.Header.Set("x-api-key", apiKey)
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("proxy error: %v", err)
		http.Error(w, "upstream service unavailable", http.StatusBadGateway)
	}

	return proxy
}

func mustGetEnv(key string) string {
	value := os.Getenv(key)
	if value == "" {
		log.Fatalf("missing required environment variable: %s", key)
	}
	return value
}

func getEnvOrDefault(key string, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
