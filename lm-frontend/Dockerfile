# Build stage
FROM golang:1.25.8 AS builder

WORKDIR /workspace

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o lm-frontend .

# Runtime stage
FROM golang:1.25.8

WORKDIR /app

COPY --from=builder /workspace/lm-frontend /app/lm-frontend
COPY --from=builder /workspace/templates /app/templates
COPY .env /app/.env

EXPOSE 3000

CMD ["./lm-frontend"]
