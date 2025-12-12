# GatePlate — ML Service

GatePlate ML is a machine learning service responsible for automatic
license plate recognition from vehicle images.

The service is implemented as a **separate microservice**
and communicates with the web application via an **HTTP API**.

---

## 🧠 Role in the System

The ML service performs the following tasks:

- Receives an image via HTTP request
- Detects the license plate using a YOLO model
- Applies image preprocessing
- Extracts text using OCR (Tesseract)
- Returns the recognized license plate number in JSON format

This service does **not** contain any web interface or database logic.

---

## 🏗 Architecture



- The ML service runs independently
- Can be deployed and scaled separately
- Follows microservice architecture principles

---

## 🚀 API Endpoint

### POST `/api/recognize`

**Request**
- `multipart/form-data`
- Field name: `file`
- Value: image file (`.jpg`, `.png`)

**Response**
```json
{
  "plate": "BC1234AB",
  "found": true
}



