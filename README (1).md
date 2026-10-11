# WEBION - Real-Time Live Interactive Shopping & Exhibition Platform

## 🚀 Overview
Webion is a first-of-its-kind real-time LIVE e-commerce platform with AR-powered virtual try-on. Buyers can visit shops virtually, interact with salespersons via live video, try on garments using AR, and get instant size recommendations.

## 🏗️ Architecture
- **Frontend (Buyer):** Next.js + React + Tailwind CSS + Agora Web SDK
- **Backend:** Node.js + Express + Socket.io + MySQL
- **Android (Seller):** Kotlin + Jetpack Compose + Agora Android SDK + Socket.io
- **AR Engine:** MediaPipe Pose Detection + Body Segmentation (dimension extraction)
- **Real-time Communication:** Agora RTC (video/audio) + Socket.io (data sync)

## 🔑 Key Features
1. **Live Video Shopping** - Buyer sees seller's shop camera in real-time
2. **Two-Way Video Call** - Split screen buyer + seller via Agora
3. **Negotiation Alert** - Buyer can ring seller for live assistance
4. **AR Garment Measurement** - Extract dimensions from mannequin via camera
5. **AR Body Measurement** - Measure buyer's body dimensions
6. **Size Match & Recommendation** - Compare garment vs buyer dimensions
7. **Virtual Try-On** - Overlay garment on buyer's body (AR)
8. **Product Catalog** - Browse products with MySQL backend
9. **Real-Time Data Sync** - Socket.io bridges Android seller ↔ Web buyer

## 📦 Tech Stack
| Layer | Technology |
|-------|-----------|
| Frontend | Next.js, React, Tailwind CSS |
| Backend | Node.js, Express.js |
| Database | MySQL |
| Video Call | Agora RTC SDK |
| Real-time Sync | Socket.io |
| Android App | Kotlin, Jetpack Compose |
| AR/ML | MediaPipe, TensorFlow |

## 🛠️ Setup

### Backend
```bash
cd webion-server
cp .env.example .env    # Add your credentials
npm install
npm start
```

### Frontend
```bash
cd webion-live
npm install
npm run dev
```

### Database
```bash
mysql -u root -p < database/schema.sql
```

### Android
Open `webion-live-android` in Android Studio and run on device.

## 📐 Data Flow
```text
Seller Android Camera → Agora → Buyer Web Screen
         ↓                              ↓
   AR scans mannequin          AR scans buyer body
         ↓                              ↓
   Garment dimensions          Buyer dimensions
         ↓                              ↓
   Socket.io → Server → MySQL → Size Match API
                                        ↓
                              Fit Score + Recommendation
```

## 👥 Team
Built for Hackathon by Team WEBION
