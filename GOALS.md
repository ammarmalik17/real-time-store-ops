# Edge-Based Store Intelligence System  
**Computer Vision → Operations Dashboard → AI Staffing Recommendations**

## Project Overview

This project demonstrates an **end-to-end, edge-deployed computer vision system** that converts existing retail camera feeds into actionable staffing and operations intelligence—without requiring new hardware, cloud dependence, or invasive data collection.

The system ingests **RTSP/IP camera streams or recorded footage**, performs **real-time people counting and occupancy tracking on-device**, visualizes trends through a **full-stack operations dashboard**, and generates **AI-driven staffing recommendations** based on historical traffic patterns.

This repository is designed as a **production-ready project** showcasing:
- Full ownership of the **computer vision pipeline**
- Real-time **performance optimization**
- **Hardware-agnostic deployment**
- Practical business outcomes (staffing efficiency, queue management, occupancy awareness)

> **Skill focus:** Computer Vision, OpenCV, Python, real-time systems, full-stack delivery

---

## Core Objectives & Deliverables

### Objective 1: People Counting & Occupancy Intelligence (Core CV System)

**Goal:**  
Accurately count store entry and exit events in real time using existing cameras while preventing double-counting and maintaining stable performance on consumer-grade hardware.

#### What Will Be Built
- Tripwire-based people counting system  
- Direction-aware entry/exit detection  
- Real-time occupancy calculation  
- Historical footfall aggregation  
- Support for RTSP streams and `.mp4` files  
- Privacy-safe processing (no identity storage)

#### Technical Approach
- **Detection:** Lightweight person detector (YOLOv8n-class model)  
- **Tracking:** SORT (Simple Online and Realtime Tracking)  
- **Counting Logic:**  
  - Virtual tripwire placed across entrance ROI  
  - Count triggered only when a tracked centroid crosses the line  
  - Direction vector validation (IN vs OUT)

#### Double-Counting Prevention
- Persistent track IDs across frames  
- Directional crossing validation  
- Temporal debounce window (ignore re-crossing within short interval)  
- Centroid distance thresholds to prevent lingering re-counts  

#### Camera & Deployment Considerations (Client FAQ Coverage)
- **Recommended angle:** 45–60° facing entrance (top-down preferred if available)  
- **ROI definition:** Narrow vertical strip across doorway to reduce false positives  
- **Minimum resolution:** 720p (1080p recommended)  
- **Lighting tolerance:** Standard retail lighting, shadow-tolerant  
- **Occlusion handling:** SORT maintains IDs through brief occlusions  

#### Deliverables
- Real-time IN/OUT counters  
- Live occupancy value  
- Hourly, daily, weekly footfall summaries  
- SQLite persistence layer  
- Configurable camera & ROI setup  

#### Success Metrics
- ≥85% counting accuracy on test footage  
- 15+ FPS on 720p streams (CPU-only baseline)  
- <100 ms latency for occupancy updates  
- Stable tracking with 40–50 simultaneous detections  

---

### Objective 2: Operations Dashboard (Full-Stack Visualization)

**Goal:**  
Expose real-time and historical store activity through a clean, responsive dashboard that turns raw CV outputs into operational insight.

#### What Will Be Built
- Web-based operations dashboard  
- Live occupancy and trend visualization  
- Customer-to-staff ratio tracking  
- Peak hour identification  
- Zone & queue monitoring indicators  
- Real-time alert notifications  

#### Technical Approach

**Backend**
- Python-based inference service  
- WebSocket / REST API for live updates  
- SQLite-backed metrics storage  
- Rule engine for alert generation  

**Frontend**
- Lightweight React or Vue dashboard  
- Real-time charts (hourly/daily/weekly)  
- Live video feed with tripwire overlay  
- Alert banners for staffing and queue issues  
- Responsive design (desktop & tablet)  

#### Deliverables
- Live occupancy display  
- Footfall trend charts  
- Customer-to-staff ratio visualization  
- Zone/queue status indicators  
- Alert notification panel  

#### Success Metrics
- Dashboard loads in <2 seconds  
- Live updates every 1 second  
- Handles 7+ days of historical data smoothly  
- No perceptible UI lag during real-time updates  

---

### Objective 3: AI Staffing Recommendations (Intelligence Layer)

**Goal:**  
Translate historical footfall patterns into actionable staffing recommendations and real-time alerts.

#### What Will Be Built
- Historical traffic pattern analysis  
- Hourly staffing recommendations  
- Under/over-staffing alerts  
- Queue pressure detection  
- Recommendation confidence scoring  

#### Technical Approach
- Rolling historical window analysis (minimum 7 days)  
- Hour-of-day demand modeling  
- Rule-based + statistical heuristics (edge-friendly)  
- Threshold-based alerting with cooldowns  
- Confidence scores based on historical consistency  

#### Deliverables
- Staffing level recommendations by hour  
- Real-time under/over-staff alerts  
- Queue congestion alerts  
- Confidence scores explaining recommendations  

#### Success Metrics
- Peak-hour alignment with historical data  
- Alerts triggered within 30 seconds of threshold breach  
- 3+ actionable recommendations per day  
- Transparent confidence scoring for trust  

---

## System Architecture

```

IP Camera / Video File
↓
Frame Ingestion (RTSP / MP4)
↓
Person Detection (YOLOv8n)
↓
Multi-Object Tracking (SORT)
↓
Tripwire Logic (IN / OUT)
↓
Occupancy & Metrics Aggregation
↓
SQLite Persistence
↓
Backend API (WebSocket / REST)
↓
Operations Dashboard
↓
AI Staffing & Alert Engine

```

---

## Deployment & Performance Optimization

### Hardware-Agnostic Design
- No reliance on Jetson or proprietary accelerators  
- CPU-first baseline; GPU optional  
- Adaptive FPS processing based on device capability  
- Dockerized deployment for portability  

### Performance Strategies
- Frame skipping on high-resolution streams  
- ROI-restricted inference (entrance only)  
- Lightweight detection + tracking (YOLOv8n + SORT)  
- Batched frame processing where possible  
- Memory-safe tracking buffers and cleanup  

### Edge Challenges & Solutions

| Challenge | Solution |
|--------|--------|
| Limited compute | Lightweight models, ROI cropping |
| Memory constraints | Aggregated metrics storage, no raw video retention |
| Network bandwidth | Local inference, no cloud streaming |
| Hardware variability | Config-driven FPS & confidence thresholds |

---

## Privacy & Compliance

- No facial recognition  
- No identity tracking  
- No video storage by default  
- Only anonymized counts and aggregates stored  
- Designed for GDPR-friendly deployment  

---

## Cross-Platform Compatibility

This system is designed to operate seamlessly on both **Windows and Linux platforms**, ensuring broad deployment flexibility for diverse retail environments.

### Platform Support
- **Windows 10/11**: Full support for consumer and enterprise Windows environments
- **Linux distributions**: Optimized for Ubuntu, CentOS, and other common server distributions
- **Hardware-agnostic**: Works on consumer-grade hardware on both platforms

### Cross-Platform Considerations
- **File Path Handling**: Using Python's `pathlib` for platform-agnostic file operations
- **Video Processing**: Abstracted video capture with platform-specific backend support (FFMPEG, DirectShow on Windows, V4L2 on Linux)
- **Hardware Acceleration**: Optional GPU acceleration detection with CPU fallback on both platforms
- **Process Management**: Cross-platform process handling and service management
- **Environment Configuration**: Platform-specific environment variable management

### Deployment Strategy
- **Docker Support**: Multi-platform Docker images for consistent deployment
- **Virtual Environment**: Cross-platform Python virtual environment setup
- **Platform-Specific Optimizations**: Adaptive performance settings based on platform capabilities

---

## Deliverables Summary

- Fully working edge-based CV system  
- Real-time people counting & occupancy tracking  
- Operations dashboard (frontend + backend)  
- AI staffing recommendation engine  
- Real-time alerting system  
- Dockerized deployment  
- Configuration & setup documentation  

---

## Success Criteria (Portfolio-Ready)

- End-to-end system runs locally on consumer hardware  
- Demonstrates real-time CV + full-stack integration  
- Answers common client technical concerns upfront  
- Clearly scoped, realistic, and production-minded  
- Strong foundation for Phase 2 behavioral analytics  

---

## Phase 2 (Future Extension – Not Included)

- Dwell-time analysis  
- Heatmaps & flow paths  
- Behavioral pattern recognition  
- Multi-camera correlation  

---

**This project showcases practical, deployable computer vision—not research demos—built with real-world constraints, measurable outcomes, and clear business value.**