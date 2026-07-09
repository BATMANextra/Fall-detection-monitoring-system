# Fall Detection Monitoring System for Eldery People Using Deep Learnig

## Overview

This project presents a cloud-free fall detection and monitoring system that performs real-time inference directly on an ESP32 microcontroller. Using motion data acquired from an MPU6050 inertial sensor, a lightweight 1D Convolutional Neural Network (CNN) detects falls while minimizing false alarms, making the system suitable for continuous monitoring of older adults.

## Features

* On-device fall detection using TinyML.
* Real-time processing on an ESP32 without an Internet connection.
* Lightweight CNN model optimized for embedded deployment.
* Audible and visual alerts through a buzzer and LED.
* Voting and state-machine mechanisms to improve robustness and reduce false positives.

## Hardware

* ESP32-WROOM-32
* MPU6050 IMU
* LiPo battery
* TP4056 charging module with protection
* Buzzer and status LED

## Software Stack

* Python, TensorFlow/Keras
* TensorFlow Lite for Microcontrollers
* Arduino IDE
* NumPy, Pandas, Scikit-learn

## Dataset and Model

The model was trained on the SisFall dataset, downsampled from 200 Hz to 50 Hz. A separable 1D CNN architecture was designed and optimized for deployment on resource-constrained hardware.

### Performance

* Accuracy: **98.51%**
* Recall: **98.84%**
* False Alarm Rate: **1.6%**
* Model size: approximately **174 KB**

## Project Structure

* `training/` – data preprocessing and model training scripts.
* `firmware/` – ESP32 source code and deployment files.
* `docs/` – diagrams, images, and supplementary documentation.

## Future Work

Future improvements include adding wireless notifications, testing with additional datasets, and further reducing power consumption.

## License

This project was developed as an end-of-study (PFE) project. Add an open-source license if you intend to distribute the code publicly.
