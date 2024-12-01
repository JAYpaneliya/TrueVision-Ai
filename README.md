# TrueVision AI 🧠

## Building AI Powered Solution for Assisting Visually Impaired Individuals

### Problem Statement

Visually impaired individuals often face challenges in understanding their environment, reading visual content, and performing tasks that rely on sight. This project aims to leverage **Generative AI** and other AI technologies to assist visually impaired individuals in perceiving and interacting with their surroundings. 

There is a need for an intelligent, adaptable, and user-friendly solution that provides:

- Real-time scene understanding
- Text-to-speech conversion for reading visual content
- Object and obstacle detection for safe navigation
- Personalized assistance for daily tasks

### Task

This project involves developing an AI-powered application using **Streamlit** that provides assistive functionalities through image analysis. The application allows users to upload an image and implements the following features:

1. **Real-Time Scene Understanding**: Generate descriptive textual output that interprets the content of the uploaded image, enabling users to understand the scene effectively.
2. **Text-to-Speech Conversion for Visual Content**: Extract text from the uploaded image using **OCR** (Optical Character Recognition) techniques and convert it into audible speech for seamless content accessibility.
3. **Object and Obstacle Detection for Safe Navigation**: Identify objects or obstacles within the image and highlight them, offering insights to enhance user safety and situational awareness.
4. **Personalized Assistance for Daily Tasks**: Provide task-specific guidance based on the uploaded image, such as recognizing items, reading labels, or providing context-specific information.

## Features

- **Scene Interpretation**: Analyze and describe the content of uploaded images, helping users understand the environment in which they are.
- **Text-to-Audio Conversion**: Convert extracted text from images into speech to help users access written content seamlessly.
- **Object Detection**: Identify and highlight objects within an image, ensuring safe navigation.
- **Assistive Guidance**: Offer task-specific assistance by analyzing images, helping users interact with their environment more effectively.

## Technologies Used

- **Streamlit**: For building the web application interface.
- **Google Gemini API**: For leveraging generative AI to provide intelligent responses based on images.
- **Tesseract OCR**: For extracting text from images.
- **PyGame**: For playing audio files created from text-to-speech.
- **PyTorch and TorchVision**: For object detection using the Faster R-CNN model.
- **gTTS (Google Text-to-Speech)**: For converting text to speech.
- **PIL (Python Imaging Library)**: For image processing.

## Installation

To run this application locally, follow the steps below:

### Prerequisites

- Python 3.7 or higher
- Install required libraries using the following command:

```bash
pip install -r requirements.txt
