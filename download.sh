#!/bin/bash

echo "🔧 Setting up environment..."
# (Optional) create virtual environment and activate it
# python3 -m venv venv
# source venv/bin/activate

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🧼 Preprocessing data..."
python preprocess.py

echo "🧠 Training the model..."
python train.py

echo "📊 Evaluating the model..."
python evaluate.py

echo "💾 Saving model and results..."
python save_model.py

echo "✅ Pipeline complete."
