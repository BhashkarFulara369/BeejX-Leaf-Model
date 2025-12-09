# 📱 Flutter TFLite Integration Guide

This guide explains how to integrate your `model.tflite` into your BeejX Flutter App and handle English-to-Hindi translation professionally.

## 1. Setup Dependencies
Add these to your `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  tflite_flutter: ^0.10.1  # For running the model
  image: ^4.0.17           # For image resizing/processing
```

## 2. Add Assets
1.  Create an `assets/` folder in your Flutter project root.
2.  Copy your `model.tflite` and `labels.txt` (the English version) into it.
3.  Register them in `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/model.tflite
    - assets/labels.txt
```

## 3. The "Translation Engine" (Code Strategy)
Instead of editing the text file, create a helper class to map English keys to Hindi display text.

Create a file `lib/services/disease_localizer.dart`:

```dart
class DiseaseLocalizer {
  // The 'Key' is what comes from labels.txt
  // The 'Value' is what you show to the Farmer
  static final Map<String, String> _hindiMap = {
    "Mandua_blast": "मंडुए का रोग (Blast)",
    "Mandua_rust": "मंडुए का रतुआ (Rust)",
    "Mandua_healthy": "मंडुआ (स्वस्थ)",
    "Potato_Early_Blight": "आलू का झुलसा रोग (Early Blight)",
    "Potato_Healthy": "आलू (स्वस्थ)",
    // ... ADD ALL 50 CLASSES HERE ...
  };

  static String getLabel(String englishLabel) {
    return _hindiMap[englishLabel] ?? englishLabel; // Fallback to English if missing
  }
}
```

## 4. The Classifier Helper
Create `lib/services/classifier.dart` to handle the heavy lifting.

```dart
import 'dart:io';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';

class Classifier {
  Interpreter? _interpreter;
  List<String> _labels = [];

  Classifier() {
    _loadModel();
    _loadLabels();
  }

  Future<void> _loadModel() async {
    _interpreter = await Interpreter.fromAsset('model.tflite');
  }

  Future<void> _loadLabels() async {
    final labelData = await rootBundle.loadString('assets/labels.txt');
    _labels = labelData.split('\n').where((s) => s.isNotEmpty).toList();
  }

  Future<String> predict(File imageFile) async {
    if (_interpreter == null) return "Error: Model not loaded";

    // 1. Preprocess Image (Resize to 224x224)
    var image = img.decodeImage(imageFile.readAsBytesSync())!;
    image = img.copyResize(image, width: 224, height: 224);

    // 2. Convert to float32 List [1, 224, 224, 3]
    var input = List.generate(1, (i) => List.generate(224, (y) => List.generate(224, (x) {
      var pixel = image.getPixel(x, y);
      return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
    })));

    // 3. Run Inference
    var output = List.filled(1 * 50, 0.0).reshape([1, 50]);
    _interpreter!.run(input, output);

    // 4. Find Best Class
    var highestProb = 0.0;
    var bestLabelIndex = 0;
    
    for (int i = 0; i < output[0].length; i++) {
        if (output[0][i] > highestProb) {
            highestProb = output[0][i];
            bestLabelIndex = i;
        }
    }

    return _labels[bestLabelIndex]; // Returns "Mandua_blast"
  }
}
```

## 5. Usage in UI
Now, in your React/UI Code, you combine them:

```dart
// 1. Get Prediction
String rawResult = await classifier.predict(myImage); // "Mandua_blast"

// 2. Translate for User
String displayResult = DiseaseLocalizer.getLabel(rawResult); // "मंडुए का रोग (Blast)"

// 3. Show it
Text(displayResult, style: TextStyle(fontSize: 20));
```

### PRO TIP: Why do it this way?
*   **Multi-Language Support**: Later, you can easily verify `if (userLanguage == 'Hindi')` vs `if (userLanguage == 'English')`.
*   **Database Sync**: Your backend database usually works better with English Keys (`Mandua_blast`) than Hindi Unicode strings.
