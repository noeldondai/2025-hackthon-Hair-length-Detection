import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Bidraj',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatelessWidget {
  // This function triggers webcam detection via Flask server
  void _runWebcamDetection() async {
    // URL updated for Chrome web mode (127.0.0.1 for localhost)
    var url = Uri.parse('http://127.0.0.1:5000/run-webcam-detection');
    var response = await http.get(url);

    if (response.statusCode == 200) {
      print('Webcam detection started');
    } else {
      print('Failed to start webcam detection');
    }
  }

  // This function triggers beard detection via Flask server
  void _runBeardDetection() async {
    // URL updated for Chrome web mode (127.0.0.1 for localhost)
    var url = Uri.parse('http://127.0.0.1:5000/run-beard-detection');
    var response = await http.get(url);

    if (response.statusCode == 200) {
      print('Beard detection started');
    } else {
      print('Failed to start beard detection');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Bidraj'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: _runWebcamDetection,
              child: Text('Start Webcam Detection'),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _runBeardDetection,
              child: Text('Start Beard Detection'),
            ),
          ],
        ),
      ),
    );
  }
}
