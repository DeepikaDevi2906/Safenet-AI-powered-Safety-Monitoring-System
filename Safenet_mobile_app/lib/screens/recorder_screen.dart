// lib/screens/recorder_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';

class RecorderScreen extends StatefulWidget {
  const RecorderScreen({super.key}); // ✅ super.key fixes lint

  @override
  State<RecorderScreen> createState() => _RecorderScreenState();
}

class _RecorderScreenState extends State<RecorderScreen> {
  FlutterSoundRecorder? _recorder;
  bool _isRecorderInitialized = false;

  @override
  void initState() {
    super.initState();
    _recorder = FlutterSoundRecorder();
    _initRecorder();
  }

  Future<void> _initRecorder() async {
    final status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      throw RecordingPermissionException('Microphone permission not granted');
    }
    await _recorder!.openRecorder();
    setState(() {
      _isRecorderInitialized = true;
    });
  }

  @override
  void dispose() {
    _recorder?.closeRecorder();
    _recorder = null;
    super.dispose();
  }

  Future<void> startRecording() async {
    if (!_isRecorderInitialized) return;
    await _recorder!.startRecorder(toFile: 'audio.aac');
  }

  Future<void> stopRecording() async {
    if (!_isRecorderInitialized) return;
    final path = await _recorder!.stopRecorder();
    debugPrint('🎙️ Recorded file path: $path'); // ✅ safer than print
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Recorder")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: startRecording,
              child: const Text("Start Recording"),
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: stopRecording,
              child: const Text("Stop Recording"),
            ),
          ],
        ),
      ),
    );
  }
}
