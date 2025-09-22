import 'dart:io';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class SoundDetector {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool _isRecording = false;

  /// Initialize recorder
  Future<void> init() async {
    await _recorder.openRecorder();
  }

  /// Start recording audio
  Future<void> startRecording() async {
    if (_isRecording) return;
    await _recorder.startRecorder(
      toFile: 'audio.aac',
      codec: Codec.aacADTS,
    );
    _isRecording = true;
  }

  /// Stop recording audio and send to AI model
  Future<Map<String, dynamic>?> stopRecordingAndAnalyze() async {
    if (!_isRecording) return null;
    _isRecording = false;
    final path = await _recorder.stopRecorder();
    if (path == null) return null;

    File audioFile = File(path);

    // Send audio file to backend AI model
    final request = http.MultipartRequest(
      'POST',
      Uri.parse('http://YOUR_BACKEND_IP:5000/analyze'),
    );
    request.files.add(await http.MultipartFile.fromPath('file', audioFile.path));

    final response = await request.send();
    final respStr = await response.stream.bytesToString();

    if (response.statusCode == 200) {
      return json.decode(respStr);
    } else {
      throw Exception("AI model failed: $respStr");
    }
  }

  /// Dispose recorder
  void dispose() {
    _recorder.closeRecorder();
  }
}
