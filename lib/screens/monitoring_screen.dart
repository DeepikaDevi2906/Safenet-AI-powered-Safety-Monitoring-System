// lib/screens/monitoring_screen.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:socket_io_client/socket_io_client.dart' as io;

class MonitoringScreen extends StatefulWidget {
  final int userId;
  const MonitoringScreen({super.key, required this.userId});

  @override
  State<MonitoringScreen> createState() => _MonitoringScreenState();
}

class _MonitoringScreenState extends State<MonitoringScreen> {
  late CameraController _cameraController;
  late FlutterSoundRecorder _audioRecorder;
  bool monitoringEnabled = false;

  final FlutterLocalNotificationsPlugin _notifications =
      FlutterLocalNotificationsPlugin();
  io.Socket? socket;

  Timer? _frameTimer;
  Timer? _audioTimer;

  final String backendUrl = "http://YOUR_BACKEND_IP:5000";

  @override
  void initState() {
    super.initState();
    _audioRecorder = FlutterSoundRecorder();
    _initNotifications();
    _initSocket();
  }

  // ---------------- Notifications ----------------
  void _initNotifications() {
    var androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');
    var settings = InitializationSettings(android: androidSettings);
    _notifications.initialize(settings);
  }

  void _showNotification(String title, String body) async {
    var androidDetails = AndroidNotificationDetails(
      'alert_channel',
      'Alerts',
      importance: Importance.max,
      priority: Priority.high,
    );
    var platformDetails = NotificationDetails(android: androidDetails);
    await _notifications.show(0, title, body, platformDetails);
  }

  // ---------------- Socket.IO ----------------
  void _initSocket() {
    socket = io.io(backendUrl, <String, dynamic>{
      'transports': ['websocket'],
      'autoConnect': true,
    });
    socket!.onConnect((_) {
      debugPrint("✅ Connected to SocketIO");
    });
    socket!.on('new_alert', (data) {
      _showNotification(data['type'], data['message']);
    });
  }

  // ---------------- Monitoring ----------------
  Future<void> _startMonitoring() async {
    setState(() => monitoringEnabled = true);

    // Camera
    final cameras = await availableCameras();
    _cameraController = CameraController(
      cameras[0],
      ResolutionPreset.medium,
      enableAudio: false,
    );
    await _cameraController.initialize();

    // Audio
    await _audioRecorder.openRecorder();
    await _audioRecorder.startRecorder(toFile: 'temp_audio.aac');

    // Periodic camera frame sending
    _frameTimer = Timer.periodic(const Duration(seconds: 3), (_) => _sendFrame());

    // Periodic audio sending
    _audioTimer = Timer.periodic(const Duration(seconds: 5), (_) => _sendAudioChunk());
  }

  Future<void> _sendFrame() async {
    if (!monitoringEnabled || !_cameraController.value.isInitialized) return;

    try {
      final XFile file = await _cameraController.takePicture();
      final bytes = await file.readAsBytes();
      final base64Image = base64Encode(bytes);

      await http.post(
        Uri.parse("$backendUrl/send-frame"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "user_id": widget.userId.toString(),
          "frame": base64Image,
        }),
      );
    } catch (e) {
      debugPrint("Error sending frame: $e");
    }
  }

  Future<void> _sendAudioChunk() async {
    if (!monitoringEnabled) return;
    try {
      final path = await _audioRecorder.stopRecorder();
      if (path == null) return;

      final fileBytes = await File(path).readAsBytes();

      var request = http.MultipartRequest("POST", Uri.parse("$backendUrl/send-audio"));
      request.fields['user_id'] = widget.userId.toString();
      request.files.add(
        http.MultipartFile.fromBytes("audio", fileBytes, filename: "audio.aac"),
      );
      await request.send();

      // Restart recorder
      await _audioRecorder.startRecorder(toFile: 'temp_audio.aac');
    } catch (e) {
      debugPrint("Error sending audio: $e");
    }
  }

  Future<void> _stopMonitoring() async {
    setState(() => monitoringEnabled = false);
    _frameTimer?.cancel();
    _audioTimer?.cancel();
    await _cameraController.dispose();
    await _audioRecorder.stopRecorder();
  }

  @override
  void dispose() {
    _stopMonitoring();
    socket?.disconnect();
    _audioRecorder.closeRecorder();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Monitoring')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: monitoringEnabled ? _stopMonitoring : _startMonitoring,
              child: Text(monitoringEnabled ? 'Stop Monitoring' : 'Start Monitoring'),
            ),
            const SizedBox(height: 20),
            if (monitoringEnabled && _cameraController.value.isInitialized)
              AspectRatio(
                aspectRatio: _cameraController.value.aspectRatio,
                child: CameraPreview(_cameraController),
              ),
          ],
        ),
      ),
    );
  }
}
