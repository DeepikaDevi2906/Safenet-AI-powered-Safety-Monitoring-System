// lib/screens/dashboard_screen.dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:geolocator/geolocator.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'camera_screen.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  String location = "Unknown";

  CameraController? _cameraController;
  FlutterSoundRecorder? _audioRecorder;
  bool _isCameraReady = false;
  bool _isMicReady = false;
  bool _isLocationReady = false;

  final FlutterLocalNotificationsPlugin _notifications =
      FlutterLocalNotificationsPlugin();
  Timer? _frameTimer;

  @override
  void initState() {
    super.initState();
    _initPermissionsAndDevices();
    _initNotifications();
  }

  Future<void> _initPermissionsAndDevices() async {
    final cameraStatus = await Permission.camera.request();
    final micStatus = await Permission.microphone.request();
    final locationStatus = await Permission.locationWhenInUse.request();

    if (cameraStatus.isGranted) {
      final cameras = await availableCameras();
      if (cameras.isNotEmpty) {
        _cameraController =
            CameraController(cameras.first, ResolutionPreset.medium);
        await _cameraController!.initialize();
        setState(() => _isCameraReady = true);
      }
    }

    if (micStatus.isGranted) {
      _audioRecorder = FlutterSoundRecorder();
      await _audioRecorder!.openRecorder();
      setState(() => _isMicReady = true);
    }

    if (locationStatus.isGranted) {
      final position = await Geolocator.getCurrentPosition(
        locationSettings:
            const LocationSettings(accuracy: LocationAccuracy.best),
      );
      setState(() {
        location = "${position.latitude}, ${position.longitude}";
        _isLocationReady = true;
      });
    }
  }

  void _initNotifications() {
    const AndroidInitializationSettings androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    const InitializationSettings initSettings =
        InitializationSettings(android: androidSettings);
    _notifications.initialize(initSettings);
  }

  void _showNotification(String title, String body) async {
    const AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails('alert_channel', 'Alerts',
            importance: Importance.max, priority: Priority.high);
    const NotificationDetails platformDetails =
        NotificationDetails(android: androidDetails);
    await _notifications.show(0, title, body, platformDetails);
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _audioRecorder?.closeRecorder();
    _frameTimer?.cancel();
    super.dispose();
  }

  Future<void> navigateToCamera() async {
    if (!_isCameraReady) return;
    if (!mounted) return;

    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => CameraScreen(controller: _cameraController!),
      ),
    );
  }

  Widget buildAlertBox() {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.red.withAlpha(128),
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Text(
        "🚨 ALERT: Something unusual detected!",
        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
      ),
    );
  }

  Widget buildSOSButton() {
    return ElevatedButton(
      onPressed: () async {
        // Get location
        if (_isLocationReady) {
          final position = await Geolocator.getCurrentPosition(
            locationSettings:
                const LocationSettings(accuracy: LocationAccuracy.best),
          );
          setState(() {
            location = "${position.latitude}, ${position.longitude}";
          });
          debugPrint("SOS triggered at: $location");
        }

        // Start audio recording
        if (_isMicReady) {
          await _audioRecorder!.startRecorder(toFile: 'sos_audio.aac');
        }

        // Open camera
        if (_isCameraReady && mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => CameraScreen(controller: _cameraController!),
            ),
          );
        }

        // Show notification
        _showNotification("SOS Alert", "🚨 SOS triggered! Check app for details");

        // Show snackbar
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
                content: Text(
                    "🚨 SOS Alert Triggered! Location & recording started")),
          );
        }
      },
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.red,
        minimumSize: const Size(200, 60),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
      child: const Text(
        "🚨 SOS",
        style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("SafeNet Dashboard")),
      body: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(height: 20),
            Text("Location: $location"),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isLocationReady
                  ? () async {
                      final position = await Geolocator.getCurrentPosition(
                        locationSettings:
                            const LocationSettings(accuracy: LocationAccuracy.best),
                      );
                      if (!mounted) return;
                      setState(() {
                        location = "${position.latitude}, ${position.longitude}";
                      });
                    }
                  : null,
              child: const Text("Get Location"),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isCameraReady ? navigateToCamera : null,
              child: const Text("Open Camera"),
            ),
            const SizedBox(height: 20),
            _isMicReady
                ? ElevatedButton(
                    onPressed: () async {
                      await _audioRecorder!.startRecorder(toFile: 'audio.aac');
                      if (!mounted) return;
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text("Recording started")),
                      );
                    },
                    child: const Text("Start Audio Recording"),
                  )
                : const Text("Microphone not available"),
            const SizedBox(height: 20),
            buildSOSButton(),
            const SizedBox(height: 20),
            buildAlertBox(),
          ],
        ),
      ),
    );
  }
}
