import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class CameraScreen extends StatelessWidget {
  final CameraController controller;

  const CameraScreen({super.key, required this.controller});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Camera")),
      body: controller.value.isInitialized
          ? CameraPreview(controller) // ✅ Shows live camera feed
          : const Center(
              child: CircularProgressIndicator(), // Loading indicator
            ),
    );
  }
}
