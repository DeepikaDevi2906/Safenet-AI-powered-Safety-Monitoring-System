import 'package:flutter/material.dart';

class CameraService extends StatefulWidget {
  const CameraService({super.key}); // ✅ fixed key

  @override
  State<CameraService> createState() => _CameraServiceState();
}

class _CameraServiceState extends State<CameraService> {
  @override
  void initState() {
    super.initState();
    debugPrint("Camera Service initialized ✅"); // ✅ replaced print
  }

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Text("Camera Service Running"),
    );
  }
}
