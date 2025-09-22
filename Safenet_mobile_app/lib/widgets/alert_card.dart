import 'package:flutter/material.dart';

class AlertCard extends StatelessWidget {
  final String title;
  final String time;

  const AlertCard({super.key, required this.title, required this.time}); // ✅ added super.key + const

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.all(10), // ✅ const
      child: ListTile(
        leading: const Icon(Icons.warning, color: Colors.red), // ✅ const
        title: Text(title),
        subtitle: Text("Time: $time"),
      ),
    );
  }
}
