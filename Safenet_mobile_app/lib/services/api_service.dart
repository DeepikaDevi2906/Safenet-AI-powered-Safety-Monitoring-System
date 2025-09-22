import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // 🔹 Change this to your Flask backend address (use your LAN/Wi-Fi IP when testing on phone)
  static const String baseUrl = "http://127.0.0.1:5000";

  /// LOGIN
  static Future<Map<String, dynamic>> login(
      String email, String password) async {
    final url = Uri.parse("$baseUrl/login");
    final response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "email": email,
        "password": password,
      }),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Login failed: ${response.body}");
    }
  }

  /// REGISTER
  static Future<Map<String, dynamic>> register(
      String name, String email, String password) async {
    final url = Uri.parse("$baseUrl/register");
    final response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "name": name,
        "email": email,
        "password": password,
      }),
    );

    if (response.statusCode == 201 || response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Registration failed: ${response.body}");
    }
  }

  /// FETCH ALERTS
  static Future<List<dynamic>> getAlerts() async {
    final url = Uri.parse("$baseUrl/alerts");
    final response = await http.get(url);

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Failed to load alerts");
    }
  }

  /// SEND ALERT (general)
  static Future<Map<String, dynamic>> sendAlert(String message) async {
    final url = Uri.parse("$baseUrl/alerts");
    final response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"message": message}),
    );

    if (response.statusCode == 201) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Failed to send alert");
    }
  }

  /// GET LIVE MONITOR STATUS
  static Future<Map<String, dynamic>> getLiveMonitorStatus() async {
    final url = Uri.parse("$baseUrl/live_status");
    final response = await http.get(url);

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Failed to fetch live status");
    }
  }

  /// SEND SOS ALERT
  static Future<Map<String, dynamic>> sendSOS({
    required String userId,
    required String message,
    required String location,
  }) async {
    final response = await http.post(
      Uri.parse("$baseUrl/sos"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "user_id": userId,
        "message": message,
        "location": location,
      }),
    );

    if (response.statusCode == 201 || response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Failed to send SOS: ${response.body}");
    }
  }
}
