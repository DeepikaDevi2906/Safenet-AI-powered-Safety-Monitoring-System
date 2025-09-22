import 'package:socket_io_client/socket_io_client.dart' as IO;

class SocketService {
  late IO.Socket socket;

  void initSocket(String backendUrl) {
    socket = IO.io(backendUrl, <String, dynamic>{
      'transports': ['websocket'],
      'autoConnect': true,
    });

    socket.onConnect((_) => print('Socket connected'));
  }

  void onAlert(Function(dynamic) callback) {
    socket.on('new_alert', callback);
  }

  void disconnect() {
    socket.disconnect();
  }
}
