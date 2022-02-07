using System;
using UnityEngine;
//using System.Runtime.Serialization.Formatters.Binary;
#if WINDOWS_UWP
using Windows.Networking.Sockets;
using Windows.Storage.Streams;
#endif

public class TCPClient : MonoBehaviour
{
    #region Unity Functions

    private void Awake()
    {
        ConnectionStatusLED.material.color = Color.red;
        #if WINDOWS_UWP
                StartCoonection();
        #endif
    }
    private void OnApplicationFocus(bool focus)
    {
        if (!focus)
        {
#if WINDOWS_UWP
            StopCoonection();
#endif
        }
    }
    #endregion // Unity Functions

    [SerializeField]
    string hostIPAddress, port;

    public Renderer ConnectionStatusLED;
    private bool connected = false;
    public bool Connected
    {
        get { return connected; }
    }

#if WINDOWS_UWP
    StreamSocket socket = null;
    public DataWriter dw;
    public DataReader dr;
    private async void StartCoonection()
    {
        if (socket != null) socket.Dispose();

        try
        {
            socket = new StreamSocket();
            var hostName = new Windows.Networking.HostName(hostIPAddress);
            await socket.ConnectAsync(hostName, port);
            dw = new DataWriter(socket.OutputStream);
            dr = new DataReader(socket.InputStream);
            dr.InputStreamOptions = InputStreamOptions.Partial;
            connected = true;
            ConnectionStatusLED.material.color = Color.green;
        }
        catch (Exception ex)
        {
            SocketErrorStatus webErrorStatus = SocketError.GetStatus(ex.GetBaseException().HResult);
            Debug.Log(webErrorStatus.ToString() != "Unknown" ? webErrorStatus.ToString() : ex.Message);
        }
    }

    private void StopCoonection()
    {
        dw?.DetachStream();
        dw?.Dispose();
        dw = null;

        dr?.DetachStream();
        dr?.Dispose();
        dr = null;

        socket?.Dispose();
        connected = false;
    }

    bool lastMessageSent = true;

    public async void SendDepthAsync(ushort[] depth, ushort[] ab, long ts, string depthTransform)
    {
        if (!lastMessageSent) return;
        lastMessageSent = false;
        try
        {
            // Write header
            dw.WriteString("d"); // header "s" stands for it is ushort array (uint16)

            // Write Length
            dw.WriteInt32(depthTransform.Length * sizeof(Char) / 2);
            dw.WriteInt32(depth.Length + ab.Length);

            // Write actual data
            dw.WriteBytes(UINT16ToBytes(depth));
            dw.WriteBytes(UINT16ToBytes(ab));
            dw.WriteInt64(ts);
            dw.WriteString(depthTransform);

            // Send out
            await dw.StoreAsync();
            await dw.FlushAsync();
        }
        catch (Exception ex)
        {
            SocketErrorStatus webErrorStatus = SocketError.GetStatus(ex.GetBaseException().HResult);
            Debug.Log(webErrorStatus.ToString() != "Unknown" ? webErrorStatus.ToString() : ex.Message);
        }
        lastMessageSent = true;
    }

    public async void SendLeftImageAsync(byte[] LFImage, long ts_left) // not in use
    {
        if (!lastMessageSent) return;
        lastMessageSent = false;
        try
        {
            // Write header
            dw.WriteString("l"); // header "f"

            // Write Length
            dw.WriteInt32(LFImage.Length);
            dw.WriteInt64(ts_left);

            // Write actual data
            dw.WriteBytes(LFImage);

            // Send out
            await dw.StoreAsync();
            await dw.FlushAsync();
        }
        catch (Exception ex)
        {
            SocketErrorStatus webErrorStatus = SocketError.GetStatus(ex.GetBaseException().HResult);
            Debug.Log(webErrorStatus.ToString() != "Unknown" ? webErrorStatus.ToString() : ex.Message);
        }
        lastMessageSent = true;
    }

    public async void SendSpatialImageAsync(byte[] LFImage, byte[] RFImage, long ts_left, long ts_right) // not in use
    {
        if (!lastMessageSent) return;
        lastMessageSent = false;
        try
        {
            // Write header
            dw.WriteString("v"); // header "f"

            // Write Length
            dw.WriteInt32(LFImage.Length + RFImage.Length);
            dw.WriteInt64(ts_left);
            dw.WriteInt64(ts_right);

            // Write actual data
            dw.WriteBytes(LFImage);
            dw.WriteBytes(RFImage);

            // Send out
            await dw.StoreAsync();
            await dw.FlushAsync();
        }
        catch (Exception ex)
        {
            SocketErrorStatus webErrorStatus = SocketError.GetStatus(ex.GetBaseException().HResult);
            Debug.Log(webErrorStatus.ToString() != "Unknown" ? webErrorStatus.ToString() : ex.Message);
        }
        lastMessageSent = true;
    }


    public async void SendSpatialImageAsync(byte[] LRFImage, long ts_left, long ts_right, string leftTransform, string rightTransform)
    {
        if (!lastMessageSent) return;
        lastMessageSent = false;
        try
        {
            // Write header
            dw.WriteString("v"); // header "f"

            // Write Timestamp and Length
            dw.WriteInt32((leftTransform.Length + rightTransform.Length) * sizeof(Char) / 2);
            dw.WriteInt32(LRFImage.Length);
            dw.WriteInt64(ts_left);
            dw.WriteInt64(ts_right);

            // Write actual data
            dw.WriteBytes(LRFImage);
            dw.WriteString(leftTransform);
            dw.WriteString(rightTransform);

            // Send out
            await dw.StoreAsync();
            await dw.FlushAsync();
        }
        catch (Exception ex)
        {
            SocketErrorStatus webErrorStatus = SocketError.GetStatus(ex.GetBaseException().HResult);
            Debug.Log(webErrorStatus.ToString() != "Unknown" ? webErrorStatus.ToString() : ex.Message);
        }
        lastMessageSent = true;
    }

#endif


    #region Helper Function
    byte[] UINT16ToBytes(ushort[] data)
    {
        byte[] ushortInBytes = new byte[data.Length * sizeof(ushort)];
        System.Buffer.BlockCopy(data, 0, ushortInBytes, 0, ushortInBytes.Length);
        return ushortInBytes;
    }
    #endregion

    #region Button Callback
    public void ConnectToServerEvent()
    {
#if WINDOWS_UWP
        if (!connected) StartCoonection();
        else StopCoonection();
#endif
    }
    #endregion
}
