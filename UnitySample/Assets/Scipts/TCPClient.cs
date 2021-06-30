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

#if WINDOWS_UWP
    StreamSocket socket = null;
    public DataWriter dw;
    public DataReader dr;
    private async void StartCoonection()
    {
        try
        {
            if (socket == null)
            {
                socket = new StreamSocket();
                var hostName = new Windows.Networking.HostName(hostIPAddress);
                await socket.ConnectAsync(hostName, port);
            }
            if (dw == null)
            {
                dw = new DataWriter(socket.OutputStream);
            }
            if (dr == null)
            {
                dr = new DataReader(socket.InputStream);
                dr.InputStreamOptions = InputStreamOptions.Partial;
            }
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
    }

    bool lastMessageSent = true;
    public async void SendUINT16Async(ushort[] data)
    {
        if (!lastMessageSent) return;
        lastMessageSent = false;
        try
        {
            // Write header
            dw.WriteString("s"); // header "s" stands for it is ushort array (uint16)
            
            // Write length and data
            dw.WriteInt32(data.Length);
            dw.WriteBytes(UINT16ToBytes(data));
            
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

    public async void SendDepthAsync(ushort[] data1, ushort[] data2, ulong data3)
    {
        if (!lastMessageSent) return;
        lastMessageSent = false;
        try
        {
            // Write header
            dw.WriteString("d"); // header "d" stands for depth packet

            // Write Length
            dw.WriteInt32(data1.Length + data2.Length);

            // Write actual data
            dw.WriteBytes(UINT16ToBytes(data1));
            dw.WriteBytes(UINT16ToBytes(data2));
            dw.WriteUInt64(data3);

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

    public async void SendTempAsync(int tempVal, ulong tempTime)
    {
        if (!lastMessageSent) return;
        lastMessageSent = false;
        try
        {
            // Write header
            dw.WriteString("t"); // header "t" stands for temperature packet

            // Write actual data
            dw.WriteInt32(tempVal);
            dw.WriteUInt64(tempTime);

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
}
