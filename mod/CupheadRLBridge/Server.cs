using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace CupheadRLBridge
{
    /// <summary>
    /// Localhost TCP server speaking newline-delimited JSON. One client at a time: a second
    /// connection gets an error line and is closed. The accept thread hands the client to a
    /// reader thread that queues lines for the Unity main thread, which may block on
    /// <see cref="WaitForLine"/> (lockstep mode).
    /// </summary>
    public sealed class Server
    {
        const int SendTimeoutMs = 2000;
        const int ReconnectGraceMs = 500;
        const int RejectDrainMs = 1000;

        readonly int port;
        readonly Queue<string> inbox = new Queue<string>();
        readonly object gate = new object();
        readonly object sendGate = new object();
        TcpListener listener;
        TcpClient client;
        NetworkStream stream;
        Thread acceptThread;
        volatile bool running;

        public volatile bool Connected;
        /// <summary>Incremented on every connect and disconnect (main thread watches it).</summary>
        public volatile int Generation;
        /// <summary>States dropped because the client was not reading fast enough.</summary>
        public int Dropped;
        public event Action<string> Log;

        public Server(int port) { this.port = port; }

        public void Start()
        {
            running = true;
            listener = new TcpListener(IPAddress.Loopback, port);
            listener.Start();
            acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "CupheadRLBridge.Accept" };
            acceptThread.Start();
        }

        public void Stop()
        {
            running = false;
            try { listener.Stop(); } catch { }
            Drop();
        }

        void AcceptLoop()
        {
            while (running)
            {
                TcpClient c;
                try { c = listener.AcceptTcpClient(); }
                catch (Exception e)
                {
                    if (running && Log != null) Log("accept: " + e.Message);
                    continue;
                }
                c.NoDelay = true;
                c.SendTimeout = SendTimeoutMs;
                // A client that just closed may not have been dropped by its reader thread yet:
                // give it a moment instead of rejecting an immediate reconnect.
                var wait = System.Diagnostics.Stopwatch.StartNew();
                while (Connected && wait.ElapsedMilliseconds < ReconnectGraceMs) Thread.Sleep(10);
                if (Connected)
                {
                    // Reject instead of silently stealing the session from the first client.
                    try
                    {
                        byte[] b = Encoding.UTF8.GetBytes("{\"type\":\"error\",\"message\":\"another client is already connected to the bridge (one env per game)\"}\n");
                        c.GetStream().Write(b, 0, b.Length);
                        // Half-close and swallow the client's hello so Close() does not send an RST
                        // that would discard the error line before the client reads it.
                        c.Client.Shutdown(SocketShutdown.Send);
                        // Bounded in total time, so a client that keeps sending cannot hang this thread.
                        var sink = new byte[4096];
                        var drain = System.Diagnostics.Stopwatch.StartNew();
                        while (drain.ElapsedMilliseconds < RejectDrainMs)
                        {
                            int left = (int)(RejectDrainMs - drain.ElapsedMilliseconds);
                            if (left <= 0 || !c.Client.Poll(left * 1000, SelectMode.SelectRead)) break;
                            if (c.Client.Receive(sink) <= 0) break;
                        }
                    }
                    catch { }
                    try { c.Close(); } catch { }
                    if (Log != null) Log("rejected a second client");
                    continue;
                }
                lock (sendGate)
                {
                    client = c;
                    stream = c.GetStream();
                }
                lock (gate) { inbox.Clear(); }
                Connected = true;
                Generation++;
                if (Log != null) Log("client connected");
                var reader = new Thread(() => ReadLoop(c)) { IsBackground = true, Name = "CupheadRLBridge.Read" };
                reader.Start();
            }
        }

        void ReadLoop(TcpClient c)
        {
            try
            {
                var reader = new StreamReader(c.GetStream(), new UTF8Encoding(false));
                string line;
                while (running && (line = reader.ReadLine()) != null)
                {
                    if (line.Length == 0) continue;
                    lock (gate)
                    {
                        inbox.Enqueue(line);
                        Monitor.PulseAll(gate);
                    }
                }
            }
            catch (Exception e)
            {
                if (running && Log != null) Log("read: " + e.Message);
            }
            lock (sendGate)
            {
                if (ReferenceEquals(client, c)) Drop();
            }
            if (Log != null && running) Log("client disconnected");
        }

        void Drop()
        {
            bool was = client != null;
            try { if (client != null) client.Close(); } catch { }
            client = null;
            stream = null;
            if (was)
            {
                Connected = false;
                Generation++;
                lock (gate) { Monitor.PulseAll(gate); }
            }
        }

        public bool TryDequeue(out string line)
        {
            lock (gate)
            {
                if (inbox.Count > 0) { line = inbox.Dequeue(); return true; }
            }
            line = null;
            return false;
        }

        /// <summary>Block until a line arrives, the client disconnects, or the timeout passes.</summary>
        public bool WaitForLine(int timeoutMs, out string line)
        {
            lock (gate)
            {
                if (inbox.Count == 0 && Connected) Monitor.Wait(gate, timeoutMs);
                if (inbox.Count > 0) { line = inbox.Dequeue(); return true; }
            }
            line = null;
            return false;
        }

        /// <summary>
        /// Send one line. Droppable lines (streamed states) are skipped when the socket's send
        /// buffer has no room, so a slow client never blocks the game's main thread; other lines
        /// block for at most <see cref="SendTimeoutMs"/> before the client is dropped.
        /// </summary>
        public void Send(string line, bool droppable = false)
        {
            lock (sendGate)
            {
                if (stream == null) return;
                try
                {
                    if (droppable && !client.Client.Poll(0, SelectMode.SelectWrite))
                    {
                        Dropped++;
                        return;
                    }
                    byte[] b = Encoding.UTF8.GetBytes(line + "\n");
                    stream.Write(b, 0, b.Length);
                }
                catch (Exception e)
                {
                    if (Log != null) Log("send failed: " + e.Message);
                    Drop();
                }
            }
        }
    }
}
