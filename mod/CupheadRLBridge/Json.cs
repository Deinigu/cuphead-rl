using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace CupheadRLBridge
{
    /// <summary>Tiny JSON reader/writer (the game ships no JSON library for .NET 3.5).</summary>
    public static class Json
    {
        // ---------------------------------------------------------------- parsing
        public static Dictionary<string, object> ParseObject(string s)
        {
            int i = 0;
            object v = ParseValue(s, ref i);
            return v as Dictionary<string, object> ?? new Dictionary<string, object>();
        }

        static void Ws(string s, ref int i)
        {
            while (i < s.Length && char.IsWhiteSpace(s[i])) i++;
        }

        static object ParseValue(string s, ref int i)
        {
            Ws(s, ref i);
            if (i >= s.Length) throw new FormatException("unexpected end of JSON");
            char c = s[i];
            if (c == '{')
            {
                var d = new Dictionary<string, object>();
                i++;
                Ws(s, ref i);
                if (s[i] == '}') { i++; return d; }
                while (true)
                {
                    Ws(s, ref i);
                    string k = ParseString(s, ref i);
                    Ws(s, ref i);
                    if (s[i] != ':') throw new FormatException("expected ':'");
                    i++;
                    d[k] = ParseValue(s, ref i);
                    Ws(s, ref i);
                    if (s[i] == ',') { i++; continue; }
                    if (s[i] == '}') { i++; return d; }
                    throw new FormatException("expected ',' or '}'");
                }
            }
            if (c == '[')
            {
                var l = new List<object>();
                i++;
                Ws(s, ref i);
                if (s[i] == ']') { i++; return l; }
                while (true)
                {
                    l.Add(ParseValue(s, ref i));
                    Ws(s, ref i);
                    if (s[i] == ',') { i++; continue; }
                    if (s[i] == ']') { i++; return l; }
                    throw new FormatException("expected ',' or ']'");
                }
            }
            if (c == '"') return ParseString(s, ref i);
            if (s.Length - i >= 4 && string.CompareOrdinal(s, i, "true", 0, 4) == 0) { i += 4; return true; }
            if (s.Length - i >= 5 && string.CompareOrdinal(s, i, "false", 0, 5) == 0) { i += 5; return false; }
            if (s.Length - i >= 4 && string.CompareOrdinal(s, i, "null", 0, 4) == 0) { i += 4; return null; }
            int start = i;
            while (i < s.Length && "+-0123456789.eE".IndexOf(s[i]) >= 0) i++;
            return double.Parse(s.Substring(start, i - start), CultureInfo.InvariantCulture);
        }

        static string ParseString(string s, ref int i)
        {
            if (s[i] != '"') throw new FormatException("expected string");
            i++;
            var sb = new StringBuilder();
            while (s[i] != '"')
            {
                char c = s[i++];
                if (c == '\\')
                {
                    char e = s[i++];
                    switch (e)
                    {
                        case 'n': sb.Append('\n'); break;
                        case 't': sb.Append('\t'); break;
                        case 'r': sb.Append('\r'); break;
                        case 'u': sb.Append((char)Convert.ToInt32(s.Substring(i, 4), 16)); i += 4; break;
                        default: sb.Append(e); break;
                    }
                }
                else sb.Append(c);
            }
            i++;
            return sb.ToString();
        }

        public static double Num(Dictionary<string, object> d, string k, double dflt)
        {
            object v;
            if (!d.TryGetValue(k, out v) || v == null) return dflt;
            if (v is bool) return (bool)v ? 1 : 0;
            if (v is double) return (double)v;
            return dflt;
        }

        public static bool Bool(Dictionary<string, object> d, string k, bool dflt)
        {
            object v;
            if (!d.TryGetValue(k, out v) || v == null) return dflt;
            if (v is bool) return (bool)v;
            if (v is double) return (double)v != 0;
            return dflt;
        }

        public static string Str(Dictionary<string, object> d, string k, string dflt)
        {
            object v;
            return d.TryGetValue(k, out v) && v is string ? (string)v : dflt;
        }
    }

    /// <summary>Allocation-light JSON object builder.</summary>
    public sealed class JsonWriter
    {
        readonly StringBuilder sb = new StringBuilder(2048);
        bool first = true;

        public JsonWriter Begin() { sb.Append('{'); first = true; return this; }
        public JsonWriter End() { sb.Append('}'); first = false; return this; }
        public JsonWriter BeginArray(string k) { Key(k); sb.Append('['); first = true; return this; }
        public JsonWriter EndArray() { sb.Append(']'); first = false; return this; }
        public JsonWriter BeginObject(string k) { Key(k); return Begin(); }

        public JsonWriter BeginElement()
        {
            if (!first) sb.Append(',');
            return Begin();
        }

        void Key(string k)
        {
            if (!first) sb.Append(',');
            first = false;
            if (k != null) { sb.Append('"').Append(k).Append("\":"); }
        }

        public JsonWriter F(string k, float v)
        {
            Key(k);
            if (float.IsNaN(v) || float.IsInfinity(v)) sb.Append("null");
            else sb.Append(v.ToString("0.###", CultureInfo.InvariantCulture));
            return this;
        }

        public JsonWriter I(string k, int v) { Key(k); sb.Append(v.ToString(CultureInfo.InvariantCulture)); return this; }
        public JsonWriter L(string k, long v) { Key(k); sb.Append(v.ToString(CultureInfo.InvariantCulture)); return this; }
        public JsonWriter B(string k, bool v) { Key(k); sb.Append(v ? "true" : "false"); return this; }

        public JsonWriter S(string k, string v)
        {
            Key(k);
            if (v == null) { sb.Append("null"); return this; }
            sb.Append('"');
            foreach (char c in v)
            {
                if (c == '"' || c == '\\') sb.Append('\\').Append(c);
                else if (c < ' ') sb.Append("\\u").Append(((int)c).ToString("x4"));
                else sb.Append(c);
            }
            sb.Append('"');
            return this;
        }

        public override string ToString() { return sb.ToString(); }
    }
}
