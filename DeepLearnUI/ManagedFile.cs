using System;
using System.Globalization;
using System.IO;

namespace DeepLearnCS
{
    public static class ManagedFile
    {
        public static CultureInfo ci = new CultureInfo("en-US");

        public static void Load1D(string filename, ManagedArray A, char delimiter = ',')
        {
            if (File.Exists(filename))
            {
                var lines = File.ReadAllLines(filename);

                if (lines.Length > 0)
                {
                    var tokens = lines[0].Split(delimiter);

                    for (int x = 0; x < A.Length(); x++)
                    {
                        A[x] = Convert.ToDouble(tokens[x], ci);
                    }
                }
            }
        }

        public static void Save1D(string filename, ManagedArray A, char delimiter = ',')
        {
            using (var file = new StreamWriter(filename, false))
            {
                for (int x = 0; x < A.x; x++)
                {
                    file.Write(A[x].ToString(ci));

                    if (x < A.x - 1)
                    {
                        file.Write(delimiter);
                    }
                }

                file.WriteLine();
            }
        }

        public static void Load1DY(string filename, ManagedArray A)
        {
            if (File.Exists(filename))
            {
                var lines = File.ReadAllLines(filename);

                for (int y = 0; y < A.Length(); y++)
                {
                    A[y] = Convert.ToDouble(lines[y], ci);
                }
            }
        }

        public static void Save1DY(string filename, ManagedArray A)
        {
            using (var file = new StreamWriter(filename, false))
            {
                for (int y = 0; y < A.Length(); y++)
                {
                    file.WriteLine(A[y].ToString(ci));
                }
            }
        }

        public static void Load2D(string filename, ManagedArray A, char delimiter = ',')
        {
            if (File.Exists(filename))
            {
                var temp = new ManagedArray(A.x, A.y);

                var lines = File.ReadAllLines(filename);

                for (int y = 0; y < A.y; y++)
                {
                    if (y < lines.Length)
                    {
                        var tokens = lines[y].Split(delimiter);

                        for (int x = 0; x < A.x; x++)
                        {
                            temp[x, y] = Convert.ToDouble(tokens[x], ci);
                        }
                    }
                }

                ManagedOps.Copy2D(A, temp, 0, 0);

                ManagedOps.Free(temp);
            }
        }

        public static void Load2DV2(string filename, ManagedArray A, char delimiter = ',')
        {
            if (File.Exists(filename))
            {
                var temp = new ManagedArray(A.x, A.y);

                using (TextReader reader = File.OpenText(filename))
                {
                    for (int y = 0; y < A.y; y++)
                    {
                        var line = reader.ReadLine();

                        if (line != null)
                        {
                            var tokens = line.Split(delimiter);

                            for (int x = 0; x < A.x; x++)
                            {
                                temp[x, y] = Convert.ToDouble(tokens[x], ci);
                            }
                        }
                    }
                }

                ManagedOps.Copy2D(A, temp, 0, 0);

                ManagedOps.Free(temp);
            }
        }

        public static void Save2D(string filename, ManagedArray A, char delimiter = ',')
        {
            using (var file = new StreamWriter(filename, false))
            {
                for (int y = 0; y < A.y; y++)
                {
                    if (y > 0)
                        file.WriteLine();

                    for (int x = 0; x < A.x; x++)
                    {
                        file.Write(A[x, y].ToString(ci));

                        if (x < A.x - 1)
                        {
                            file.Write(delimiter);
                        }
                    }
                }
            }
        }

        public static void Load2D4D(string filename, ManagedArray A, int i, int j, char delimiter = ',')
        {
            if (File.Exists(filename))
            {
                var temp = new ManagedArray(A.x, A.y);

                var lines = File.ReadAllLines(filename);

                for (int y = 0; y < A.y; y++)
                {
                    if (y < lines.Length)
                    {
                        var tokens = lines[y].Split(delimiter);

                        for (int x = 0; x < A.x; x++)
                        {
                            temp[x, y] = Convert.ToDouble(tokens[x], ci);
                        }
                    }
                }

                ManagedOps.Copy2D4DIJ(A, temp, i, j);

                ManagedOps.Free(temp);
            }
        }

        public static void Save2D4D(string filename, ManagedArray A, int i, int j, char delimiter = ',')
        {
            using (var file = new StreamWriter(filename, false))
            {
                var temp = new ManagedArray(A.x, A.y);

                ManagedOps.Copy4DIJ2D(temp, A, i, j);

                for (int y = 0; y < A.y; y++)
                {
                    for (int x = 0; x < A.x; x++)
                    {
                        file.Write("{0}", temp[x, y].ToString(ci));

                        if (x < A.x - 1)
                        {
                            file.Write(delimiter);
                        }
                    }

                    file.WriteLine();
                }

                ManagedOps.Free(temp);
            }
        }

        public static void Load3D(string filename, ManagedArray A, char delimiter = ',')
        {
            if (File.Exists(filename))
            {
                var lines = File.ReadAllLines(filename);

                for (int y = 0; y < A.y; y++)
                {
                    if (y < lines.Length)
                    {
                        var tokens = lines[y].Split(delimiter);

                        for (int z = 0; z < A.z; z++)
                        {
                            for (int x = 0; x < A.x; x++)
                            {
                                A[x, y, z] = Convert.ToDouble(tokens[z * A.x + x], ci);
                            }
                        }
                    }
                }
            }
        }

        public static void Load3DV2(string filename, ManagedArray A, char delimiter = ',')
        {
            if (File.Exists(filename))
            {
                using (TextReader reader = File.OpenText(filename))
                {
                    for (int y = 0; y < A.y; y++)
                    {
                        var line = reader.ReadLine();

                        if (line != null)
                        {
                            var tokens = line.Split(delimiter);

                            for (int z = 0; z < A.z; z++)
                            {
                                for (int x = 0; x < A.x; x++)
                                {
                                    A[x, y, z] = Convert.ToDouble(tokens[z * A.x + x], ci);
                                }
                            }
                        }
                    }
                }
            }
        }

        public static void Save3D(string filename, ManagedArray A, char delimiter = ',')
        {
            using (var file = new StreamWriter(filename, false))
            {
                for (int y = 0; y < A.y; y++)
                {
                    for (int z = 0; z < A.z; z++)
                    {
                        for (int x = 0; x < A.x; x++)
                        {
                            file.Write(A[x, y, z].ToString(ci));

                            if (z < A.z - 1 || x < A.x - 1)
                            {
                                file.Write(delimiter);
                            }
                        }
                    }

                    file.WriteLine();
                }
            }
        }

        public static void Load3D2D(string filename, ManagedArray A, char delimiter = ',')
        {
            if (File.Exists(filename))
            {
                var xx = A.x;
                var yy = A.y;
                var zz = A.z;
                var size2D = xx * yy;

                A.Reshape(xx * yy, zz);

                var lines = File.ReadAllLines(filename);

                for (int y = 0; y < yy; y++)
                {
                    var xoffset = y * xx;

                    if (y < lines.Length)
                    {
                        var tokens = lines[y].Split(delimiter);

                        for (int z = 0; z < zz; z++)
                        {
                            var yoffset = z * size2D;

                            for (int x = 0; x < xx; x++)
                            {
                                A[xoffset + x, yoffset] = Convert.ToDouble(tokens[z * xx + x], ci);
                            }
                        }
                    }
                }
            }
        }

        public static void Load3D2DV2(string filename, ManagedArray A, char delimiter = ',')
        {
            if (File.Exists(filename))
            {
                var xx = A.x;
                var yy = A.y;
                var zz = A.z;
                var size2D = xx * yy;

                A.Reshape(xx * yy, zz);

                using (TextReader reader = File.OpenText(filename))
                {
                    for (int y = 0; y < yy; y++)
                    {
                        var line = reader.ReadLine();

                        if (line != null)
                        {
                            var xoffset = y * xx;

                            var tokens = line.Split(delimiter);

                            for (int z = 0; z < zz; z++)
                            {
                                var yoffset = z * size2D;

                                for (int x = 0; x < xx; x++)
                                {
                                    A[xoffset + x, yoffset] = Convert.ToDouble(tokens[z * xx + x], ci);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
