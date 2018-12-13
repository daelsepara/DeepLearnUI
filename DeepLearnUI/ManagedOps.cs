namespace DeepLearnCS
{
    public static class ManagedOps
    {
        public static void MemCopy(ManagedArray dst, int dstoffset, ManagedArray src, int srcoffset, int count)
        {
            for (var i = 0; i < count; i++)
                dst[dstoffset + i] = src[srcoffset + i];
        }

        public static void Set(ManagedArray dst, double value)
        {
            for (var i = 0; i < dst.Length(); i++)
                dst[i] = value;
        }

        // Copy 2D[minx + x][miny + y]
        public static void Copy2D(ManagedArray dst, ManagedArray src, int minx, int miny)
        {
            if (miny >= 0 & miny < src.y)
            {
                for (var y = 0; y < dst.y; y++)
                {
                    var srcoffset = (miny + y) * src.x + minx;
                    var dstoffset = y * dst.x;

                    MemCopy(dst, dstoffset, src, srcoffset, dst.x);
                }
            }
        }

        // Copy 2D[index_list[minx + x]][miny + y]
        public static void Copy2D(ManagedArray dst, ManagedArray src, int minx, int miny, ManagedIntList index_list)
        {
            if (miny >= 0 & miny < src.y)
            {
                for (var y = 0; y < dst.y; y++)
                {
                    var sx = (miny + y) * src.x;
                    var dx = y * dst.x;

                    for (var x = 0; x < dst.x; x++)
                    {
                        var xx = index_list[minx + x];

                        var srcoffset = sx + xx;
                        var dstoffset = dx + x;

                        dst[dstoffset] = src[srcoffset];
                    }
                }
            }
        }

        // Copy 2D[x][y] to 2D[minx + x][miny + y]
        public static void Copy2DOffset(ManagedArray dst, ManagedArray src, int minx, int miny)
        {
            if (miny >= 0 & miny < dst.y & src.y > 0)
            {
                for (var y = 0; y < src.y; y++)
                {
                    var dstoffset = (miny + y) * dst.x + minx;
                    var srcoffset = y * src.x;

                    MemCopy(dst, dstoffset, src, srcoffset, src.x);
                }
            }
        }

        // Copy 3D[minx + x][miny + y][minz + z]
        public static void Copy3D(ManagedArray dst, ManagedArray src, int minx, int miny, int minz)
        {
            if (minx >= 0 & minx < src.x & miny >= 0 & miny < src.y & minz >= 0 & minz < src.z)
            {
                for (var z = 0; z < dst.z; z++)
                {
                    var offsetd = z * dst.y;
                    var offsets = (minz + z) * src.y + miny;

                    for (var y = 0; y < dst.y; y++)
                    {
                        var dstoffset = (offsetd + y) * dst.x;
                        var srcoffset = (offsets + y) * src.x + minx;

                        MemCopy(dst, dstoffset, src, srcoffset, dst.x);
                    }
                }
            }
        }

        // Copy 3D[minx + x][miny + y][index_list[minz + z]]
        public static void Copy3D(ManagedArray dst, ManagedArray src, int minx, int miny, int minz, ManagedIntList index_list)
        {
            if (minx >= 0 & minx < src.x & miny >= 0 & miny < src.y & minz >= 0 & minz < src.z)
            {
                for (var z = 0; z < dst.z; z++)
                {
                    var zz = index_list[minz + z];

                    var offsets = zz * src.y + miny;
                    var offsetd = z * dst.y;

                    for (var y = 0; y < dst.y; y++)
                    {
                        var dstoffset = (offsetd + y) * dst.x;
                        var srcoffset = (offsets + y) * src.x + minx;

                        MemCopy(dst, dstoffset, src, srcoffset, dst.x);
                    }
                }
            }
        }

        // Copies a 4D [index][x][y][z] to 3D [x][y][z]
        public static void Copy4D3D(ManagedArray dst, ManagedArray src, int index)
        {
            MemCopy(dst, 0, src, index * dst.Length(), dst.Length());
        }

        // Copies a 3D [x][y][z] to 4D [index][x][y][z] with subsampling
        public static void Copy3D4D(ManagedArray dst, ManagedArray src, int index, int step)
        {
            if (dst.z == src.z)
            {
                for (var z = 0; z < dst.z; z++)
                {
                    var offsetd = index * dst.z * dst.y + z * dst.y;
                    var offsets = z * src.y;

                    for (var y = 0; y < dst.y; y++)
                    {
                        var dstoffset = (offsetd + y) * dst.x;
                        var srcoffset = (offsets + y * step) * src.x;

                        for (var x = 0; x < dst.x; x++)
                        {
                            dst[dstoffset + x] = src[srcoffset + x * step];
                        }
                    }
                }
            }
        }

        // Copies a 3D [x][y][z] to 4D [index][x][y][z] with maxpool
        public static void Pool3D4D(ManagedArray dst, ManagedArray src, int index, int step)
        {
            if (dst.z == src.z)
            {
                for (var z = 0; z < dst.z; z++)
                {
                    var offsets = z * src.y;
                    var offsetd = index * dst.z * dst.y + z * dst.y;

                    for (var y = 0; y < dst.y; y++)
                    {
                        var dstoffset = (offsetd + y) * dst.x;

                        var ys = y * step;

                        for (var x = 0; x < dst.x; x++)
                        {
                            var maxval = double.MinValue;
                            var xs = x * step;

                            for (var yy = 0; yy < step; yy++)
                            {
                                var dy = ys + yy;
                                var vstep = (offsets + dy) * src.x;

                                for (var xx = 0; xx < step; xx++)
                                {
                                    var dx = xs + xx;

                                    if (dx < src.x && dy < src.y)
                                    {
                                        var val = src[vstep + dx];

                                        if (val > maxval)
                                            maxval = val;
                                    }
                                }
                            }

                            dst[dstoffset + x] = maxval;
                        }
                    }
                }
            }
        }

        // Copies a 3D [x][y][z] to 4D [index][x][y][z]
        public static void Copy3D4D(ManagedArray dst, ManagedArray src, int index)
        {
            MemCopy(dst, index * src.Length(), src, 0, src.Length());
        }

        // Copies a 2D [x][y] to 3D [index][x][y]
        public static void Copy2D3D(ManagedArray dst, ManagedArray src, int index)
        {
            var size2D = src.x * src.y;

            if (index >= 0 & index < dst.z & src.x == dst.x & src.y == dst.y)
            {
                var dstoffset = index * size2D;

                for (var y = 0; y < src.y; y++)
                {
                    var srcoffset = y * src.x;

                    MemCopy(dst, dstoffset + srcoffset, src, srcoffset, src.x);
                }
            }
        }

        // Copies a 2D [x][y] to 4D [index][x][y][z]
        public static void Copy2D4D(ManagedArray dst, ManagedArray src, int z, int index)
        {
            var size2D = src.x * src.y;
            var size3D = size2D * dst.z;

            if (index >= 0 & src.x == dst.x & src.y == dst.y)
            {
                var dstoffset = index * size3D + z * size2D;

                for (var y = 0; y < src.x; y++)
                {
                    var srcoffset = y * src.x;

                    MemCopy(dst, srcoffset + dstoffset, src, srcoffset, src.x);
                }
            }
        }

        // Copies a 4D [index][x][y][z] to 2D [x][y] 
        public static void Copy4D2D(ManagedArray dst, ManagedArray src, int z, int index)
        {
            var size2D = dst.x * dst.y;
            var size3D = size2D * src.z;

            if (index >= 0 & src.x == dst.x & src.y == dst.y)
            {
                var srcoffset = index * size3D + z * size2D;

                for (var y = 0; y < dst.y; y++)
                {
                    var dstoffset = y * dst.x;

                    MemCopy(dst, dstoffset, src, srcoffset + dstoffset, dst.x);
                }
            }
        }

        // Copies a 4D [i][j][x][y] to a 2D [x][y] array
        public static void Copy4DIJ2D(ManagedArray dst, ManagedArray src, int i, int j)
        {
            var size2D = dst.x * dst.y;
            var srcoffset = (i * src.j + j) * size2D;

            if (j < src.j & i < src.i)
            {
                MemCopy(dst, 0, src, srcoffset, size2D);
            }
        }

        // Copies a 2D [x][y] array to a 4D [i][j][x][y] 
        public static void Copy2D4DIJ(ManagedArray dst, ManagedArray src, int i, int j)
        {
            var size2D = src.x * src.y;
            var dstoffset = (i * dst.j + j) * size2D;

            if (j >= 0 & j < dst.j & i >= 0 & i < dst.i)
            {
                MemCopy(dst, dstoffset, src, 0, size2D);
            }
        }

        // Fisher–Yates shuffle algorithm
        public static void Shuffle(ManagedIntList index_list)
        {
            System.Random random = new System.Random(System.Guid.NewGuid().GetHashCode());

            int n = index_list.Length();

            for (int i = n - 1; i > 1; i--)
            {
                int rnd = random.Next(i + 1);

                var value = index_list[rnd];

                index_list[rnd] = index_list[i];

                index_list[i] = value;
            }
        }

        public static void Free(params ManagedLayer[] trash)
        {
            foreach (var item in trash)
            {
                if (item != null)
                    item.Free();
            }
        }

        public static void Free(params ManagedArray[] trash)
        {
            foreach (var item in trash)
            {
                if (item != null)
                    item.Free();
            }
        }

        public static void Free(params ManagedIntList[] trash)
        {
            foreach (var item in trash)
            {
                if (item != null)
                    item.Free();
            }
        }
    }
}
