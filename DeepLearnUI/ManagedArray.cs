namespace DeepLearnCS
{
    unsafe public class ManagedIntList
    {
        int* Data = null;

        public int x;

        public ManagedIntList(int size)
        {
            x = size;

            Data = MemOps.IntList(size);
        }

        // 1D arrays
        public int this[int ix]
        {
            get
            {
                return Data[ix];
            }

            set
            {
                Data[ix] = value;
            }
        }

        public int Length()
        {
            return x;
        }

        public void Free()
        {
            MemOps.Free(Data);

            x = 0;

            Data = null;
        }
    }

    unsafe public class ManagedArray
    {
        double* Data = null;

        public int x;
        public int y;
        public int z;
        public int i;
        public int j;

        public ManagedArray()
        {

        }

        public ManagedArray(ManagedArray a, bool initialize = true)
        {
            Resize(a, initialize);
        }

        public ManagedArray(int size, bool initialize = true)
        {
            Resize(size, initialize);
        }

        public ManagedArray(int sizex, int sizey, bool initialize = true)
        {
            Resize(sizex, sizey, initialize);
        }

        public ManagedArray(int sizex, int sizey, int sizez, bool initialize = true)
        {
            Resize(sizex, sizey, sizez, initialize);
        }

        // For 4D arrays of type: [i][j] of [x][y] and [i] of [x][y][z]
        public ManagedArray(int sizex, int sizey, int sizez, int sizei, int sizej, bool initialize = true)
        {
            Resize(sizex, sizey, sizez, sizei, sizej, initialize);
        }

        public double this[int ix]
        {
            get
            {
                return Data[ix];
            }

            set
            {
                Data[ix] = value;
            }
        }

        public double this[int ix, int iy]
        {
            get
            {
                return Data[iy * x + ix];
            }

            set
            {
                Data[iy * x + ix] = value;
            }
        }

        public double this[int ix, int iy, int iz]
        {
            get
            {
                return Data[(iz * y + iy) * x + ix];
            }

            set
            {
                Data[(iz * y + iy) * x + ix] = value;
            }
        }

        public void Resize(int size, bool initialize = true)
        {
            MemOps.Free(Data);

            x = size;
            y = 1;
            z = 1;
            i = 1;
            j = 1;

            Data = MemOps.New(size, initialize);
        }

        public void Resize(int sizex, int sizey, bool initialize = true)
        {
            MemOps.Free(Data);

            x = sizex;
            y = sizey;
            z = 1;
            i = 1;
            j = 1;

            Data = MemOps.New(x, y, initialize);
        }

        public void Resize(int sizex, int sizey, int sizez, bool initialize = true)
        {
            MemOps.Free(Data);

            x = sizex;
            y = sizey;
            z = sizez;
            i = 1;
            j = 1;

            Data = MemOps.New(x * y * z, initialize);
        }

        // For 4D arrays of type: [i][j] of [x][y] and [i] of [x][y][z]
        public void Resize(int sizex, int sizey, int sizez, int sizei, int sizej, bool initialize = true)
        {
            MemOps.Free(Data);

            x = sizex;
            y = sizey;
            z = sizez;
            i = sizei;
            j = sizej;

            Data = MemOps.New(x * y * z * i * j, initialize);
        }

        public void Resize(ManagedArray a, bool initialize = true)
        {
            Resize(a.x, a.y, a.z, a.i, a.j, initialize);
        }

        public int Length()
        {
            return x * y * z * i * j;
        }

        // Reshape without modifying data
        public void Reshape(int ix = 1, int iy = 1, int iz = 1, int ii = 1, int ij = 1)
        {
            x = ix;
            y = iy;
            z = iz;
            i = ii;
            j = ij;
        }

        public void Free()
        {
            MemOps.Free(Data);

            x = 0;
            y = 0;
            z = 0;
            i = 0;
            j = 0;

            Data = null;
        }
    }
}
