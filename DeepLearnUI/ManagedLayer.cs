namespace DeepLearnCS
{
    public class ManagedLayer
    {
        public LayerTypes Type;

        public int OutputMaps;
        public int Scale;
        public int KernelSize;

        public ManagedArray FeatureMap = null;     // FeatureMap[i][j][x][y]
        public ManagedArray DeltaFeatureMap = null;

        public ManagedArray Activation = null; // Activation[i][x][y][z]
        public ManagedArray Delta = null;

        // 1D
        public ManagedArray Bias = null;
        public ManagedArray DeltaBias = null;

        public ManagedLayer()
        {
            Type = LayerTypes.Input;
        }

        public ManagedLayer(int scale)
        {
            Type = LayerTypes.Subsampling;
            Scale = scale;
        }

        public ManagedLayer(int outputMaps, int kernelSize)
        {
            Type = LayerTypes.Convolution;
            KernelSize = kernelSize;
            OutputMaps = outputMaps;
        }

        public void Free()
        {
            ManagedOps.Free(FeatureMap, DeltaFeatureMap);
            ManagedOps.Free(Activation, Delta);
            ManagedOps.Free(Bias, DeltaBias);
        }
    }
}
