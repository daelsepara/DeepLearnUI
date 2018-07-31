namespace DeepLearnCS
{
    public class ConvolutionalNeuralNetworkOptions
    {
        public double Alpha;
        public int BatchSize;
        public int Epochs;
        public int Items;
        public bool Shuffle;

        public ConvolutionalNeuralNetworkOptions(double alpha, int batchsize, int epochs, int items, bool shuffle = false)
        {
            Alpha = alpha;
            BatchSize = batchsize;
            Epochs = epochs;
            Items = items;
            Shuffle = shuffle;
        }

        public ConvolutionalNeuralNetworkOptions()
        {
            Alpha = 1.0;
            BatchSize = 50;
            Epochs = 1;
            Items = 50;
            Shuffle = false;
        }
    }
}
