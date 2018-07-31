﻿using DeepLearnCS;
using System;
using System.Drawing;
using System.Windows.Forms;

namespace DeepLearnUI
{
    public partial class DeepLearnUI : Form
    {
        int Classification = 0;
        double[] Probability = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

        ManagedCNN cnn;
        Bitmap Digit;

        bool IsDrawing = false;
        bool IsErasing = false;
        bool IsActivated = false;

        OpenFileDialog OpenImageDialog;

        string BaseDirectory = "./";

        public DeepLearnUI()
        {
            InitializeComponent();

            InitializeNetwork();

            InitializeUI();
        }

        void InitializeUI()
        {
            Digit = new Bitmap(DigitBox.Width, DigitBox.Height);

            CopyBitmap(Digit);

            CopyClassification();

            OpenImageDialog = new OpenFileDialog();
            OpenImageDialog.InitialDirectory = BaseDirectory;
            OpenImageDialog.Filter = "png files (*.png)|*.png|jpg files (*.jpg)|*.jpg|bmp files (*.bmp)|*.bmp|All files (*.*)|*.*";
            OpenImageDialog.FilterIndex = 1;

            for (int i = 0; i < cnn.Layers.Count; i++)
            {
                if (cnn.Layers[i].Type == LayerTypes.Input)
                {
                    NetworkLayers.Items.Add(String.Format("{0} Input", i));
                }
                else if (cnn.Layers[i].Type == LayerTypes.Convolution)
                {
                    NetworkLayers.Items.Add(String.Format("{0} Convolution", i));
                }
                else if (cnn.Layers[i].Type == LayerTypes.Subsampling)
                {
                    NetworkLayers.Items.Add(String.Format("{0} Subsampling / Pooling", i));
                }
            }

            Probability0.ReadOnly = true;
            Probability1.ReadOnly = true;
            Probability2.ReadOnly = true;
            Probability3.ReadOnly = true;
            Probability4.ReadOnly = true;
            Probability5.ReadOnly = true;
            Probability6.ReadOnly = true;
            Probability7.ReadOnly = true;
            Probability8.ReadOnly = true;
            Probability9.ReadOnly = true;

            FeatureMapPanel.Hide();
        }

        void CopyClassification()
        {
            Probability0.Text = Probability[0].ToString("g", ManagedMatrix.ci);
            Probability1.Text = Probability[1].ToString("g", ManagedMatrix.ci);
            Probability2.Text = Probability[2].ToString("g", ManagedMatrix.ci);
            Probability3.Text = Probability[3].ToString("g", ManagedMatrix.ci);
            Probability4.Text = Probability[4].ToString("g", ManagedMatrix.ci);
            Probability5.Text = Probability[5].ToString("g", ManagedMatrix.ci);
            Probability6.Text = Probability[6].ToString("g", ManagedMatrix.ci);
            Probability7.Text = Probability[7].ToString("g", ManagedMatrix.ci);
            Probability8.Text = Probability[8].ToString("g", ManagedMatrix.ci);
            Probability9.Text = Probability[9].ToString("g", ManagedMatrix.ci);

            ScoreBox.Text = String.Format("{0}%", (Probability[Classification] * 100).ToString("00.00", ManagedMatrix.ci));
            ClassificationBox.Text = String.Format("{0}", Classification.ToString("D", ManagedMatrix.ci));
        }

        void CleanUpUI()
        {
            Digit.Dispose();
        }

        void InitializeNetwork()
        {
            cnn = new ManagedCNN();

            cnn.AddLayer(new ManagedLayer());
            cnn.AddLayer(new ManagedLayer(6, 5));
            cnn.AddLayer(new ManagedLayer(2));
            cnn.AddLayer(new ManagedLayer(12, 5));
            cnn.AddLayer(new ManagedLayer(2));

            // Setup layer 1 - convolution layer
            cnn.LoadFeatureMap(BaseDirectory, "Layer02FeatureMap", 1, 5, 5, 1, 6);
            cnn.LoadFeatureMapBias(BaseDirectory, "Layer02Bias", 1, 6);

            // Setup layer 3 - convolution layer
            cnn.LoadFeatureMap(BaseDirectory, "Layer04FeatureMap", 3, 5, 5, 6, 12);
            cnn.LoadFeatureMapBias(BaseDirectory, "Layer04Bias", 3, 12);

            // Load Network Weights
            cnn.LoadNetworkWeights(BaseDirectory, "NetworkWeights", 192, 10);
            cnn.LoadNetworkBias(BaseDirectory, "NetworkBias", 10);
        }

        void CleanUpNetwork()
        {
            cnn.Free();
        }

        private void DeepLearnUI_FormClosing(object sender, FormClosingEventArgs e)
        {
            CleanUpNetwork();
            CleanUpUI();
        }

        private void Draw(Bitmap bitmap, int x, int y, bool erase = false)
        {
            SolidBrush redBrush = new SolidBrush(Color.Red);

            using (var g = Graphics.FromImage(bitmap))
            {
                var brush = erase ? new SolidBrush(Color.Black) : new SolidBrush(Color.White);

                var size = IsDrawing ? 16 : 32;

                g.FillEllipse(brush, x, y, size, size);
            }

            CopyBitmap(bitmap);
        }

        private void ClearBitmap(Bitmap bitmap)
        {
            using (var g = Graphics.FromImage(bitmap))
            {
                g.Clear(Color.Black);
            }

            CopyBitmap(bitmap);
        }

        public void CopyBitmap(Bitmap bitmap)
        {
            if (DigitBox.Image != null)
                DigitBox.Image.Dispose();

            if (bitmap != null)
            {
                DigitBox.Image = bitmap.Clone(new Rectangle(0, 0, bitmap.Width, bitmap.Height), System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            }
        }

        public void DrawActivationMap(int layer, int map)
        {
            if (IsActivated)
            {
                var bitmap = Activation.Get(cnn, layer, map);

                if (ActivationMap.Image != null)
                    ActivationMap.Image.Dispose();

                if (bitmap != null)
                {
                    ActivationMap.Image = bitmap.Clone(new Rectangle(0, 0, bitmap.Width, bitmap.Height), System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                    bitmap.Dispose();
                }
            }
        }

        public void DrawFeatureMap(int layer, int i, int j)
        {
            if (IsActivated)
            {
                var bitmap = Feature.Get(cnn, layer, i, j);

                if (FeatureMap.Image != null)
                    FeatureMap.Image.Dispose();

                if (bitmap != null)
                {
                    FeatureMap.Image = bitmap.Clone(new Rectangle(0, 0, bitmap.Width, bitmap.Height), System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                    bitmap.Dispose();
                }
            }
        }

        private void ButtonClassify_Click(object sender, EventArgs e)
        {
            // Classify image
            Classify.Bitmap(Digit, cnn, ref Classification, ref Probability);

            CopyClassification();

            if (IsActivated)
            {
                var layer = NetworkLayers.SelectedIndex;
                var map = ActivationMapsScroll.Value;

                DrawActivationMap(layer, map);

                if (cnn.Layers[layer].Type == LayerTypes.Convolution)
                {
                    var i = FeatureMapI.Value;
                    var j = FeatureMapJ.Value;

                    FeatureMapX.Text = cnn.Layers[layer].FeatureMap.x.ToString("D", ManagedMatrix.ci);
                    FeatureMapY.Text = cnn.Layers[layer].FeatureMap.y.ToString("D", ManagedMatrix.ci);
                    FeatureMapIText.Text = i.ToString("D", ManagedMatrix.ci);
                    FeatureMapJText.Text = j.ToString("D", ManagedMatrix.ci);

                    DrawFeatureMap(layer, i, j);
                }
            }

            IsActivated = true;
        }

        private void DigitBox_MouseMove(object sender, MouseEventArgs e)
        {
            if (IsDrawing || IsErasing)
            {
                Draw(Digit, e.X, e.Y, IsErasing);
            }
        }

        private void DigitBox_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
            {
                IsErasing = true;
                IsDrawing = false;
            }
            else
            {
                IsErasing = false;
                IsDrawing = true;
            }

            Draw(Digit, e.X, e.Y, IsErasing);
        }

        private void DigitBox_MouseUp(object sender, MouseEventArgs e)
        {
            IsDrawing = false;
            IsErasing = false;
        }

        private void ClearButton_Click(object sender, EventArgs e)
        {
            ClearBitmap(Digit);
        }

        private void LoadButton_Click(object sender, EventArgs e)
        {
            if (OpenImageDialog.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    var src = new Bitmap(OpenImageDialog.FileName);
                    var scaled = Classify.Resize(src, DigitBox.Width, DigitBox.Height);

                    if (Digit != null)
                    {
                        Digit.Dispose();
                        Digit = scaled.Clone(new Rectangle(0, 0, scaled.Width, scaled.Height), System.Drawing.Imaging.PixelFormat.Format24bppRgb);

                        CopyBitmap(Digit);
                    }

                    src.Dispose();
                    scaled.Dispose();
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error: Could not read file from disk. Original error: " + ex.Message);
                }
            }
        }

        private void NetworkLayers_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (IsActivated)
            {
                var layer = NetworkLayers.SelectedIndex;

                ActivationMapsScroll.Enabled = true;
                ActivationMapsScroll.Value = 0;
                ActivationMapsScroll.Minimum = 0;
                ActivationMapsScroll.Maximum = cnn.Layers[layer].Activation.i - 1;

                var map = ActivationMapsScroll.Value;

                ActivationX.Text = cnn.Layers[layer].Activation.x.ToString("D", ManagedMatrix.ci);
                ActivationY.Text = cnn.Layers[layer].Activation.y.ToString("D", ManagedMatrix.ci);
                ActivationMapsTextBox.Text = map.ToString("D", ManagedMatrix.ci);

                DrawActivationMap(layer, map);

                if (cnn.Layers[layer].Type != LayerTypes.Convolution)
                {
                    FeatureMapPanel.Hide();
                    FeatureMapJ.Enabled = false;
                    FeatureMapI.Enabled = false;
                }
                else
                {
                    FeatureMapPanel.Show();

                    FeatureMapJ.Enabled = true;
                    FeatureMapI.Enabled = true;

                    FeatureMapJ.Value = 0;
                    FeatureMapJ.Minimum = 0;
                    FeatureMapJ.Maximum = cnn.Layers[layer].FeatureMap.j - 1;

                    FeatureMapI.Value = 0;
                    FeatureMapI.Minimum = 0;
                    FeatureMapI.Maximum = cnn.Layers[layer].FeatureMap.i - 1;

                    var i = FeatureMapI.Value;
                    var j = FeatureMapJ.Value;

                    FeatureMapX.Text = cnn.Layers[layer].FeatureMap.x.ToString("D", ManagedMatrix.ci);
                    FeatureMapY.Text = cnn.Layers[layer].FeatureMap.y.ToString("D", ManagedMatrix.ci);
                    FeatureMapIText.Text = i.ToString("D", ManagedMatrix.ci);
                    FeatureMapJText.Text = j.ToString("D", ManagedMatrix.ci);

                    DrawFeatureMap(layer, i, j);
                }
            }
        }

        private void ActivationMapsScroll_ValueChanged(object sender, EventArgs e)
        {
            if (IsActivated)
            {
                var layer = NetworkLayers.SelectedIndex;
                var map = ActivationMapsScroll.Value;

                ActivationMapsTextBox.Text = map.ToString("D", ManagedMatrix.ci);

                ActivationX.Text = cnn.Layers[layer].Activation.x.ToString("D", ManagedMatrix.ci);
                ActivationY.Text = cnn.Layers[layer].Activation.y.ToString("D", ManagedMatrix.ci);

                DrawActivationMap(layer, map);
            }
        }

        private void FeatureMapJ_ValueChanged(object sender, EventArgs e)
        {
            if (IsActivated)
            {
                var layer = NetworkLayers.SelectedIndex;

                if (cnn.Layers[layer].Type == LayerTypes.Convolution)
                {
                    var i = FeatureMapI.Value;
                    var j = FeatureMapJ.Value;

                    FeatureMapJText.Text = j.ToString("D", ManagedMatrix.ci);

                    DrawFeatureMap(layer, i, j);
                }
            }
        }

        private void FeatureMapI_ValueChanged(object sender, EventArgs e)
        {
            if (IsActivated)
            {
                var layer = NetworkLayers.SelectedIndex;

                if (cnn.Layers[layer].Type == LayerTypes.Convolution)
                {
                    var i = FeatureMapI.Value;
                    var j = FeatureMapJ.Value;

                    FeatureMapIText.Text = i.ToString("D", ManagedMatrix.ci);

                    DrawFeatureMap(layer, i, j);
                };
            }
        }
    }
}
